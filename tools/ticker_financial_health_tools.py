import requests
import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

# —— 必填：SEC 强制要求可识别的 User-Agent（你的姓名/公司 + 联系方式）——
SEC_HEADERS = {
    "User-Agent": "blackgu1985831@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

# =========================
# 基础工具
# =========================
def get_company_facts(cik: str) -> Dict:
    """
    读取 SEC companyfacts：
    https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
    cik 可不带前导 0；内部会补齐到 10 位。
    """
    cik_padded = str(int(cik)).zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()

def _pick_unit_key(units: Dict[str, list],
                   prefer=("USD", "USDm", "USDth", "EUR", "GBP", "pure")) -> Optional[str]:
    if not units:
        return None
    for u in prefer:
        if u in units:
            return u
    return next(iter(units), None)

def _unit_scale(unit_key: str) -> float:
    # 这里只做“数量级换算”，不同币种不自动汇率换算（需要的话可外部再统一到 USD）
    return {
        "USD": 1, "USDm": 1_000_000, "USDth": 1_000,
        "EUR": 1, "GBP": 1, "pure": 1
    }.get(unit_key, 1)

def _extract_concept(facts_json: Dict, taxonomy: str, tag: str,
                     prefer_units=("USD", "USDm", "USDth")) -> pd.DataFrame:
    """
    从指定 taxonomy（如 'us-gaap', 'ifrs-full', 'aapl' 扩展等）抽取某标签数据。
    返回列：['fy','fp','form','end','filed','unit_key','taxonomy','tag','value']
    """
    node = facts_json.get("facts", {}).get(taxonomy, {}).get(tag)
    if not node:
        return pd.DataFrame()

    units = node.get("units", {})
    unit_key = _pick_unit_key(units, prefer_units)
    if not unit_key:
        return pd.DataFrame()

    df = pd.DataFrame(units[unit_key])  # 常见列：fy, fp, form, end, val, accn, filed, frame...
    if df.empty:
        return df

    df = df.copy()
    df["value"] = pd.to_numeric(df.get("val"), errors="coerce") * _unit_scale(unit_key)
    df["unit_key"] = unit_key
    df["taxonomy"] = taxonomy
    df["tag"] = tag
    # 只保留核心列（其他有需要可再加）
    keep = ["fy","fp","form","end","filed","unit_key","taxonomy","tag","value"]
    return df[[c for c in keep if c in df.columns]]

def _dedupe_latest(df: pd.DataFrame,
                   keys=("fy","fp","end","form","taxonomy","tag")) -> pd.DataFrame:
    """同一期多次披露时，按 filed 取最新。"""
    if df.empty:
        return df
    df = df.copy()
    df["filed"] = pd.to_datetime(df.get("filed"), errors="coerce")
    df = df.sort_values("filed").drop_duplicates(list(keys), keep="last")
    return df

def _similarity(a: str, b: str) -> float:
    """标签名相似度（0~1），用于模糊匹配."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def _regex_from_tag(tag: str, synonyms: Optional[List[str]] = None) -> re.Pattern:
    """
    从目标标签构造一个较宽松的正则（大小写不敏感；允许中划线/下划线/驼峰差异），
    并合入用户提供的同义词。
    例如 DebtCurrent -> r'(?i)Debt[_-]?Current$' 等。
    """
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", tag)
    base = r"[_\- ]?".join(parts) if parts else re.escape(tag)
    alts = [fr"{base}$"]
    if synonyms:
        for s in synonyms:
            p = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", s)
            alts.append(fr"{'[_\- ]?'.join(p) if p else re.escape(s)}$")
    pattern = r"(?i)(" + "|".join(alts) + r")"
    return re.compile(pattern)

def _contains_any(s: str, kws: Optional[List[str]]) -> bool:
    if not kws:
        return True
    s = s.lower()
    return any(k.lower() in s for k in kws)

def _contains_none(s: str, kws: Optional[List[str]]) -> bool:
    if not kws:
        return True
    s = s.lower()
    return all(k.lower() not in s for k in kws)

def _gather_candidates(facts_json: Dict,
                       target_tag: str,
                       synonyms: Optional[List[str]],
                       prefer_units: Tuple[str, ...],
                       only_forms: Optional[List[str]],
                       min_sim: float = 0.66,
                       required_keywords: Optional[List[str]] = None,
                       deny_keywords: Optional[List[str]] = None) -> List[pd.DataFrame]:
    """
    收集所有可能的候选数据（精确匹配 + 模糊匹配），返回 DataFrame 列表。

    参数：
        facts_json: 公司财务数据的JSON对象，包含所有已披露的财务概念
        target_tag: 目标财务标签名称（如"DebtCurrent"）
        synonyms: 目标标签的同义词列表，用于扩展匹配范围
        prefer_units: 偏好的货币单位元组，按优先级排序
        only_forms: 限定的财务报表类型列表（如["10-K", "10-Q"]），为空则不限制
        min_sim: 最小相似度阈值，用于模糊匹配，默认0.66
        required_keywords: 标签必须包含的关键词列表，为空则不限制
        deny_keywords: 标签不能包含的关键词列表，为空则不限制

    返回：
        包含所有匹配到的财务数据的DataFrame列表，每个DataFrame代表一组匹配的财务数据
    """
    candidates: List[pd.DataFrame] = []

    # 1) 精确匹配：优先在标准命名空间中查找精确匹配项
    for tax in ("us-gaap", "ifrs-full"):
        df = _extract_concept(facts_json, tax, target_tag, prefer_units)
        if not df.empty:
            if only_forms:
                df = df[df["form"].isin(only_forms)]
            candidates.append(_dedupe_latest(df))

    # 所有命名空间里“精确同名”的其他匹配
    facts = facts_json.get("facts", {}) or {}
    for tax, concepts in facts.items():
        if tax in ("us-gaap", "ifrs-full"):
            continue
        if target_tag in concepts:
            df = _extract_concept(facts_json, tax, target_tag, prefer_units)
            if not df.empty:
                if only_forms:
                    df = df[df["form"].isin(only_forms)]
                candidates.append(_dedupe_latest(df))

    # 2) 模糊/同义词匹配：跨所有命名空间进行模糊匹配
    regex = _regex_from_tag(target_tag, synonyms)
    for tax, concepts in facts.items():
        for tag in concepts.keys():

            # 关键词门槛：必须包含 required_keywords，且不包含 deny_keywords
            if not _contains_any(tag, required_keywords):  # e.g. debt/borrow/loan
                continue
            if not _contains_none(tag, deny_keywords):  # e.g. asset
                continue

            # 正则先筛一层
            if not (regex.search(tag) or _similarity(target_tag, tag) >= min_sim):
                continue
            df = _extract_concept(facts_json, tax, tag, prefer_units)
            if not df.empty:
                if only_forms:
                    df = df[df["form"].isin(only_forms)]
                candidates.append(_dedupe_latest(df))

    return [c for c in candidates if not c.empty]

def _narrow_candidates_by_form(candidates: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    自动收窄候选的表单类型：
      - 若整体候选中出现 10-K/10-Q，则只保留 10-K/10-Q（含 /A 修订）；
      - 否则若出现 20-F/6-K，则只保留 20-F/6-K（含 /A 修订）；
      - 否则不变。
    """
    if not candidates:
        return candidates

    all_forms = pd.concat(candidates, ignore_index=True)["form"].dropna().astype(str).unique().tolist()
    present = set(all_forms)

    us_core = {"10-K", "10-K/A", "10-Q", "10-Q/A"}
    fpi_core = {"20-F", "20-F/A", "6-K", "6-K/A"}

    if {"10-K", "10-Q"} & present:
        narrowed = [df[df["form"].isin(us_core)] for df in candidates]
    elif {"20-F", "6-K"} & present:
        narrowed = [df[df["form"].isin(fpi_core)] for df in candidates]
    else:
        narrowed = candidates

    return [df for df in narrowed if not df.empty]

def _score_candidate(df: pd.DataFrame) -> Tuple[int, pd.Timestamp, float]:
    """
    为候选打分：
      1) 观测数越多越好；
      2) 最新 filed 越晚越好；
      3) 数值规模（绝对值之和）越大（或更稳定）越好（此处用幅度近似衡量）。
    """
    n_obs = len(df)
    latest = pd.to_datetime(df["filed"]).max() if "filed" in df.columns else pd.NaT
    magnitude = df["value"].abs().sum(skipna=True)
    return n_obs, latest, magnitude

def _pick_best(candidates: List[pd.DataFrame]) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame()
    scored = sorted(
        ((*_score_candidate(c), i) for i, c in enumerate(candidates)),
        key=lambda x: (x[0], x[1], x[2]),
        reverse=True
    )
    return candidates[scored[0][-1]]

def get_concept_series_robust(
    cik: str,
    target_tag: str,
    *,
    synonyms: Optional[List[str]] = None,
    prefer_units: Tuple[str, ...] = ("USD", "USDm", "USDth"),
    only_forms: Optional[List[str]] = None,
    min_similarity: float = 0.66,
    required_keywords: Optional[List[str]] = None,
    deny_keywords: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    传入标签名，返回“最稳”的时间序列（自动在 us-gaap / ifrs-full / 公司扩展里找相近标签）。
    选择规则：观测数最多 → 提交时间最新 → 数值规模更稳定（绝对值之和更大作为近似）。

    参数：
      - cik: 公司 CIK（可不带前导0）
      - target_tag: 目标标签名（如 "DebtCurrent"）
      - synonyms: 同义/近义标签（如 ["ShortTermBorrowings","LongTermDebtCurrent","BorrowingsCurrent"]）
      - prefer_units: 单位优先级（默认 USD > USDm > USDth）
      - only_forms: 仅保留特定表单类型（如 ["10-K","10-Q"]）
      - min_similarity: 模糊匹配的最小相似度阈值（0~1）

    返回：
      DataFrame（按期末日与 filed 排序），列包含：
      ['fy','fp','form','end','filed','unit_key','taxonomy','tag','value']
    """
    facts = get_company_facts(cik)
    cands = _gather_candidates(
        facts_json=facts,
        target_tag=target_tag,
        synonyms=synonyms,
        prefer_units=prefer_units,
        only_forms=only_forms,
        min_sim=min_similarity,
        required_keywords=required_keywords,
        deny_keywords=deny_keywords
    )

    # 按优先级自动收窄 10-K/10-Q 或 20-F/6-K
    _narrow_candidates_by_form(cands)

    best = _pick_best(cands)
    if best.empty:
        return best
    # 排序 & 重排列
    best = best.sort_values(["end", "filed"]).reset_index(drop=True)
    # ----------------------------------------------------------------------
    # fy: 财年（Fiscal Year），整数，例如 2024。同一公司的 fy 与 end/fp 搭配唯一标识一期财报。
    # fp: 财务期间（Fiscal Period），常见值：Q1、Q2、Q3、Q4、FY（全年）。少数 FPI/IFRS 报告也会出现 H1 等。
    # form: 申报表类型，例如 10-K（年报）、10-Q（季报）、10-K/A（年报更正）、20-F、6-K、8-K 等。
    # end: 期末日（或即时日）。对 duration 概念是区间的 endDate，对 instant 概念是 instant；统一表示成 YYYY-MM-DD。
    # filed: 申报材料的提交日期（filing date），YYYY-MM-DD。
    # unit_key: 取数所用的单位键，例如 USD、USDm、USDth、shares、pure 等。
    # taxonomy: 概念所属命名空间：us-gaap、ifrs-full，或公司的扩展命名空间（例如 aapl 之类）。
    # tag: 概念名（Concept/Element），如 DebtCurrent、RevenueFromContractWithCustomerExcludingAssessedTax。
    # value: 观测值（已按 unit_key 做数量级换算后的数值）。
    # ----------------------------------------------------------------------
    cols = ["fy","fp","form","end","filed","unit_key","taxonomy","tag","value"]
    return best[[c for c in cols if c in best.columns]]

def get_revenues(cik: str) -> pd.Series:

    REVENUE_TAGS = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
        "Revenue",
        "RevenueFromContractsWithCustomers",
        "RevenueFromSaleOfGoods",
        "RevenueFromRenderingOfServices",
    ]

    return get_concept_series_robust(
        cik=cik,
        target_tag="Revenues",
        synonyms=REVENUE_TAGS,
        min_similarity=0.80,
        only_forms=["10-K","10-Q","20-F","6-K"],
        required_keywords=["revenue", "sales"],
        deny_keywords=["deferred", "contractliability", "otherincome", "interest", "dividend", "gain"]
    )

def get_gross_profit(cik: str) -> pd.Series:
    return get_concept_series_robust(
        cik=cik,
        target_tag="GrossProfit",
        synonyms=["GrossProfit"],
        min_similarity=0.80,
        only_forms=["10-K", "10-Q"],
        required_keywords=["gross", "profit"],
        deny_keywords=["net", "operating", "comprehensive"]
    )

def get_cfo(cik: str) -> pd.Series:

    CFO_TAGS = [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        "NetCashProvidedByUsedInOperatingActivitiesDiscontinuedOperations",
        "NetCashFlowsFromUsedInOperatingActivities",
    ]

    return get_concept_series_robust(
        cik=cik,
        target_tag="NetCashProvidedByUsedInOperatingActivities",
        synonyms=CFO_TAGS,
        min_similarity=0.80,
        only_forms=["10-K", "10-Q"],
        required_keywords=["cash", "operating"],
        deny_keywords=["investing", "financing", "capex", "interestpaid", "dividend"]
    )

def get_capex(cik: str) -> pd.Series:
    CAPEX_TAGS = [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpenditures",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsForPropertyPlantAndEquipment",
        "PurchaseOfPropertyPlantAndEquipment",
    ]
    return get_concept_series_robust(
        cik=cik,
        target_tag="PaymentsToAcquirePropertyPlantAndEquipment",
        synonyms=CAPEX_TAGS,
        min_similarity=0.80,
        only_forms=["10-K", "10-Q"],
        required_keywords=["property", "plant", "equipment", "capitalexpenditure", "capex"],
        deny_keywords=["proceeds", "sale", "disposal", "acquisitionofbusiness"]
    )

def get_ebit(cik: str) -> pd.Series:
    EBIT_TAGS = [
        "OperatingIncomeLoss",
        "ProfitLossFromOperatingActivities",
        "OperatingProfit",
        "EarningsBeforeInterestAndTaxes",
    ]

    return get_concept_series_robust(
        cik=cik,
        target_tag="operatingIncomeLoss",
        synonyms=EBIT_TAGS,
        min_similarity=0.80,
        only_forms=["10-K", "10-Q"],
        required_keywords=["operating", "income", "profit", "ebit"],
        deny_keywords=["cash", "beforeincometaxes", "ebt", "netincome", "comprehensive", "nonoperating"]
    )

def get_d_and_a(cik: str) -> pd.Series:
    D_AND_A_TAGS = [
        "DepreciationAndAmortization",
        "AmortizationOfIntangibleAssets",
        "Depreciation",
        "DepreciationAndAmortisationExpense",
    ]
    return get_concept_series_robust(
        cik=cik,
        target_tag="DepreciationDepletionAndAmortization",
        synonyms=D_AND_A_TAGS,
        min_similarity=0.80,
        only_forms=["10-K", "10-Q"],
        required_keywords=["depreciation", "amort"],
        deny_keywords=["accumulated", "capitalized", "capitalised", "paid", "impairment"]
    )

def get_ebitda(cik: str) -> pd.Series:
    ebit_df = get_ebit(cik)
    d_and_a_df = get_d_and_a(cik)
    return (ebit_df if pd.notna(ebit_df) else np.nan) + (d_and_a_df if pd.notna(d_and_a_df) else 0)

def get_cash(cik: str) -> pd.Series:
    CASH_TAGS = [
        "CashAndCashEquivalentsAtCarryingValueIncludingDiscontinuedOperations",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "RestrictedCashAndCashEquivalents",
    ]

    return get_concept_series_robust(
        cik=cik,
        target_tag="CashAndCashEquivalentsAtCarryingValue",
        synonyms=CASH_TAGS,
        min_similarity=0.80,
        only_forms=["10-K", "10-Q"],
        required_keywords=["cash", "equivalent"],
        deny_keywords=["asset", "investment", "shortterminvestment"]
    )

def get_st_debt(cik: str) -> pd.Series:
    DEBT_CURRENT = [
        "DebtCurrent",
        "ShortTermBorrowings",
        "LongTermDebtCurrent",
        "CommercialPaper", "CommercialPaperAtCarryingValue",
        "NotesPayableCurrent",
        "OtherShortTermBorrowings", "OtherLongTermDebtCurrent",
    ]

    return get_concept_series_robust(
        cik=cik,
        target_tag="DebtCurrent",
        synonyms=DEBT_CURRENT,
        min_similarity=0.80,
        only_forms=["10-K", "10-Q"],
        required_keywords=["debt","borrow","loan","bond","note","debenture","commercialpaper"],
        deny_keywords=["asset","lease"])

def get_lt_debt(cik: str) -> pd.Series:
    DEBT_LONG_TAGS = [
        "LongTermDebtNoncurrent",
        "LongtermBorrowings",
    ]
    return get_concept_series_robust(
        cik=cik,
        target_tag="LongTermDebtNoncurrent",
        synonyms=DEBT_LONG_TAGS,
        min_similarity=0.80,
        only_forms=["10-K", "10-Q"],
        required_keywords=["debt","borrow","loan","bond","note","debenture"],
        deny_keywords=["asset","lease"])

def get_interest(cik: str) -> pd.Series:
    INTEREST_TAGS = [
        "InterestExpense",
        "InterestExpenseOperating",
        "InterestExpenseNonoperating",
        "FinanceCosts",
    ]

    return get_concept_series_robust(
        cik=cik,
        target_tag="InterestExpense",
        synonyms=INTEREST_TAGS,
        min_similarity=0.80,
        only_forms=["10-K", "10-Q"],
        required_keywords=["interest", "financecost", "financecosts", "finance"],
        deny_keywords=["income", "net", "paid", "capitalized", "capitalised", "receive", "receivable"]
    )

# rev = pick(fs, ("income_statement", "revenues"),
#            ("income_statement", "salesRevenueNet"),
#            ("income_statement", "revenueFromContractWithCustomerExcludingAssessedTax"))
# gp = pick(fs, ("income_statement", "grossProfit"))
# cfo = pick(fs, ("cash_flow_statement", "netCashProvidedByUsedInOperatingActivities"))
# capex = pick(fs, ("cash_flow_statement", "paymentsToAcquirePropertyPlantAndEquipment"),
#              ("cash_flow_statement", "purchaseOfPropertyAndEquipment"))
# ebit = pick(fs, ("income_statement", "operatingIncomeLoss"),
#             ("income_statement", "incomeLossFromContinuingOperationsBeforeIncomeTaxes"))
# d_and_a = pick(fs, ("income_statement", "depreciationAndAmortization"),
#                ("cash_flow_statement", "depreciationDepletionAndAmortization"))
# ebitda = (ebit if pd.notna(ebit) else np.nan) + (d_and_a if pd.notna(d_and_a) else 0)
# cash = pick(fs, ("balance_sheet", "cashAndCashEquivalentsAtCarryingValue"),
#             ("balance_sheet", "cashCashEquivalentsAndShortTermInvestments"))
# st_debt = pick(fs, ("balance_sheet", "debtCurrent"),
#                ("balance_sheet", "shortTermBorrowings"))
# lt_debt = pick(fs, ("balance_sheet", "longTermDebtNoncurrent"),
#                ("balance_sheet", "longTermDebt"))
# interest = pick(fs, ("income_statement", "interestExpense"))
# period_end = it.get("end_date") or it.get("fiscal_period_end_date")