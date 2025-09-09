import os
import pytz
import pandas as pd, numpy as np
from polygon import StocksClient, ReferenceClient
from datetime import date
from polygon.enums import Timespan
from datetime import datetime, timedelta

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

stocks_client = StocksClient(api_key=POLYGON_API_KEY)
reference_client = ReferenceClient(api_key=POLYGON_API_KEY)

def normalize_end_to_prev_session(end_iso: str) -> str:
    # 将 end（YYYY-MM-DD）限制到美东今天（含）以前的上一天；周末/假日用交易日历更稳妥
    eastern_today = datetime.now(pytz.timezone("America/New_York")).date()
    end_date = datetime.fromisoformat(end_iso).date()
    if end_date >= eastern_today:
        end_date = end_date - timedelta(days=1)
    return end_date.isoformat()

def get_daily_bar(ticker: str, start: date, end: date, adjusted=True) -> pd.Series:
    aggregate_bars_response = stocks_client.get_aggregate_bars(
        symbol = ticker,
        from_date = start.isoformat(),
        to_date = normalize_end_to_prev_session(end.isoformat()),
        timespan = Timespan.DAY,
        adjusted = adjusted
    )

    if (aggregate_bars_response.get("status")
            and aggregate_bars_response.get("status") == "ERROR"):
        raise Exception(f"Error fetching aggregate_bars: {aggregate_bars_response.get('status')}，"
                        f"case: {aggregate_bars_response.get('error')}")

    aggregate_bars = aggregate_bars_response.get("results") or []
    rows = [{"t": pd.to_datetime(agg.get("t"), unit="ms"), "c": agg.get("c")} for agg in aggregate_bars]

    df = pd.DataFrame(rows)
    if df is None or len(df) == 0:
        raise ValueError("No data returned")

    df = df.sort_values("t").copy()
    df["dt"] = (pd.to_datetime(df["t"], unit="ms", utc=True)
                .dt.tz_convert("America/New_York"))
    df = df.set_index("dt")
    return df

def cagr(df: pd.DataFrame):
    """
    计算年化复合增长率(CAGR)

    Args:
        df (pd.DataFrame): 包含股票价格数据的DataFrame，必须包含列"c"(收盘价)和"dt"(日期时间)

    Returns:
        float: 年化复合增长率
    """
    # 提取首尾收盘价
    first_close_price = df["c"].iloc[0]
    last_close_price = df["c"].iloc[-1]

    # 计算时间跨度
    start_date = df.index[0].date()
    end_date = df.index[-1].date()
    years = (end_date - start_date).days / 365.25
    if years <= 0:
        raise ValueError(f"Non-positive years: {years}")

    return (last_close_price / first_close_price) ** (1 / years) - 1


def max_drawdown_and_recovery(px: pd.Series, tol=1e-9):
    """
    计算最大回撤及恢复时间

    Args:
        px (pd.Series): 价格序列，索引为时间
        tol (float): 容忍度阈值，用于判断是否恢复到峰值水平，默认为1e-9

    Returns:
        tuple: (最大回撤, 峰值到新高总时长, 回撤谷底到新高时长, 峰值时间, 谷底时间, 恢复时间, 是否仍在回撤中)
               最大回撤为负值，时长单位为天，如果未创新高则对应时长为 NaN
    """
    # 计算累计最大值序列和回撤序列
    peak = px.cummax()
    dd = px/peak - 1.0
    mdd = dd.min()                                # 最大回撤（负值）
    t_trough = dd.idxmin()                        # 谷底时间
    t_peak = px.loc[:t_trough].idxmax()           # 对应峰值时间

    # 计算从谷底开始恢复到接近原峰值的时间点
    after = px.loc[t_trough:]
    target = px.loc[t_peak] * (1 - tol)
    rec_idx = after[after >= target].index

    # 判断是否已恢复到峰值水平
    if len(rec_idx):
        t_recover = rec_idx[0]
        recovery_days = (t_recover - t_trough).days
        duration_days = (t_recover - t_peak).days
        still_in_drawdown = False
    else:
        t_recover = None
        # 用"到样本末日"的时长作为"进行中的回撤时长"
        recovery_days = np.nan
        duration_days = (px.index[-1] - t_peak).days
        still_in_drawdown = True
    return float(mdd), duration_days, recovery_days, t_peak, t_trough, t_recover, still_in_drawdown


def annual_volatility(px: pd.Series, use_log=False):
    """
    计算年化波动率

    Args:
        px (pd.Series): 价格序列，索引为时间
        use_log (bool): 是否使用对数收益率计算，默认为False

    Returns:
        float: 年化波动率
    """
    r = np.log(px/px.shift(1)).dropna() if use_log else px.pct_change().dropna()
    return float(r.std() * np.sqrt(252))

    # for it in results:
    #     fs = it.get("financials", {})
    #     income_statement = fs.get("income_statement", {})         # 利润表
    #     cash_flow_statement = fs.get("cash_flow_statement", {})   # 现金流量表
    #     balance_sheet = fs.get("balance_sheet", {})               # 资产负债表
    #     comprehensive_income = fs.get("comprehensive_income", {}) # 综合收益
    #
    #     revenues = float(income_statement.get("revenues", {}).get("value"))
    #     gross_profit = float(income_statement.get("gross_profit", {}).get("value"))
    #     cfo = float(cash_flow_statement.get("net_cash_flow_from_operating_activities", {}).get("value"))
    #     capex = float(cash_flow_statement.get("", {}).get("value"))
    #     ebit = float(income_statement.get("operating_income_loss", {}).get("value"))
    #     d_and_a = float(income_statement.get("depreciation_and_amortization", {}).get("value"))
    #     ebitda = (ebit if pd.notna(ebit) else np.nan) + (d_and_a if pd.notna(d_and_a) else 0)
    #     cash = float(balance_sheet.get("cash", {}).get("value"))
    #     st_debt = float(balance_sheet.get("", {}).get("value"))
    #     lt_debt = float(balance_sheet.get("long_term_debt", {}).get("value"))
    #     interest = float(income_statement.get("interest_and_debt_expense", {}).get("value"))
    #     period_end = it.get("end_date") or it.get("fiscal_period_end_date")


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