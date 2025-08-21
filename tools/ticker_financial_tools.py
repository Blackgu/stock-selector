import os
import pytz
import pandas as pd, numpy as np
from polygon import StocksClient
from datetime import date
from polygon.enums import Timespan
from datetime import datetime, timedelta

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

stocks_client = StocksClient(api_key=POLYGON_API_KEY)

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
            and aggregate_bars_response.get("status") != "OK"):
        raise Exception(f"Error fetching aggregate_bars: {aggregate_bars_response.get('status')}，"
                        f"case: {aggregate_bars_response.get('error')}")

    aggregate_bars = aggregate_bars_response.get("results") or []
    rows = [{"t": pd.to_datetime(agg.get("t"), unit="ms"), "c": agg.get("c")} for agg in aggregate_bars]
    df = pd.DataFrame(rows).set_index("t").sort_index()
    return df

def cagr(px: pd.Series):
    """
    计算年化复合增长率(CAGR)

    Args:
        px (pd.Series): 价格序列，索引为时间

    Returns:
        float: 年化复合增长率
    """
    years = (px.index[-1] - px.index[0]).days / 365.25
    return (px.iloc[-1] / px.iloc[0]) ** (1 / years) - 1

def max_drawdown_and_recovery(px: pd.Series):
    """
    计算最大回撤及恢复时间

    Args:
        px (pd.Series): 价格序列，索引为时间

    Returns:
        tuple: (最大回撤, 峰值到新高总时长, 回撤谷底到新高时长)
               最大回撤为负值，时长单位为天，如果未创新高则对应时长为 NaN
    """
    peak = px.cummax()
    dd = px/peak - 1.0
    mdd = dd.min()                                # 最大回撤（负值）
    t_trough = dd.idxmin()                        # 谷底时间
    t_peak = px.loc[:t_trough].idxmax()           # 对应峰值时间
    # 恢复到此前峰值的时间（若未恢复则为 NaN）
    after = px.loc[t_trough:]
    rec_idx = after[after >= px.loc[t_peak]].index
    t_recover = rec_idx[0] if len(rec_idx) else None
    recovery_days = (t_recover - t_trough).days if t_recover else np.nan
    duration_days = (t_recover - t_peak).days if t_recover else np.nan  # 峰→新高总时长
    return float(mdd), duration_days, recovery_days

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
