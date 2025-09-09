import json
from settings import logger
from agents.intent_agent import classify_intent
from agents.task_decompose_agent import decompose_task
from tools.ticker_query_tools import get_stock_tickers
from tools.ticker_financial_action_tools import (get_daily_bar, cagr, max_drawdown_and_recovery,
                                                 annual_volatility)
from polygon.enums import TickerMarketType, StockFinancialsTimeframe, StockFinancialsSortKey
from datetime import date, timedelta
from tools.ticker_financial_health_tools import get_revenues, get_cfo, get_cash
from tools.ticker_query_tools import get_stock_tickers

if __name__ == '__main__':
    # intent = classify_intent("TSLA这支股票怎么样？")
    # logger.info(intent)
    # sub_tasks_response = decompose_task(intent)
    # logger.info(sub_tasks_response)

    # tickers = get_stock_tickers(ticker="BABA",
    #                             market=TickerMarketType.STOCKS,
    #                             limit=10,
    #                             max_pages=1)
    # print(json.dumps(tickers, indent=2))

    # start = date.today() - timedelta(days=365*5+15)
    # end = date.today()
    # df = get_daily_bar("LULU", start, end, adjusted=True)
    # px = df["c"]
    # metrics = {
    #     "CAGR": cagr(df),
    #     "MaxDrawdown": max_drawdown_and_recovery(px)[0],
    #     "DrawdownDuration_days": max_drawdown_and_recovery(px)[1],
    #     "RecoveryTime_days": max_drawdown_and_recovery(px)[2],
    #     "t_peak": max_drawdown_and_recovery(px)[3],
    #     "t_trough": max_drawdown_and_recovery(px)[4],
    #     "t_recover": max_drawdown_and_recovery(px)[5],
    #     "still_in_drawdown": max_drawdown_and_recovery(px)[6],
    #     "Volatility_ann": annual_volatility(px, use_log=True)
    # }
    # print(metrics)

    # 例：以“短期债务口径”为例 —— 目标标签 DebtCurrent，
    # 并给出几条常见同义：短借、当前到期的长债、IFRS 的 BorrowingsCurrent。
    cik = "0001577552"  # Apple Inc.
    df = get_revenues(cik=cik)
    # df.to_excel("apple_debt_current.xlsx")
    print(df.tail(10))