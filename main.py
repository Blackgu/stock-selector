import json
from settings import logger
from agents.intent_agent import classify_intent
from agents.task_decompose_agent import decompose_task
from tools.polygon_tools import get_stock_tickers
from tools.ticker_financial_tools import get_daily_bar, cagr, max_drawdown_and_recovery, annual_volatility
from polygon.enums import TickerMarketType
from datetime import date, timedelta

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

    start = date.today() - timedelta(days=365*1+15)
    end = date.today()
    px = get_daily_bar("META", start, end, adjusted=False)["c"]

    metrics = {
        "CAGR": cagr(px),
        "MaxDrawdown": max_drawdown_and_recovery(px)[0],
        "DrawdownDuration_days": max_drawdown_and_recovery(px)[1],
        "RecoveryTime_days": max_drawdown_and_recovery(px)[2],
        "Volatility_ann": annual_volatility(px, use_log=True)
    }
    print(metrics)