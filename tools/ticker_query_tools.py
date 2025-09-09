import os
import re
from polygon import ReferenceClient, StocksClient
from polygon.enums import TickerSortType, SortOrder
from typing import (Dict, Iterator, Any, Optional,
                    Callable, Iterable, Union, Pattern, Set)

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# stocks_client = StocksClient(api_key=POLYGON_API_KEY)
reference_client = ReferenceClient(api_key=POLYGON_API_KEY)

# ---------------------- 数值类型过滤器 --------------------------
NumFunc = Callable[[Dict[str, Any]], Optional[float]]

class NumericFilter:
    path: Optional[str] = None
    func: Optional[NumFunc] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    eq_value: Optional[float] = None
    ne_value: Optional[float] = None
    missing_ok: bool = False

def get_num_from_path(data:dict[str, Any], path: str) -> Optional[float]:
    if path is None:
        return None

    value: Any = data
    if not isinstance(value, dict):
        return None

    for key in path.split("."):
        value = value.get(key)
        if value is None or not isinstance(value, dict):
            return None

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        if "value" in value and isinstance(value["value"], (int, float)):
            return float(value["value"])
    return None

def do_numeric_filter(data: dict[str, Any], filters: list[NumericFilter]) -> bool:
    if not filters:
        return True

    for f in filters:
        value: Optional[float]
        if f.func is not None:
            value = f.func(data)
        elif f.path is not None:
            value = get_num_from_path(data, f.path)
        else:
            return False

        if value is None:
            if not f.missing_ok:
                return False
            else:
                continue

        if f.min_value is not None and value < f.min_value:
            return False
        if f.max_value is not None and value > f.max_value:
            return False
    return True

# ---------------------- 股票筛选过滤器 --------------------------
TickerPred = Callable[[str, Dict[str, Any]], bool]

def get_ticker_filter(
    include_tickers: Optional[Iterable[str]],
    exclude_tickers: Optional[Iterable[str]],
    ticker_regex: Optional[Union[str, Pattern]],
    ticker_predicate: Optional[TickerPred],
):
    inc: Optional[Set[str]] = set(map(str.upper, include_tickers)) if include_tickers else None
    exc: Optional[Set[str]] = set(map(str.upper, exclude_tickers)) if exclude_tickers else None
    regex: Optional[Pattern] = re.compile(ticker_regex, re.IGNORECASE) if isinstance(ticker_regex, str) else ticker_regex

    def pass_ticker(ticker: str, data: Dict[str, Any]) -> bool:
        u_ticker = ticker.upper()
        if inc is not None and u_ticker not in inc:
            return False
        if exc is not None and u_ticker in exc:
            return False
        if regex is not None and not regex.search(ticker):
            return False
        if ticker_predicate is not None and not ticker_predicate(ticker, data):
            return False
        return True

    return pass_ticker

def stream_fetch_stock_tickers(
        ticker: Optional[str] = None,
        ticker_type: Optional[str] = None,
        cik: Optional[str] = None,
        market: Optional[str] = None,
        sort: str = "filing_date",
        order: str = "asc",
        limit: int = 1000,
        max_pages: Optional[int] = None,

        include_tickers: Optional[Iterable[str]] = None,
        exclude_tickers: Optional[Iterable[str]] = None,
        ticker_regex: Optional[Union[str, Pattern]] = None,
        ticker_predicate: Optional[TickerPred] = None,
        numeric_filters: Optional[list[NumericFilter]] = None
) -> Iterator[Dict[str, Any]]:
    pages = 0
    get_tickers_response = reference_client.get_tickers(
        limit = limit,
        all_pages = False,
        max_pages = max_pages,
        symbol = ticker,
        symbol_type = ticker_type,
        cik = cik,
        market = market,
        sort = sort,
        order = order,
    )

    if (get_tickers_response.get("status")
            and get_tickers_response.get("status") == "ERROR"):
        raise Exception(f"Error fetching tickers: {get_tickers_response.get('status')}，"
                        f"case: {get_tickers_response.get('error')}")

    while get_tickers_response:
        pages += 1
        for ticker in (get_tickers_response.get("results") or []):

            if not do_numeric_filter(ticker, numeric_filters):
                continue

            pass_ticker = get_ticker_filter(
                include_tickers,
                exclude_tickers,
                ticker_regex,
                ticker_predicate,
            )

            if pass_ticker(ticker.get("ticker"), ticker):
                yield ticker

        if max_pages is not None and pages >= max_pages:
            break

        ticker_page = reference_client.get_next_page(get_tickers_response)
        if not ticker_page:
            break

def get_stock_tickers(
        ticker: Optional[str] = None,
        ticker_type: Optional[str] = None,
        cik: Optional[str] = None,
        market: Optional[str] = None,
        sort: str = TickerSortType.TICKER,
        order: str = SortOrder.ASC,
        limit: int = 1000,
        max_pages: Optional[int] = None,

        include_tickers: Optional[Iterable[str]] = None,
        exclude_tickers: Optional[Iterable[str]] = None,
        ticker_regex: Optional[Union[str, Pattern]] = None,
        ticker_predicate: Optional[TickerPred] = None,
        numeric_filters: Optional[list[NumericFilter]] = None
) -> list[Dict[str, Any]]:

    all_tickers = []
    for t in stream_fetch_stock_tickers(
        ticker = ticker,
        ticker_type = ticker_type,
        cik = cik,
        market = market,
        sort = sort,
        order = order,
        limit = limit,
        max_pages = max_pages,
        numeric_filters = numeric_filters,
        include_tickers = include_tickers,
        exclude_tickers = exclude_tickers,
        ticker_regex = ticker_regex,
        ticker_predicate = ticker_predicate,
    ):
        all_tickers.append(t)
    return all_tickers