"""
Filter finance_tools_original.json, keeping only tools that can query
stock / asset prices (stocks, crypto, FX, commodities).
"""

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Keywords checked in trimmed description + full tool name (case-insensitive).
# Ordered from most-specific to most-general.
# ---------------------------------------------------------------------------
PRICE_KEYWORDS = [
    # ---- Direct price query phrases ----
    "stock price",
    "share price",
    "current price",
    "real-time price",
    "realtime price",
    "real time price",
    "live price",
    "latest price",
    "last price",
    "spot price",
    "market price",
    "average price",
    "closing price",
    "opening price",
    "last traded price",
    "last trade price",
    # ---- Quote ----
    "stock quote",
    "real-time quote",
    "realtime quote",
    "live quote",
    "live market",         # "live market data" (groww)
    "market quote",
    "price quote",
    "bulk quote",
    # ---- OHLC / candlestick ----
    "ohlc",                # ohlc / ohlcv cover both
    "ohlcv",
    "kline",               # Asian-style candlestick API name
    "candlestick",
    "open, high, low, close",
    "open high low close",
    # ---- Historical / time-series price data ----
    "price data",
    "price chart",
    "price history",
    "price and volume",
    "historical price",
    "historical candle",
    "historical stock",
    "intraday price",
    "intraday chart",
    "intraday stock",
    "time series",         # financial time-series APIs (alphavantage, FMP)
    # ---- API naming conventions with short descriptions ----
    # (checked against the full tool path, not just description)
    "time_series",         # e.g. alphavantage/time_series_daily
    "hist_data",           # e.g. akshare/china-stock get_hist_data
    "realtime_data",       # e.g. akshare/china-stock get_realtime_data
    "ticker_price",        # e.g. binance bn_ticker_price
    "avg_price",           # e.g. binance bn_avg_price
    "ticker_24hr",         # e.g. binance bn_ticker_24hr  (24h price stats)
    # ---- Crypto prices ----
    "coin price",
    "crypto price",
    "token price",
    "cryptocurrency price",
    "digital currency",    # alphavantage digital_currency_daily etc.
    # ---- FX / commodity prices ----
    "exchange rate",
    "crude oil",           # Brent / WTI crude oil price
    "oil price",
    "commodity price",
    # ---- Chinese keywords ----
    "行情",                # market/price data (行情数据, 实时行情, 历史行情)
    "K线",                 # candlestick chart
    "股价",                # stock price
    "价格",                # price
    "涨跌",                # price change (open+close change)
    "开盘",                # opening price
    "收盘",                # closing price
    "最新价",              # latest price
    "报价",                # quote / bid price
]

_PATTERN = re.compile(
    "|".join(re.escape(kw) for kw in PRICE_KEYWORDS),
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Negative filters: exclude tools whose PURPOSE is clearly NOT a price query.
# Checked against the full trimmed description (case-insensitive).
# ---------------------------------------------------------------------------
EXCLUDE_IF_ONLY_MATCH = {
    # "exchange rate" alone should not pull in general currency-conversion
    # calculators that just happen to fetch a rate as a side effect.
    # (If the description also mentions price/quote explicitly, it still passes.)
}

# Patterns in the tool name that indicate it is definitely NOT a price query
# even if a keyword accidentally matches (e.g. analyst price targets).
NAME_EXCLUSION = re.compile(
    r"(?:price_target|pricetarget|target_price|targetprice"
    r"|EXCEL_|excel_"                      # spreadsheet tools
    r"|add_chart|list_chart|update_chart"  # Excel chart management
    r"|upload_price"                       # uploading price data ≠ querying
    r"|run_strategy|run_backtest|backtest" # strategy execution tools
    r"|reference_currencies"              # lists currency symbols, not prices
    r")",
    re.IGNORECASE,
)


def _trim_desc(desc: str) -> str:
    """
    Strip parameter-doc sections (Args:, Parameters:, Examples:, Returns:, Note:)
    to avoid matching keywords that only appear in usage examples.
    Handles both newline-separated and inline section markers.
    """
    cutoff = re.search(
        r"(?:(?:\.\s+)|(?:\n\s*))(?:Args|Parameters|Arguments|Examples?|Returns?|Notes?|Raises?)\s*:",
        desc,
        re.IGNORECASE,
    )
    return desc[: cutoff.start()] if cutoff else desc


def is_stock_price_tool(tool: dict) -> bool:
    name = tool["function"]["name"]
    desc = tool["function"].get("description", "")

    # Hard-exclude tools whose name matches known non-price patterns
    if NAME_EXCLUSION.search(name):
        return False

    text = f"{name} {_trim_desc(desc)}"
    return bool(_PATTERN.search(text))


def main():
    src = Path(__file__).parent / "finance_tools_original.json"
    dst = Path(__file__).parent / "finance_tools_stock_price.json"

    with src.open(encoding="utf-8") as f:
        tools = json.load(f)

    kept = [t for t in tools if is_stock_price_tool(t)]

    with dst.open("w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    # ---- summary ----
    print(f"Total tools  : {len(tools)}")
    print(f"Kept (price) : {len(kept)}")
    print(f"Removed      : {len(tools) - len(kept)}")
    print(f"Output       : {dst}")
    print()

    # Group by server for overview
    servers: dict[str, list[str]] = {}
    for t in kept:
        sn = t["server_name"]
        servers.setdefault(sn, []).append(t["function"]["name"].split("/")[-1])

    print(f"{'Server':<40} {'Count':>5}  Tools")
    print("-" * 80)
    for sn, names in sorted(servers.items(), key=lambda x: -len(x[1])):
        print(f"  {sn:<38} {len(names):>5}  {', '.join(names[:6])}"
              + (" ..." if len(names) > 6 else ""))


if __name__ == "__main__":
    main()
