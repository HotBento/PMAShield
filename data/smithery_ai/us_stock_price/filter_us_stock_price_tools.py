"""
Second-pass filter: from finance_tools_stock_price.json,
keep only tools that query **US stock prices**.

Strategy: deny-list approach (start from 229 price-query tools,
then exclude everything that is clearly non-US or non-stock).

Exclusion logic:
  1. Entire servers that are non-US market focused
  2. Specific tool-name patterns within remaining servers (crypto, FX,
     commodities, non-US exchanges, non-price content)
  3. Exact tool names that are non-price even within US-focused servers
"""

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Entire servers to exclude (all tools in these servers are non-US)
# ---------------------------------------------------------------------------
EXCLUDE_SERVERS: set[str] = {
    # ---- Crypto exchanges / DeFi ----
    "solana-defi",          # Solana on-chain token prices
    "bot-trading",          # K-pop lightstick tokens
    "binance",              # Binance global crypto exchange
    "binance-th-mcp",       # Binance Thailand
    "bitkub",               # Thai crypto exchange
    "bithumb-mcp",          # Korean crypto exchange
    "coinranking-mcp",      # CoinRanking crypto
    "coinex_mcp_server",    # CoinEx crypto exchange
    "free-coin-price-mcp",  # CoinGecko coin prices
    "mcp-crypto-price",     # Generic crypto prices
    "apix420mcp1",          # Crypto price + currency convert
    "duckchain-mcp",        # DuckChain crypto market
    "uniswap-trader-mcp",   # Uniswap DEX swap price
    "mcp",                  # Generic token OHLCV (crypto)
    # ---- Asian stock markets ----
    "china-stock-mcp",      # Chinese A/B/H-share market
    "akshare-one-mcp",      # AkShare — Chinese financial data
    "mcp-aktools",          # AkShare tools (primarily Chinese)
    # ---- Indian stock markets ----
    "groww-mcp-server",     # Groww (India)
    "blinkxmcp",            # BlinkX (India)
    # ---- Prediction / game / niche markets ----
    "eve-online-mcp",       # EVE Online in-game market
    "dflow-mcp",            # Kalshi prediction market
    # ---- FX / currency only ----
    "seahboonkeong-chat-bnmapi",  # Malaysian BNM exchange rates
    "currency-and-oil",     # Russian RUB exchange rate + Brent price
    "test",                 # Generic currency converter
    # ---- Miscellaneous non-stock ----
    "calculator",           # Bond price calculator
}

# ---------------------------------------------------------------------------
# 2. Tool-name patterns to exclude within remaining servers
#    (case-insensitive, matched against the short name = last path component)
# ---------------------------------------------------------------------------
_EXCLUDE_PATTERN = re.compile(
    r"(?:"
    # Crypto tools in multi-asset servers (FMP, financial-data, alphavantage)
    r"[Cc]rypto|digital_currency|CryptoCurrency"
    # Forex / FX tools
    r"|[Ff]orex|forex_|fx_intraday|fx_daily|fx_weekly|fx_monthly|exchange_rate"
    # Commodity prices
    r"|crude_oil|CommodityQuote|CommodityPrice"
    # Analyst-grade news (not price data)
    r"|GradeNews|GradeLatest"
    # International (non-US) stocks
    r"|InternationalStock"
    # Insider trading data
    r"|InsiderTransaction"
    # Earnings releases / announcements
    r"|EarningsRelease"
    r")",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# 3. Exact short-name exclusions (non-price tools that passed the first filter)
# ---------------------------------------------------------------------------
_EXACT_EXCLUDE: set[str] = {
    "getDividends",          # financial-data: dividend schedule (≠ price chart)
    "getStockSplits",        # financial-data: split history
    "getSplitsCalendar",     # financial-data: upcoming splits
    "getEtfSymbols",         # financial-data: ETF symbol list, not prices
    "getStockGradeNews",     # FMP: analyst grade news
    "getStockGradeLatestNews",  # FMP: analyst grade news
    "getHistoricalSectorPE",    # FMP: P/E ratios by sector (≠ price)
    "getHistoricalIndustryPE",  # FMP: P/E ratios by industry (≠ price)
    "okx_feeds",             # turf-mcp-server: OKX crypto exchange data (≠ US stock)
}


def is_us_stock_price_tool(tool: dict) -> bool:
    server = tool["server_name"]
    name = tool["function"]["name"]
    short = name.split("/")[-1]

    if server in EXCLUDE_SERVERS:
        return False
    if _EXCLUDE_PATTERN.search(short):
        return False
    if short in _EXACT_EXCLUDE:
        return False
    return True


def main():
    src = Path(__file__).parent / "finance_tools_stock_price.json"
    dst = Path(__file__).parent / "finance_tools_us_stock_price.json"

    with src.open(encoding="utf-8") as f:
        tools = json.load(f)

    kept = [t for t in tools if is_us_stock_price_tool(t)]

    with dst.open("w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print(f"Input  (price tools) : {len(tools)}")
    print(f"Kept (US stock price): {len(kept)}")
    print(f"Removed              : {len(tools) - len(kept)}")
    print(f"Output               : {dst}")
    print()

    servers: dict[str, list[str]] = {}
    for t in kept:
        sn = t["server_name"]
        servers.setdefault(sn, []).append(t["function"]["name"].split("/")[-1])

    print(f"{'Server':<42} {'#':>4}  Tools (first 6)")
    print("-" * 85)
    for sn, names in sorted(servers.items(), key=lambda x: -len(x[1])):
        sample = ", ".join(names[:6]) + (" ..." if len(names) > 6 else "")
        print(f"  {sn:<40} {len(names):>4}  {sample}")


if __name__ == "__main__":
    main()
