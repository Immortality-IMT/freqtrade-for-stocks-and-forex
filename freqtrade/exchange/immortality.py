import asyncio
import logging
import time
import traceback
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, cast

import pandas as pd
import requests
from web3 import Web3
from web3.exceptions import ContractLogicError

from freqtrade.data.converter import ohlcv_to_dataframe
from freqtrade.enums import CandleType
from freqtrade.exceptions import ExchangeError, OperationalException
from freqtrade.exchange.common import retrier
from freqtrade.exchange.ohlcv_providers import build_default_pool
from freqtrade.exchange.stockexchange import Stockexchange


logger = logging.getLogger(__name__)


# Constants
BSC_WS_URL = "wss://bsc-mainnet.nodereal.io/ws/v1/{ws_api_key}"
BSC_RPC_URL = "https://bsc-mainnet.nodereal.io/v1/{api_key}"
PANCAKESWAP_ROUTER_ADDR = Web3.to_checksum_address("0x10ED43C718714eb63d5aA57B78B54704E256024E")
WBNB_ADDR = Web3.to_checksum_address("0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c")
IMMORTALITY_ADDR = Web3.to_checksum_address("0x2bF2141eD175f3236903cF07de33D7324871802D")  # IMT
PAIR_ADDRESS = Web3.to_checksum_address(
    "0xfA56E9AbcaA45207bE5E43cF475Ee061768CA915"
)  # IMT/BNB pair
MIN_INTERVAL = 60.0  # seconds between NodeReal calls
NODEREAL_FREE_URL = "https://open-platform.nodereal.io/{api_key}/pancakeswap-free/graphql"

# Trading parameters
IMT_DECIMALS = 8
BUY_BNB_AMOUNT = 0.004  # Match config.json stake_amount
SELL_IMT_QUANTITY = None  # Dynamically set in sell method based on buy amount or 10_000_000
RPC_SYNC_DELAY_SECONDS = 7
# Dry-run realism (override with "dry_run_buy_fee" / "dry_run_sell_fee" in config.json)
DRY_RUN_BUY_FEE = 0.0  # live buy() assumes no token transfer fee on the way in
DRY_RUN_SELL_FEE = 0.10  # same 10% reflection fee live sell() assumes
DRY_ORDER_CACHE_MAX = 500  # simulated orders remembered for get_order()

# NodeReal RPC budget (free tier is quota- and CUPS-limited). Override in config.json:
#   "price_cache_ttl", "balance_cache_ttl"
PRICE_CACHE_TTL = 15.0  # seconds to reuse the on-chain price instead of a new getReserves call
BALANCE_CACHE_TTL = 30.0  # seconds to reuse wallet balances (cleared after every trade)
BALANCE_STALE_MAX = 300.0  # on RPC errors, serve cached balances up to this old
RPC_ERROR_BACKOFF = 5.0  # after an RPC failure, don't retry for this many seconds
TRUNCATE = 0
ROUND = 1
DECIMAL_PLACES = 2
SIGNIFICANT_DIGITS = 3

# ABIs
PAIR_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "getReserves",
        "outputs": [
            {"internalType": "uint112", "name": "_reserve0", "type": "uint112"},
            {"internalType": "uint112", "name": "_reserve1", "type": "uint112"},
            {"internalType": "uint32", "name": "_blockTimestampLast", "type": "uint32"},
        ],
        "stateMutability": "view",
        "type": "function",
    }
]
ROUTER_ABI = [
    {
        "name": "getAmountsOut",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "amountIn", "type": "uint256"},  # codespell:ignore
            {"name": "path", "type": "address[]"},
        ],
        "outputs": [{"name": "", "type": "uint256[]"}],
    },
    {
        "name": "swapExactETHForTokensSupportingFeeOnTransferTokens",
        "type": "function",
        "stateMutability": "payable",
        "inputs": [
            {"name": "amountOutMin", "type": "uint256"},
            {"name": "path", "type": "address[]"},
            {"name": "to", "type": "address"},
            {"name": "deadline", "type": "uint256"},
        ],
        "outputs": [],
    },
    {
        "name": "swapExactTokensForETHSupportingFeeOnTransferTokens",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "amountIn", "type": "uint256"},  # codespell:ignore
            {"name": "amountOutMin", "type": "uint256"},
            {"name": "path", "type": "address[]"},
            {"name": "to", "type": "address"},
            {"name": "deadline", "type": "uint256"},
        ],
        "outputs": [],
    },
]
TOKEN_ABI = [
    {
        "name": "balanceOf",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "owner", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "approve",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}],
        "outputs": [{"name": "", "type": "bool"}],
    },
    {
        "name": "allowance",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "decimals",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint8"}],
    },
]


def fetch_ohlcv_from_server(pair_address: str, since: int, url: str) -> list[list]:
    """
    Fetch OHLCV data from JSON cache at the specified URL, filter by timestamp, and return candles.
    Shape: { "data": [ {timestamp, open, high, low, close, volume}, … ] }
    """
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        raw = resp.json().get("data", [])
    except Exception as e:
        logger.error(f"Failed to load OHLCV JSON from {url}: {e}")
        return []

    candles = []
    for b in raw:
        ts_ms = int(b["timestamp"]) * 1000
        if ts_ms <= since:
            continue
        try:
            candles.append(
                [
                    ts_ms,
                    float(b["open"]),
                    float(b["high"]),
                    float(b["low"]),
                    float(b["close"]),
                    float(b["volume"]),
                ]
            )
        except Exception as e:
            logger.warning(f"Malformed OHLCV bucket: {b} ({e})")
    return sorted(candles, key=lambda x: x[0])  # Sort oldest to newest


def _epoch_ms(dates: pd.Series) -> pd.Series:
    """
    Epoch milliseconds of a datetime Series, independent of its resolution.
    pandas 3 yields datetime64[ms]/[us] (not always [ns]), so `astype("int64") // 10**6`
    silently returns wrong values and filters every row out.
    """
    epoch = pd.Timestamp(0, tz="UTC") if dates.dt.tz is not None else pd.Timestamp(0)
    return (dates - epoch) // pd.Timedelta(milliseconds=1)


def send_tx(w3, fn, wallet_address: str, private_key: str, value: int = 0) -> str:
    """Sends a transaction and waits for confirmation."""
    tx_params = {
        "from": wallet_address,
        "gas": 300000,
        "gasPrice": max(int(w3.eth.gas_price * 1.1), w3.to_wei("5", "gwei")),
        "nonce": w3.eth.get_transaction_count(wallet_address),
    }
    if value:
        tx_params["value"] = value
    try:
        transaction = fn.build_transaction(tx_params)
        signed = w3.eth.account.sign_transaction(transaction, private_key=private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=600)
        if receipt.status == 0:
            raise ExchangeError(f"Transaction {tx_hash.hex()} failed")
        return tx_hash.hex()
    except Exception as e:
        logger.error(f"Transaction failed: {str(e)}")
        raise ExchangeError(f"Failed to send transaction: {str(e)}")


@dataclass
class Candle:
    """Represents an OHLCV candle."""

    timestamp: int  # Start time in milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float


class CandleBuilder:
    """Builds OHLCV candles from trade data for a given timeframe."""

    def __init__(self, timeframe: str):
        self.timeframe_seconds = Immortality.timeframe_to_seconds(timeframe)
        self.current_candle: Candle | None = None
        self.decimals = 8  # IMT decimals

    def update(self, price: float, volume: float, timestamp_ms: int) -> Candle | None:
        """Updates the current candle with new trade data."""
        candle_start = (timestamp_ms // (self.timeframe_seconds * 1000)) * (
            self.timeframe_seconds * 1000
        )
        if not self.current_candle or self.current_candle.timestamp < candle_start:
            finalized = self.current_candle
            self.current_candle = Candle(candle_start, price, price, price, price, volume)
            return finalized
        else:
            self.current_candle.high = max(self.current_candle.high, price)
            self.current_candle.low = min(self.current_candle.low, price)
            self.current_candle.close = price
            self.current_candle.volume += volume
            return None

    def get_current(self) -> Candle | None:
        """Returns the current in-progress candle."""
        return self.current_candle


class Immortality(Stockexchange):
    """Custom exchange class for PancakeSwap integration with real-time OHLCV streaming."""

    _use_ccxt = False
    _ft_has_default = {
        "ohlcv_candle_limit": 200,  # ← down from 1000
        "order_time_in_force": ["gtc"],
        "stoploss_on_exchange": False,
        "ws_enabled": True,
        "ws_auto_reconnect": True,
        "ws_reconnect_interval": 30,
        "watch_ohlcv": True,
        "use_entry_signal": True,
        "use_exit_signal": True,
    }

    id = "immortality"

    def __init__(
        self,
        config: dict,
        *,
        exchange_config: dict | None = None,
        validate: bool = True,
        load_leverage_tiers: bool = False,
    ) -> None:
        super().__init__(
            config,
            exchange_config=exchange_config,
            validate=validate,
            load_leverage_tiers=load_leverage_tiers,
        )
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"Full configuration: {config}")  # Debug config loading
        self.dry_run = config.get("dry_run", False)
        self.slippage_tolerance = config.get("slippage_tolerance", 0.15)  # Default 15%

        self.server_ohlcv_url = self.config.get("ohlcv_url", "http://14.180.249.77/ohlcv.json")
        self.pair_address = "0xfa56e9abcaa45207be5e43cf475ee061768ca915"  # IMT/WBNB, lowercase
        self.ohlcv_pool = build_default_pool(self.config, self.logger)

        self.api_key_value = None  # Cache HTTP API key
        self.w3 = Web3(Web3.HTTPProvider(BSC_RPC_URL.format(api_key=self.api_key)))
        if not self.w3.is_connected():
            raise OperationalException("Cannot connect to BSC RPC")
        self.latest_ohlcv: dict[tuple[str, str, str], pd.DataFrame] = {}
        self._last_call_time = 0.0
        self._last_price: float | None = None
        self._dry_orders: dict[str, dict] = {}
        self._price_ts = -1e9  # monotonic time of last successful price fetch
        self._price_backoff_until = 0.0
        self._price_rpc_calls = 0
        self._price_stats_ts = time.monotonic()
        self._balances_cache: dict | None = None
        self._balances_ts = -1e9
        self._min_interval = MIN_INTERVAL
        self.candle_builders: dict[tuple[str, str], CandleBuilder] = {}
        cache_path = "user_data/data/immortality/cache_IMT-BNB_5m.csv"
        try:
            df = pd.read_csv(cache_path, parse_dates=["date"])
            self.latest_ohlcv[("IMT/BNB", "5m", "spot")] = df
            self.logger.info(f"Loaded OHLCV cache from {cache_path}, {len(df)} rows")
        except FileNotFoundError:
            self.logger.info(f"No cache file found at {cache_path}, starting empty")

    @property
    def name(self):
        return "immortality"

    @property
    def api_key(self) -> str:
        """Retrieve NodeReal HTTP API key from configuration."""
        if self.api_key_value is not None:
            return self.api_key_value
        exchange_conf = (
            self.exchange_config
            if getattr(self, "exchange_config", None)
            else self.config.get("exchange", {})
        )
        key = exchange_conf.get("api_key", "").strip()
        self.logger.info(f"NodeReal HTTP API key retrieved: {key!r}")
        if not key:
            self.logger.error(
                "Missing 'nodereal_api_key' in config.json. Add key to the 'exchange' section."
            )
            raise OperationalException("NodeReal API key is required for OHLCV data retrieval.")
        self.logger.debug(f"Using NodeReal HTTP API key: {key}")
        self.api_key_value = key
        return key

    @property
    def wallet(self) -> str:
        """Retrieve wallet address from configuration."""
        exchange_conf = (
            self.exchange_config
            if getattr(self, "exchange_config", None)
            else self.config.get("exchange", {})
        )
        key = exchange_conf.get("key", "").strip()
        self.logger.info(f"Wallet address retrieved: {key!r}")
        if not key:
            self.logger.error(
                "Missing 'key' (wallet address) in config.json. Add key to the 'exchange' section."
            )
            raise OperationalException("Wallet address is required for trading.")
        self.logger.debug(f"Using wallet address: {key}")
        return self.w3.to_checksum_address(key)

    @property
    def private_key(self) -> str:
        """Retrieve private key from configuration."""
        exchange_conf = (
            self.exchange_config
            if getattr(self, "exchange_config", None)
            else self.config.get("exchange", {})
        )
        secret = exchange_conf.get("secret", "").strip()
        if not secret:
            self.logger.error(
                "Missing 'secret' (private key) in config.json. Add key to the 'exchange' section."
            )
            raise OperationalException("Private key is required for trading.")
        self.logger.debug(
            "Using private key: DO NOT OUTPUT PRIVATE KEY"
        )  # DO NOT OUTPUT PRIVATE KEY
        return secret

    def ws_connection_reset(self):
        pass

    async def watch_ohlcv(
        self,
        pair: str,
        timeframe: str,
        limit: int | None = None,
    ) -> AsyncGenerator[list[float], None]:
        """Streams real-time OHLCV data for the specified pair and timeframe."""
        if pair != "IMT/BNB":
            raise OperationalException(f"Pair {pair} not supported")
        self.validate_timeframes(timeframe)

        key = (pair, timeframe)
        if key not in self.candle_builders:
            self.candle_builders[key] = CandleBuilder(timeframe)
            historical = self.get_ohlcv(pair, timeframe, limit=1)
            if not historical.empty:
                last = historical.iloc[-1]
                self.candle_builders[key].current_candle = Candle(
                    int(last["date"].timestamp() * 1000),
                    last["open"],
                    last["high"],
                    last["low"],
                    last["close"],
                    last["volume"],
                )
                self.logger.info(f"Initialized {pair}/{timeframe} with historical candle")

        while True:
            try:
                builder = self.candle_builders[key]
                current = builder.get_current()
                if current:
                    yield [
                        current.timestamp,
                        current.open,
                        current.high,
                        current.low,
                        current.close,
                        current.volume,
                    ]
                await asyncio.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in watch_ohlcv for {pair}/{timeframe}: {str(e)}")
                await asyncio.sleep(5)

    def refresh_latest_ohlcv(self, pairs: list[str]) -> None:
        for item in pairs:
            try:
                if isinstance(item, tuple):
                    pair = item[0]
                    timeframe = item[1] if len(item) > 1 else self.config.get("timeframe", "1h")
                    candle_type = item[2] if len(item) > 2 else CandleType.SPOT
                else:
                    pair = item
                    timeframe = self.config.get("timeframe", "1h")
                    candle_type = CandleType.SPOT

                ohlcv = self.get_ohlcv(pair, timeframe, limit=200, candle_type=candle_type)
                key = (pair, timeframe, candle_type.value)

                if key in self.candle_builders:
                    current = self.candle_builders[(pair, timeframe)].get_current()
                    if current:
                        latest = pd.DataFrame(
                            [
                                [
                                    current.timestamp,
                                    current.open,
                                    current.high,
                                    current.low,
                                    current.close,
                                    current.volume,
                                ]
                            ],
                            columns=["timestamp", "open", "high", "low", "close", "volume"],
                        )
                        latest["date"] = pd.to_datetime(latest["timestamp"], unit="ms", utc=True)
                        ohlcv = pd.concat(
                            [ohlcv, latest[["date", "open", "high", "low", "close", "volume"]]]
                        )
                        ohlcv = ohlcv.drop_duplicates(subset="date").sort_values("date")

                if not ohlcv.empty:
                    ohlcv = ohlcv.tail(200)
                    self.latest_ohlcv[key] = ohlcv
                    self.logger.info(
                        f"Refreshed and pruned OHLCV for {pair}/{timeframe}, candles: {len(ohlcv)}"
                    )
                else:
                    self.logger.warning(f"No OHLCV data for {pair}/{timeframe}, cache not updated")
            except Exception as e:
                self.logger.error(f"Failed to refresh OHLCV for {pair}/{timeframe}: {str(e)}")

    def interpolate_ohlcv(self, raw: list[list], timeframe: str) -> list[list]:
        """Interpolate hourly OHLCV data to a target timeframe."""
        target_seconds = self.timeframe_to_seconds(timeframe)
        if target_seconds >= 3600:
            return raw

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("date", inplace=True)

        df = (
            df.resample(f"{target_seconds // 60}min")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .interpolate(method="linear")
            .ffill()
        )

        df["high"] = df["high"] * 1.001
        df["low"] = df["low"] * 0.999
        df["volume"] = df["volume"].apply(lambda x: max(x / (3600 / target_seconds), 0.0))

        result = []
        for ts, row in df.iterrows():
            result.append(
                [
                    int(ts.timestamp() * 1000),
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                    row["volume"],
                ]
            )
        return result

    def process_and_filter_dataframe(
        self,
        df: pd.DataFrame,
        cached_df: pd.DataFrame | None,
        since_ms: int,
        pair_str: str,
        timeframe: str,
    ) -> pd.DataFrame:
        if since_ms:
            df = df[_epoch_ms(df["date"]) >= since_ms]

        cutoff_ms = int(time.time() * 1000) - (90 * 24 * 3600 * 1000)
        df = df[_epoch_ms(df["date"]) >= cutoff_ms]

        if cached_df is not None and not cached_df.empty:
            df = pd.concat([cached_df, df]).drop_duplicates(subset="date").sort_values("date")
            self.logger.debug(f"Merged with cached data, total candles: {len(df)}")

        if df.empty:
            self.logger.warning(
                "Empty OHLCV DataFrame after filtering for %s/%s, using fallback",
                pair_str,
                timeframe,
            )
            return self._get_fallback_candle(timeframe, pair_str)

        self.logger.debug("OHLCV DataFrame sample:\n%s", df.head(5).to_string())

        self._validate_latest_price(df, pair_str, timeframe)

        self.logger.info("Retrieved %d candles for %s/%s", len(df), pair_str, timeframe)

        df = self._append_synthetic_candle_if_needed(df, timeframe)

        if len(df) > 200:
            df = df.tail(200)
            self.logger.debug("Trimmed returned OHLCV DataFrame to last 200 rows")

        return df

    def _append_synthetic_candle_if_needed(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        try:
            if df.empty:
                self.logger.warning("Empty DataFrame, cannot append synthetic candle")
                return df
            last_ts = df["date"].iloc[-1]
            now_utc = pd.Timestamp.utcnow()
            interval = pd.Timedelta(seconds=self.timeframe_to_seconds(timeframe))

            if now_utc - last_ts >= interval:
                price = self.get_price()  # Use current market price
                new_row = {
                    "date": now_utc,
                    "open": price,
                    "high": price * 1.001,
                    "low": price * 0.999,
                    "close": price,
                    "volume": 0.0,
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                self.logger.debug(
                    f"Appended synthetic candle at {now_utc} with price {price}"
                    f"new index: {len(df) - 1}"
                )
            df = df.reset_index(drop=True)  # Reset index to ensure sequential indices
            return df
        except Exception as e:
            self.logger.error(f"Failed to append synthetic candle: {e}")
            return df

    def _get_fallback_candle(self, timeframe: str, pair_str: str) -> pd.DataFrame:
        """Return a fallback OHLCV DataFrame with synthetic price data."""
        try:
            current_price = self.get_price()
        except ExchangeError as e:
            self.logger.error(f"Failed to fetch fallback price: {str(e)}")
            current_price = 0.00000001  # Emergency floor

        ts = int(time.time() * 1000)
        fallback = [
            [
                ts,
                current_price,
                current_price * 1.001,
                current_price * 0.999,
                current_price,
                0.0,
            ]
        ]

        return ohlcv_to_dataframe(
            fallback,
            timeframe,
            pair_str,
            fill_missing=False,
            drop_incomplete=True,
        )

    def _validate_latest_price(self, df: pd.DataFrame, pair_str: str, timeframe: str) -> None:
        """Log comparison between latest close price and current market price."""
        try:
            latest_close = df.iloc[-1]["close"]
            current_price = self.get_price()
            diff_pct = abs(latest_close - current_price) / current_price * 100
            self.logger.debug(
                f"Price validation for {pair_str}/{timeframe}: "
                f"close={latest_close}, get_price={current_price}, diff={diff_pct:.2f}%"
            )
        except ExchangeError as e:
            self.logger.error(f"Price validation failed: {str(e)}")

    def rate_limit_error_handler(
        self, cached_df: pd.DataFrame | None, timeframe: str, pair_str: str
    ) -> pd.DataFrame:
        """Handle 429 rate limit error by generating synthetic candles or falling back."""
        self.logger.error(
            "Rate limit exceeded (429) for %s/%s, using cached data", pair_str, timeframe
        )
        if cached_df is not None and not cached_df.empty:
            last_candle = cached_df.iloc[-1]
            last_ts = last_candle["date"]
            interval = pd.Timedelta(seconds=self.timeframe_to_seconds(timeframe))
            now_utc = pd.Timestamp.utcnow()
            synthetic_candles = []
            current_ts = last_ts + interval
            while current_ts <= now_utc:
                synthetic_candles.append(
                    [
                        int(current_ts.timestamp() * 1000),
                        last_candle["close"],
                        last_candle["close"] * 1.001,
                        last_candle["close"] * 0.999,
                        last_candle["close"],
                        0.0,
                    ]
                )
                current_ts += interval
            if synthetic_candles:
                df = ohlcv_to_dataframe(
                    synthetic_candles,
                    timeframe,
                    pair_str,
                    fill_missing=True,
                    drop_incomplete=True,
                )
                df = pd.concat([cached_df, df]).drop_duplicates(subset="date").sort_values("date")
                self.logger.info(
                    "Generated %d synthetic candles for %s/%s due to 429 error",
                    len(synthetic_candles),
                    pair_str,
                    timeframe,
                )
                return df.tail(200)
        return self._get_fallback_candle(timeframe, pair_str)

    def create_synthetic_candles(
        self,
        last_candle: pd.Series,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        interval_seconds: int,
    ) -> list[list[float]]:
        """Generate synthetic candles starting from `start_time` until `end_time`."""
        synthetic = []
        current_ts = start_time
        while current_ts <= end_time:
            ts_ms = int(current_ts.timestamp() * 1000)
            close = last_candle["close"]
            synthetic.append(
                [
                    ts_ms,
                    close,
                    close * 1.001,
                    close * 0.999,
                    close,
                    0.0,
                ]
            )
            current_ts += pd.Timedelta(seconds=interval_seconds)
        return synthetic

    def _handle_empty(
        self, cached_df: pd.DataFrame | None, timeframe: str, pair_str: str, interval_s: int
    ) -> pd.DataFrame:
        """Handle case when no OHLCV data is fetched."""
        self.logger.warning("No OHLCV data fetched for %s/%s, checking cache", pair_str, timeframe)
        if cached_df is not None and not cached_df.empty:
            self.logger.info("Using cached OHLCV data")
            return cached_df.tail(200)
        self.logger.warning("No cached data available, using fallback candle")
        return self._get_fallback_candle(timeframe, pair_str)

    def prepare_cache_and_throttle(
        self,
        pair: str,
        timeframe: str,
        since_ms: int,
        candle_type: CandleType,
    ) -> tuple[str, int, pd.DataFrame | None]:
        pair_str = pair
        key = (pair_str, timeframe, candle_type.value)
        cached_df = self.latest_ohlcv.get(key)

        if cached_df is not None and not cached_df.empty:
            if not pd.api.types.is_datetime64_any_dtype(cached_df["date"]):
                self.logger.warning("cached_df['date'] is not datetime, converting...")
                cached_df["date"] = pd.to_datetime(cached_df["date"], errors="coerce")
                before = len(cached_df)
                cached_df = cached_df.dropna(subset=["date"])
                dropped = before - len(cached_df)
                if dropped:
                    self.logger.warning(f"Dropped {dropped} malformed date rows from cache")

            if not cached_df.empty:
                latest_ts_ms = int(cached_df["date"].iloc[-1].timestamp() * 1000)
                if since_ms < latest_ts_ms:
                    since_ms = latest_ts_ms
                    self.logger.debug(
                        f"Using cached latest timestamp: {since_ms} for {pair_str}/{timeframe}"
                    )
            else:
                self.logger.warning("All cached dates were invalid. No valid timestamps found.")

        return pair_str, since_ms, cached_df

    def get_ohlcv(
        self,
        pair: str,
        timeframe: str,
        since_ms: int = 0,
        limit: int = 200,
        candle_type: CandleType = CandleType.SPOT,
    ) -> pd.DataFrame:
        try:
            pair_str, orig_since_ms, cached_df = self.prepare_cache_and_throttle(
                pair, timeframe, since_ms, candle_type
            )
            interval_s = self.timeframe_to_seconds(timeframe)

            raw = self.ohlcv_pool.fetch(
                ref_price=self._last_price,
                need=200 if orig_since_ms == 0 else 0,
            )

            if raw:
                self.logger.info(
                    "OHLCV via %s: %d hourly candles, %s -> %s UTC",
                    self.ohlcv_pool.last_provider,
                    len(raw),
                    pd.to_datetime(raw[0][0], unit="ms"),
                    pd.to_datetime(raw[-1][0], unit="ms"),
                )

            if not raw:
                return self._handle_empty(cached_df, timeframe, pair_str, interval_s)

            raw = [c for c in raw if c[0] > orig_since_ms]

            if interval_s < 3600:
                cap = 50 if orig_since_ms == 0 else 10
                recent = raw[-cap:]
                raw = self.interpolate_ohlcv(recent, timeframe)
                self.logger.debug("Interpolated to %d candles for %s", len(raw), timeframe)

            df = ohlcv_to_dataframe(
                raw, timeframe, pair_str, fill_missing=True, drop_incomplete=True
            )

            df = self.process_and_filter_dataframe(
                df, cached_df, orig_since_ms, pair_str, timeframe
            )

            if not df.empty:
                last_ts = df["date"].iloc[-1]
                now = pd.Timestamp.utcnow().floor(f"{interval_s}s")
                if now > last_ts:
                    dummy = pd.Series(df.iloc[-1])
                    synth2 = self.create_synthetic_candles(
                        dummy, last_ts + pd.Timedelta(seconds=interval_s), now, interval_s
                    )
                    if synth2:
                        df2 = ohlcv_to_dataframe(
                            synth2, timeframe, pair_str, fill_missing=True, drop_incomplete=True
                        )
                        df = pd.concat([df, df2]).drop_duplicates(subset="date").sort_values("date")
                        self.logger.info(
                            "Appended %d post-raw synthetic candles for %s/%s",
                            len(synth2),
                            pair_str,
                            timeframe,
                        )

            df["date"] = pd.to_datetime(df["date"]).dt.floor(f"{interval_s}s")

            if not df.empty:
                latest_price = df["close"].iloc[-1]
                self.logger.info(f"Latest price for {pair_str} ({timeframe}): {latest_price:.18f}")

            return self._persist_and_return(df, pair_str, timeframe)

        except requests.HTTPError as e:
            if e.response.status_code == 429:
                return self.rate_limit_error_handler(cached_df, timeframe, pair_str)
            self.logger.error(
                "HTTP error fetching OHLCV for %s/%s: %s", pair_str, timeframe, str(e)
            )
            return self._get_fallback_candle(timeframe, pair_str)

        except Exception:
            self.logger.error(
                "Error fetching OHLCV for %s/%s:\n%s", pair_str, timeframe, traceback.format_exc()
            )
            return pd.DataFrame()

    def _persist_and_return(self, df: pd.DataFrame, pair_str: str, timeframe: str) -> pd.DataFrame:
        """Helper to trim, cache in memory, write to disk, and return."""
        df = df.tail(200)
        cache_key = (pair_str, timeframe, "spot")
        self.latest_ohlcv[cache_key] = df

        cache_path = (
            f"user_data/data/immortality/cache_{pair_str.replace('/', '-')}_{timeframe}.csv"
        )
        try:
            df.to_csv(cache_path, index=False)
            self.logger.debug(f"Persisted OHLCV cache to {cache_path}")
        except Exception as e:
            self.logger.error(f"Failed to write OHLCV cache to {cache_path}: {e}")
        return df

    def klines(
        self,
        pair: str | tuple,
        timeframe: str | None = None,
        since: int | None = None,
        limit: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        if isinstance(pair, tuple):
            pair_str = pair[0]
            timeframe_from_tuple = pair[1] if len(pair) > 1 and pair[1] else None
            candle_type = pair[2] if len(pair) > 2 else CandleType.SPOT
            if timeframe_from_tuple and timeframe is None:
                timeframe = timeframe_from_tuple
                self.logger.debug(f"Using timeframe {timeframe} from tuple {pair}")
        else:
            pair_str = pair
            candle_type = CandleType.SPOT
        if timeframe is None:
            timeframe = self.config.get("timeframe", "1h")
            self.logger.info(
                f"No timeframe provided for {pair_str}, using default from config: {timeframe}"
            )
        self.logger.debug(
            f"Calling klines: pair={pair_str}, timeframe={timeframe}, "
            f"since={since}, limit={limit}, candle_type={candle_type}"
        )
        try:
            self.validate_timeframes(timeframe)
            since_ms = since if since is not None else 0
            limit = min(limit or cast(int, self._ft_has_default["ohlcv_candle_limit"]), 200)
            return self.get_ohlcv(pair_str, timeframe, since_ms, limit, candle_type)
        except Exception as e:
            self.logger.error(f"Error in klines for {pair_str}/{timeframe}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    def validate_timeframes(self, timeframe: str) -> None:
        """Validates that the timeframe is supported."""
        supported = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        if timeframe not in supported:
            raise OperationalException(f"Timeframe {timeframe} not supported")

    @staticmethod
    def timeframe_to_seconds(timeframe: str) -> int:
        """Converts timeframe string to seconds."""
        units = {"m": 60, "h": 3600, "d": 86400}
        num = int(timeframe[:-1])
        unit = timeframe[-1]
        return num * units[unit]

    def validate_config(self, config: dict) -> None:
        """Freqtrade startup hook. No ccxt-backed validation for an on-chain exchange."""
        return

    def get_option(self, param: str, default: Any = None) -> Any:
        """
        Mirror of Exchange.get_option(): look up a feature flag in _ft_has_default.
        Unknown flags return `default` (None), i.e. "feature not supported", which makes
        freqtrade skip optional steps such as the wallet-history migration.
        """
        return self._ft_has_default.get(param, default)

    def get_pairs(self) -> list[str]:
        return ["IMT/BNB"]

    def _init_ccxt(self, exchange_conf, ccxt_wrapper, ccxt_config):
        return None

    def get_proxy_coin(self) -> str:
        return self.config["stake_currency"]

    def get_markets(self):
        return {
            "IMT/BNB": {
                "id": "IMT/BNB",
                "symbol": "IMT/BNB",
                "base": "IMT",
                "quote": "BNB",
                "active": True,
                "spot": True,
                "precision": {"amount": 8, "price": 8},
                "limits": {
                    "amount": {"min": 0.001, "max": 1000000},
                    "price": {"min": 0.00000001, "max": 1000000},
                },
            }
        }

    def reload_markets(self) -> None:
        self.logger.info("reload_markets called — no action required for Immortality.")

    def market_is_tradable(self, market: dict[str, Any]) -> bool:
        return market.get("active", False) and market.get("spot", False)

    def fetch_positions(self) -> list[dict]:
        """Returns an empty list as spot trading does not use positions."""
        self.logger.debug("fetch_positions called — returning empty list for spot trading")
        return []

    @property
    def token(self):
        return self.w3.eth.contract(address=IMMORTALITY_ADDR, abi=TOKEN_ABI)

    @property
    def router(self):
        return self.w3.eth.contract(address=PANCAKESWAP_ROUTER_ADDR, abi=ROUTER_ABI)

    @property
    def pair(self):
        return self.w3.eth.contract(address=PAIR_ADDRESS, abi=PAIR_ABI)

    def get_price(self, max_age: float | None = None) -> float:
        """
        On-chain price (BNB per IMT) from pool reserves.

        Cached for `price_cache_ttl` seconds so the many callers (candle validation,
        ticker, rate, fallbacks) share one RPC call instead of each making their own.
        `max_age=0` forces a fresh read (used where freshness matters). After an RPC
        failure the last known price is served without retrying for RPC_ERROR_BACKOFF s.
        """
        now = time.monotonic()
        ttl = (
            float(self.config.get("price_cache_ttl", PRICE_CACHE_TTL))
            if max_age is None
            else max_age
        )
        if self._last_price is not None and max_age is None:
            if now - self._price_ts < ttl or now < self._price_backoff_until:
                return self._last_price
        elif self._last_price is not None and ttl > 0 and now - self._price_ts < ttl:
            return self._last_price
        try:
            reserves = self.pair.functions.getReserves().call()
            reserve0 = float(reserves[0]) / 10**IMT_DECIMALS  # IMT reserve
            reserve1 = float(reserves[1]) / 10**18  # BNB reserve
            if reserve0 == 0:
                raise ExchangeError("Reserve0 is zero, cannot calculate price")
            price = reserve1 / reserve0  # Price in BNB per IMT
            self._last_price = price
            self._price_ts = now
            self._price_rpc_calls += 1
            if now - self._price_stats_ts >= 600:
                self.logger.info(f"RPC price calls in last 10 min: {self._price_rpc_calls}")
                self._price_rpc_calls, self._price_stats_ts = 0, now
            self.logger.debug(f"Fetched price from reserves: {price} BNB/IMT")
            return price
        except Exception as e:
            self._price_backoff_until = time.monotonic() + RPC_ERROR_BACKOFF
            self.logger.error(f"Price fetch error: {str(e)}")
            if self._last_price is not None:
                self.logger.warning(
                    f"Using last known price {self._last_price} due to fetch failure"
                )
                return self._last_price
            raise ExchangeError(f"Failed to fetch price and no cache available: {str(e)}")

    def get_ticker(self, pair: str, refresh: bool | None = None) -> dict:
        try:
            price = self.get_price()
            return {"bid": price, "ask": price, "last": price}
        except Exception as e:
            self.logger.error(f"Price fetch error: {e}")
            raise ExchangeError(f"Failed to fetch ticker: {e}")

    @retrier
    def buy(
        self, pair: str, amount: float, rate: float, time_in_force: str = "gtc", **kwargs
    ) -> dict:
        try:
            current_price = self.get_price()
            self.logger.debug(f"Current market price: {current_price} BNB/IMT")
            actual_rate = rate if rate > 0 else current_price

            required_bnb = amount * actual_rate
            amt_wei = self.w3.to_wei(required_bnb, "ether")
            self.logger.debug(
                f"Calculated: amount={amount} IMT, rate={actual_rate},"
                f"required_bnb={required_bnb}, amt_wei={amt_wei}"
            )

            balances = self.get_balances(max_age=0)
            bnb_balance = balances["BNB"]["free"]
            estimated_gas_cost = self.w3.to_wei(0.0005, "ether")
            total_required_wei = amt_wei + estimated_gas_cost
            if self.w3.from_wei(total_required_wei, "ether") > bnb_balance:
                raise ExchangeError(
                    f"Insufficient BNB balance: {bnb_balance} BNB available, "
                    f"{self.w3.from_wei(total_required_wei, 'ether')} BNB required"
                )

            path = [WBNB_ADDR, IMMORTALITY_ADDR]
            out = self.router.functions.getAmountsOut(amt_wei, path).call()
            min_out = int(out[-1] * (1 - self.slippage_tolerance))
            self.logger.debug(
                f"Estimated output: {out[-1] / 10**IMT_DECIMALS} IMT,"
                f"min_out={min_out / 10**IMT_DECIMALS}"
            )
            deadline = int(time.time()) + 180

            try:
                self.router.functions.swapExactETHForTokensSupportingFeeOnTransferTokens(
                    min_out, path, self.wallet, deadline
                ).call({"from": self.wallet, "value": amt_wei})
            except ContractLogicError as e:
                self.logger.error(f"Swap simulation reverted: {e}")
                raise ExchangeError(f"Swap simulation revert: {e}")

            fn = self.router.functions.swapExactETHForTokensSupportingFeeOnTransferTokens(
                min_out, path, self.wallet, deadline
            )
            tx_hash = send_tx(
                self.w3,
                fn,
                self.wallet,
                self.private_key,
                value=amt_wei,
            )

            self._balances_ts = -1e9
            decimals = self.token.functions.decimals().call()
            real_amount = out[-1] / (10**decimals)
            self._last_buy_amount = real_amount
            cost = required_bnb
            fee = {"cost": 0.0, "currency": "BNB"}

            return {
                "id": tx_hash,
                "status": "closed",
                "symbol": pair,
                "type": "market",
                "side": "buy",
                "price": actual_rate,
                "amount": real_amount,
                "filled": real_amount,
                "remaining": 0.0,
                "cost": cost,
                "fee": fee,
                "info": {"tx_hash": tx_hash},
            }
        except ExchangeError:
            raise
        except Exception as e:
            self.logger.error(f"Buy order failed for {pair}: {e}")
            raise ExchangeError(f"Buy order failed: {e}")

    @retrier
    def sell(
        self, pair: str, amount: float, rate: float, time_in_force: str = "gtc", **kwargs
    ) -> dict:
        try:
            reflection_rate = Decimal("0.10")
            net_ratio = Decimal("1.00") - reflection_rate
            decimals = self.token.functions.decimals().call()

            gross_amount = Decimal(amount) / net_ratio
            units = int(gross_amount * (10**decimals))

            balance = self.token.functions.balanceOf(self.wallet).call()
            if balance < units:
                raise ExchangeError(f"Insufficient IMT balance: {balance / 10**decimals}")

            allowance = self.token.functions.allowance(self.wallet, PANCAKESWAP_ROUTER_ADDR).call()
            if allowance < units:
                send_tx(
                    self.w3,
                    self.token.functions.approve(PANCAKESWAP_ROUTER_ADDR, units),
                    self.wallet,
                    self.private_key,
                )
                time.sleep(RPC_SYNC_DELAY_SECONDS)

            path = [IMMORTALITY_ADDR, WBNB_ADDR]
            out = self.router.functions.getAmountsOut(units, path).call()
            real_bnb = out[-1] / 10**18
            min_bnb = int(out[-1] * (1 - self.slippage_tolerance))
            deadline = int(time.time()) + 180

            try:
                self.router.functions.swapExactTokensForETHSupportingFeeOnTransferTokens(
                    units, min_bnb, path, self.wallet, deadline
                ).call({"from": self.wallet})
            except ContractLogicError as e:
                self.logger.error(f"Swap simulation reverted: {e}")
                raise ExchangeError(f"Swap simulation revert: {e}")

            fn = self.router.functions.swapExactTokensForETHSupportingFeeOnTransferTokens(
                units, min_bnb, path, self.wallet, deadline
            )
            tx_hash = send_tx(self.w3, fn, self.wallet, self.private_key)
            self._balances_ts = -1e9

            fee = {"cost": 0.0, "currency": "BNB"}

            return {
                "id": tx_hash,
                "status": "closed",
                "symbol": pair,
                "type": "market",
                "side": "sell",
                "price": rate,
                "amount": amount,
                "filled": amount,
                "remaining": 0.0,
                "cost": real_bnb,
                "fee": fee,
                "info": {"tx_hash": tx_hash},
            }

        except ExchangeError:
            raise

        except Exception as e:
            self.logger.error(f"Sell order failed for {pair}: {e}")
            raise ExchangeError(f"Sell order failed: {e}")

    def get_balances(self, max_age: float | None = None) -> dict:
        if getattr(self, "dry_run", False):
            stake_amt = float(self.config.get("stake_amount", 0.0))
            fake_balances = {
                "BNB": {"free": stake_amt, "used": 0.0, "total": stake_amt},
                "IMT": {"free": 0.0, "used": 0.0, "total": 0.0},
            }
            self.logger.info(f"Dry-run mode: faking BNB balance = {stake_amt}")
            return fake_balances

        now = time.monotonic()
        ttl = (
            float(self.config.get("balance_cache_ttl", BALANCE_CACHE_TTL))
            if max_age is None
            else max_age
        )
        if self._balances_cache is not None and ttl > 0 and now - self._balances_ts < ttl:
            return {k: dict(v) for k, v in self._balances_cache.items()}
        try:
            bnb_balance_wei = self.w3.eth.get_balance(self.wallet)
            bnb_balance = self.w3.from_wei(bnb_balance_wei, "ether")
            imt_balance_raw = self.token.functions.balanceOf(self.wallet).call()
            imt_balance = imt_balance_raw / 10**IMT_DECIMALS

            balances = {
                "BNB": {"free": float(bnb_balance), "used": 0.0, "total": float(bnb_balance)},
                "IMT": {"free": float(imt_balance), "used": 0.0, "total": float(imt_balance)},
            }
            self._balances_cache, self._balances_ts = balances, time.monotonic()
            self.logger.debug(f"Fetched balances: {balances}")
            self.logger.info(f"BNB Balance: {bnb_balance}, IMT Balance: {imt_balance}")
            return {k: dict(v) for k, v in balances.items()}
        except Exception as e:
            self.logger.error(f"Failed to fetch balances: {str(e)}")
            if self._balances_cache is not None and now - self._balances_ts < BALANCE_STALE_MAX:
                self.logger.warning(
                    f"Serving cached balances ({now - self._balances_ts:.0f}s old) after RPC error"
                )
                return {k: dict(v) for k, v in self._balances_cache.items()}
            raise ExchangeError(f"Failed to fetch balances: {str(e)}")

    def get_balance(self) -> dict:
        return self.get_balances()

    def get_conversion_rate(self, currency: str, stake_currency: str) -> float:
        if currency == stake_currency:
            return 1.0

        if currency == "IMT" and stake_currency == "BNB":
            try:
                rate = self.get_price()
                self.logger.debug(f"Conversion rate IMT→BNB: {rate}")
                return rate
            except Exception as e:
                self.logger.error(f"Failed to get conversion rate: {e}")
                return 1.0

        self.logger.warning(
            f"Conversion rate from {currency} to {stake_currency} not supported, defaulting to 1.0"
        )
        return 1.0

    def get_rate(self, pair: str, side: str = "buy", **kwargs) -> float:
        try:
            rate = self.get_price()
            self.logger.debug(f"get_rate() called for {pair}, side={side}: {rate}")
            return rate
        except ExchangeError as e:
            self.logger.error(f"Failed to get rate for {pair}: {e}")
            raise

    def get_min_pair_stake_amount(self, pair: str, *args, **kwargs) -> float:
        stake_amount = self.config.get("stake_amount")
        if stake_amount is not None:
            try:
                return float(stake_amount)
            except (ValueError, TypeError):
                self.logger.warning(
                    f"Invalid stake_amount in config: {stake_amount}, defaulting to 0"
                )
        default_min = 0.0001
        self.logger.debug(f"No valid stake_amount found, using default min stake {default_min}")
        return default_min

    def get_max_pair_stake_amount(self, pair: str, *args, **kwargs) -> float:
        stake_amount = self.config.get("stake_amount")
        if stake_amount is not None:
            try:
                return float(stake_amount)
            except (ValueError, TypeError):
                self.logger.warning(
                    f"Invalid stake_amount in config: {stake_amount}, defaulting to unlimited"
                )
        max_default = float("inf")
        self.logger.debug("No valid stake_amount found, using no max limit")
        return max_default

    def _simulate_order(self, pair: str, side: str, amount: float, price: float | None) -> dict:
        side = side.lower()
        ts = int(time.time() * 1000)
        mid = self.get_price(max_age=0)
        buy_fee = float(self.config.get("dry_run_buy_fee", DRY_RUN_BUY_FEE))
        sell_fee = float(self.config.get("dry_run_sell_fee", DRY_RUN_SELL_FEE))
        quoted = True
        try:
            if side == "buy":
                rate = price if price and price > 0 else mid
                bnb_in = amount * rate
                amt_wei = self.w3.to_wei(Decimal(str(bnb_in)), "ether")
                out = self.router.functions.getAmountsOut(
                    amt_wei, [WBNB_ADDR, IMMORTALITY_ADDR]
                ).call()
                filled = out[-1] / 10**IMT_DECIMALS * (1.0 - buy_fee)
                cost = bnb_in
            elif side == "sell":
                units = int(Decimal(str(amount)) * Decimal(str(1.0 - sell_fee)) * 10**IMT_DECIMALS)
                out = self.router.functions.getAmountsOut(
                    units, [IMMORTALITY_ADDR, WBNB_ADDR]
                ).call()
                filled = amount
                cost = out[-1] / 10**18
            else:
                raise OperationalException(f"Invalid side: {side}")
            if filled <= 0 or cost <= 0:
                raise ValueError(f"empty quote (filled={filled}, cost={cost})")
        except OperationalException:
            raise
        except Exception as e:
            quoted = False
            self.logger.warning(f"Dry run: router quote failed ({e}); using reserve price {mid}")
            filled, cost = amount, amount * mid

        eff_price = cost / filled
        impact_pct = (eff_price / mid - 1.0) * 100.0 if mid else 0.0
        self.logger.info(
            f"Dry run {side} {pair}: {filled:.8f} IMT for {cost:.8f} BNB "
            f"@ {eff_price:.12g} (mid {mid:.12g}, {impact_pct:+.2f}%, quoted={quoted})"
        )
        order = {
            "id": f"dry_run_{ts}_{side}",
            "symbol": pair,
            "type": "market",
            "side": side,
            "price": eff_price,
            "average": eff_price,
            "amount": filled,
            "filled": filled,
            "remaining": 0.0,
            "status": "closed",
            "cost": cost,
            "fee": {
                "cost": 0.0,
                "currency": self.config.get("stake_currency", "BNB"),
                "rate": 0.0,
                "type": "taker",
            },
            "timestamp": ts,
            "datetime": pd.to_datetime(ts, unit="ms").isoformat(),
            "info": {
                "dry_run": True,
                "quoted": quoted,
                "mid_price": mid,
                "price_impact_pct": impact_pct,
            },
        }
        if side == "buy":
            self._last_buy_amount = filled
        self._dry_orders[order["id"]] = order
        while len(self._dry_orders) > DRY_ORDER_CACHE_MAX:
            self._dry_orders.pop(next(iter(self._dry_orders)))
        return order

    def get_order(self, order_id: str, pair: str) -> dict:
        if str(order_id).startswith("dry_run_"):
            order = self._dry_orders.get(order_id)
            if order is None:
                self.logger.warning(f"Unknown dry-run order {order_id}; reporting it as closed")
                return {
                    "id": order_id,
                    "order_id": order_id,
                    "pair": pair,
                    "symbol": pair,
                    "status": "closed",
                    "filled": True,
                    "amount": None,
                    "price": None,
                    "info": {"dry_run": True},
                }
            return {**order, "order_id": order_id, "pair": pair}
        try:
            receipt = self.w3.eth.get_transaction_receipt(order_id)
            return {
                "order_id": order_id,
                "pair": pair,
                "status": "closed" if receipt.status == 1 else "failed",
                "filled": receipt.status == 1,
                "amount": None,
                "price": None,
            }
        except Exception as e:
            self.logger.error(f"Failed to fetch order {order_id}: {str(e)}")
            raise ExchangeError(f"Failed to fetch order: {str(e)}")

    def get_pair_quote_currency(self, pair: str) -> str:
        return pair.split("/")[1]

    def cancel_order(self, order_id: str, pair: str) -> dict:
        self.logger.warning(f"Cancel order not supported for {pair}/{order_id}")
        return {"order_id": order_id, "status": "canceled"}

    def get_fee(self, symbol: str, taker_or_maker: str = "taker", **kwargs) -> float:
        return 0.005

    @property
    def markets(self):
        return {
            "IMT/BNB": {
                "id": "IMT/BNB",
                "symbol": "IMT/BNB",
                "base": "IMT",
                "quote": "BNB",
                "active": True,
                "spot": True,
                "precision": {"amount": 8, "price": 8},
                "limits": {
                    "amount": {"min": 0.001, "max": 1000000},
                    "price": {"min": 0.00000001, "max": 1000000},
                },
            }
        }

    def create_order(
        self,
        pair: str,
        ordertype: str,
        side: str,
        amount: float,
        price: float | None = None,
        params: dict | None = None,
        **kwargs,
    ) -> dict:
        self.logger.warning("-= CREATE ORDER =-")
        params = params or {}
        otype = ordertype.lower()
        if otype == "limit":
            self.logger.warning(f"Converting limit order to market for {pair}")
            otype = "market"
        if otype != "market":
            raise OperationalException(
                f"Order type {ordertype} not supported; only 'market' or 'limit' allowed"
            )

        time_in_force = params.pop("time_in_force", "gtc")

        if getattr(self, "dry_run", False):
            return self._simulate_order(pair, side, amount, price)
        if side.lower() == "buy":
            actual_price = price if price is not None else 0.0
            return self.buy(pair, amount, actual_price, time_in_force=time_in_force, **params)
        elif side.lower() == "sell":
            actual_price = price if price is not None else 0.0
            return self.sell(pair, amount, actual_price, time_in_force=time_in_force, **params)
        else:
            raise OperationalException(f"Invalid side: {side}")

    def get_pair_base_currency(self, pair: str) -> str:
        if pair == "IMT/BNB":
            return "IMT"
        raise OperationalException(f"Unsupported pair: {pair}")

    def get_funding_fees(self, pair: str, **kwargs) -> float:
        side = kwargs.get("side", "")
        amount = kwargs.get("amount", 0.0)
        price = kwargs.get("price", 0.0)
        self.logger.debug(
            f"get_funding_fees called for {pair}, side={side}, amount={amount}, price={price}"
        )
        return 0.0

    def get_precision_amount(self, pair: str) -> int:
        return DECIMAL_PLACES

    def get_precision_price(self, pair: str) -> int:
        return DECIMAL_PLACES

    @property
    def precisionMode(self):
        return DECIMAL_PLACES

    @property
    def precision_mode_price(self):
        return self.precisionMode

    def get_contract_size(self, pair):
        return 1

    def check_order_canceled_empty(self, order: dict) -> bool:
        if not order:
            return True
        status = order.get("status", "").lower()
        return status in ["canceled", "cancelled", "not-found"]

    def order_has_fee(self, order: dict) -> bool:
        return True

    def extract_cost_curr_rate(self, *args, **kwargs) -> tuple[float, str, float]:
        self.logger.debug("Immortality.extract_cost_curr_rate called.")
        self.logger.debug(f"  args: {args}")
        self.logger.debug(f"  kwargs: {kwargs}")
        cost = 0.0
        currency = self.config.get("stake_currency", "BNB")
        rate = 0.0
        if args and len(args) > 0 and isinstance(args[0], dict) and "cost" in args[0]:
            fee_info = args[0]
            cost = float(fee_info.get("cost", 0.0))
            currency = fee_info.get("currency", currency)
        elif args and len(args) > 2:
            arg_cost = args[2]
            try:
                cost = float(arg_cost)
            except (ValueError, TypeError):
                self.logger.warning(f"Could not convert args[2] '{arg_cost}' to float for cost.")
                cost = 0.0
        self.logger.debug(
            f"extract_cost_curr_rate returning cost={cost}, currency={currency}, rate={rate}"
        )
        return cost, currency, rate

    @property
    def margin_mode(self):
        return None

    def get_liquidation_price(
        self,
        pair,
        amount,
        current_price=None,
        order_side=None,
        order_type=None,
        open_rate=None,
        is_short=None,
        stake_amount=None,
        leverage=None,
        wallet_balance=None,
    ):
        return None

    def log_latest_prices(self, all_pairs: list[str], timeframe: str):
        """
        Logs the latest close price for each trading pair.
        """
        for pair in all_pairs:
            try:
                df = self.latest_ohlcv.get((pair, timeframe, "spot"))
                if df is not None and not df.empty:
                    latest_price = df.iloc[-1]["close"]
                    self.logger.info(f"Current price for {pair}: {latest_price:.8f}")
                else:
                    self.logger.warning(f"No OHLCV data for {pair}")
            except Exception as e:
                self.logger.error(f"Failed to log price for {pair}: {e}")
