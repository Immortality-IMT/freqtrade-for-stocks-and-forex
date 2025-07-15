import asyncio
import json
import logging
import threading
import time
import traceback
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
import requests
import websocket
from web3 import Web3

from freqtrade.data.converter import ohlcv_to_dataframe
from freqtrade.exceptions import ExchangeError, InsufficientFundsError, OperationalException
from freqtrade.exchange.common import retrier
from freqtrade.exchange.stockexchange import Stockexchange


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


def get_nodereal_graphql_url(api_key: str) -> str:
    """Returns the NodeReal GraphQL URL with the provided API key."""
    return NODEREAL_FREE_URL.format(api_key=api_key)


def init_websocket(self):
    """Initializes WebSocket connection using NodeReal URL."""
    max_retries = 5
    ws_url = BSC_WS_URL.format(ws_api_key=self.ws_api_key)
    for attempt in range(max_retries):
        try:
            self.logger.info(
                f"Attempting WebSocket connection (attempt {attempt + 1}/{max_retries}) to {ws_url}"
            )
            self.ws = websocket.WebSocket()
            self.ws.connect(ws_url)
            self.logger.info(f"WebSocket connection established to {ws_url}")
            break
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            else:
                self.logger.error("Max retries reached, giving up on WebSocket connection")


def fetch_ohlcv_nodereal(api_key: str, pair_address: str, limit: int) -> list[list]:
    """Fetches OHLCV data from NodeReal's PancakeSwap GraphQL API."""
    query = f"""
    {{
      pairHourDatas(
        first: {limit},
        orderBy: hourStartUnix,
        orderDirection: desc,
        where: {{ pair: "{pair_address.lower()}" }}
      ) {{
        hourStartUnix
        reserve0
        reserve1
        hourlyVolumeToken0
      }}
    }}
    """
    url = NODEREAL_FREE_URL.format(api_key=api_key)
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(url, json={"query": query}, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", {}).get("pairHourDatas", [])
        logging.debug(f"NodeReal API raw response: {data}")
    except (requests.RequestException, ValueError) as e:
        logging.error(f"Failed to fetch OHLCV from NodeReal: {str(e)}")
        return []

    candles: list[list] = []
    prev_price = None
    for entry in reversed(data):  # Reverse to chronological order
        try:
            ts = int(entry["hourStartUnix"]) * 1000
            r0 = float(entry["reserve0"])  # IMT
            r1 = float(entry["reserve1"])  # BNB
            if r0 <= 0 or r1 <= 0:
                logging.warning(
                    "Invalid reserves: "
                    f"{pair_address} at {ts}: reserve0={r0}, reserve1={r1}, skipping"
                )
                continue
            price = r1 / r0  # BNB/IMT
            open_p = prev_price if prev_price is not None else price
            close_p = price
            high_p = max(open_p, close_p) * 1.001
            low_p = min(open_p, close_p) * 0.999
            volume = float(entry["hourlyVolumeToken0"]) if entry["hourlyVolumeToken0"] else 0.0
            candles.append([ts, open_p, high_p, low_p, close_p, volume])
            prev_price = price
            logging.debug(f"OHLCV entry for {pair_address} at {ts}: price={price}, volume={volume}")
        except (KeyError, ValueError) as e:
            logging.warning(f"Skipping invalid OHLCV entry: {str(e)}")
            continue
    return candles


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
        logging.error(f"Transaction failed: {str(e)}")
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
        self.slippage_tolerance = config.get("slippage_tolerance", 0.05)  # Default 5%
        self.api_key_value = None  # Cache HTTP API key
        self.ws_api_key_value = None  # Cache WebSocket API key
        self.w3 = Web3(Web3.HTTPProvider(BSC_RPC_URL.format(api_key=self.api_key)))
        if not self.w3.is_connected():
            raise OperationalException("Cannot connect to BSC RPC")
        self.latest_ohlcv: dict[tuple[str, str, str], pd.DataFrame] = {}
        self._last_call_time = 0.0
        self._last_price: float | None = None
        self._min_interval = MIN_INTERVAL
        self.candle_builders: dict[tuple[str, str], CandleBuilder] = {}
        cache_path = "user_data/data/immortality/cache_IMT-BNB_5m.csv"
        try:
            df = pd.read_csv(cache_path, parse_dates=["date"])
            self.latest_ohlcv[("IMT/BNB", "5m", "spot")] = df
            self.logger.info(f"Loaded OHLCV cache from {cache_path}, {len(df)} rows")
        except FileNotFoundError:
            self.logger.info(f"No cache file found at {cache_path}, starting empty")
        try:
            self._configure_ws()
            self.logger.info("Immortality exchange initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize WebSocket: {str(e)}")
            raise OperationalException(f"Immortality initialization failed: {str(e)}")

    @property
    def name(self):
        return "immortality"

    def _configure_ws(self, websocket_url: str | None = None):
        """Configures the WebSocket connection for real-time data."""
        ws_url = websocket_url or BSC_WS_URL.format(ws_api_key=self.ws_api_key)
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.logger.info(
                    f"Attempt WebSocket connect (attempt {attempt + 1}/{max_retries}) to {ws_url}"
                )
                self._exchange_ws = websocket.WebSocketApp(
                    ws_url,
                    on_open=self._on_ws_open,
                    on_message=self._on_ws_message,
                    on_error=self._on_ws_error,
                    on_close=self._on_ws_close,
                )
                self._ws_thread = threading.Thread(target=self._run_ws_forever, daemon=True)
                self._ws_thread.start()
                self.logger.info(f"WebSocket connection initialized to {ws_url}")
                break
            except Exception as e:
                self.logger.error(f"WebSocket connection failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff: 1s, 2s, 4s, etc.
                else:
                    self.logger.error("Max retries reached, giving up on WebSocket connection")

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
    def ws_api_key(self) -> str:
        """Retrieve NodeReal WebSocket API key from configuration, falling back to HTTP key."""
        return self.api_key

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
        self.logger.info(f"Private key retrieved: {secret!r}")
        if not secret:
            self.logger.error(
                "Missing 'secret' (private key) in config.json. Add key to the 'exchange' section."
            )
            raise OperationalException("Private key is required for trading.")
        self.logger.debug(f"Using private key: {secret}")
        return secret

    def _on_ws_open(self, ws):
        """Handles WebSocket connection opening."""
        swap_topic = (
            "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"  # Swap event
        )
        subscription = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_subscribe",
            "params": ["logs", {"address": PAIR_ADDRESS, "topics": [swap_topic]}],
        }
        try:
            ws.send(json.dumps(subscription))
            self.logger.info(f"Subscribed to Swap events for pair {PAIR_ADDRESS}")
        except Exception as e:
            self.logger.error(f"WebSocket subscription failed: {str(e)}")

    def _on_ws_message(self, ws, message):
        """Processes incoming WebSocket messages (Swap events)."""
        try:
            data = json.loads(message)
            if "params" not in data or "result" not in data["params"]:
                return
            log = data["params"]["result"]
            topics = log.get("topics", [])
            if topics[0] != "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822":
                return

            data_hex = log["data"]
            amount0In = int(data_hex[2:66], 16) / 10**8  # IMT
            amount1Out = int(data_hex[130:194], 16) / 10**18  # BNB
            # Handle both IMT->BNB and BNB->IMT swaps
            if amount0In > 0 and amount1Out > 0:
                price = amount1Out / amount0In  # BNB/IMT
                volume = amount0In
            elif amount0In == 0 and amount1Out == 0:
                amount0Out = int(data_hex[66:130], 16) / 10**8
                amount1In = int(data_hex[194:258], 16) / 10**18
                if amount0Out == 0 or amount1In == 0:
                    return
                price = amount1In / amount0Out
                volume = amount0Out
            else:
                return

            timestamp_ms = int(time.time() * 1000)
            pair = "IMT/BNB"
            supported_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            for tf in supported_timeframes:
                key = (pair, tf)
                if key not in self.candle_builders:
                    self.candle_builders[key] = CandleBuilder(tf)
                builder = self.candle_builders[key]
                finalized_candle = builder.update(price, volume, timestamp_ms)
                if finalized_candle:
                    self.logger.debug(f"Finalized candle for {pair}/{tf}: {finalized_candle}")
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {str(e)}")

    def _on_ws_error(self, ws, error):
        """Handles WebSocket errors."""
        self.logger.error(f"WebSocket error: {error}")
        self.ws_connection_reset()

    def _on_ws_close(self, ws, close_status_code, close_msg):
        """Handles WebSocket closure."""
        self.logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.ws_connection_reset()

    def _run_ws_forever(self):
        """Runs the WebSocket connection in a separate thread."""
        try:
            self._exchange_ws.run_forever()
        except Exception as e:
            self.logger.error(f"WebSocket thread crashed: {str(e)}")
            self.ws_connection_reset()

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
        """Refreshes the latest OHLCV data, including real-time candles."""
        for item in pairs:
            try:
                pair = item[0] if isinstance(item, tuple) else item
                timeframe = (
                    item[1] if isinstance(item, tuple) else self.config.get("timeframe", "1h")
                )
                candle_type = item[2] if isinstance(item, tuple) and len(item) > 2 else "spot"

                ohlcv = self.get_ohlcv(pair, timeframe, limit=200)
                key = (pair, timeframe, candle_type)

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

                # Prune cache to keep only the latest 200 candles
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

    def _append_synthetic_candle_if_needed(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Append a synthetic flat candle if the latest one is outdated."""
        try:
            last_ts = df["date"].iloc[-1]
            now_utc = pd.Timestamp.utcnow()
            interval = pd.Timedelta(seconds=self.timeframe_to_seconds(timeframe))

            if now_utc - last_ts >= interval:
                price = df["close"].iloc[-1]
                new_row = {
                    "date": now_utc,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 0.0,
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                self.logger.debug(f"Appended synthetic candle at {now_utc} with price {price}")
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

    def prepare_cache_and_throttle(
        self, pair: str | tuple, timeframe: str, since_ms: int
    ) -> tuple[str, int, pd.DataFrame | None]:
        if isinstance(pair, tuple):
            pair_str = pair[0]
            self.logger.debug(f"Received tuple pair {pair}, using pair_str={pair_str}")
        else:
            pair_str = pair

        self.logger.debug(
            "Fetching OHLCV: pair=%s, timeframe=%s, since_ms=%s",
            pair_str,
            timeframe,
            since_ms,
        )

        if pair_str != "IMT/BNB":
            raise OperationalException(f"Pair {pair_str} not supported")

        if not self.api_key:
            self.logger.error("NodeReal API key not provided in config.")
            raise OperationalException("NodeReal API key is required for OHLCV data retrieval.")

        cache_key = (pair_str, timeframe, "spot")
        cached_df = self.latest_ohlcv.get(cache_key)

        if cached_df is not None and not cached_df.empty:
            latest_ts_ms = int(cached_df["date"].iloc[-1].timestamp() * 1000)
            if since_ms < latest_ts_ms:
                since_ms = latest_ts_ms
                self.logger.debug(
                    f"Using cached latest timestamp: {since_ms} for {pair_str}/{timeframe}"
                )

        now_s = time.time()
        if now_s - self._last_call_time < self._min_interval:
            to_sleep = self._min_interval - (now_s - self._last_call_time)
            self.logger.info("Throttling NodeReal call; sleeping %.1fs", to_sleep)
            time.sleep(to_sleep)
        self._last_call_time = time.time()

        return pair_str, since_ms, cached_df

    def process_and_filter_dataframe(
        self,
        df: pd.DataFrame,
        cached_df: pd.DataFrame | None,
        since_ms: int,
        pair_str: str,
        timeframe: str,
    ) -> pd.DataFrame:
        if since_ms:
            df = df[df["date"].astype("int64") // 10**6 >= since_ms]

        cutoff_ms = int(time.time() * 1000) - (90 * 24 * 3600 * 1000)
        df = df[(df["date"].astype("int64") // 10**6) >= cutoff_ms]

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

    def get_ohlcv(
        self, pair: str | tuple, timeframe: str, since_ms: int = 0, limit: int = 200
    ) -> pd.DataFrame:
        """Fetches historical OHLCV data, merges with cache, generates synthetic candles on failure,
        and persists the latest 200 candles to disk."""
        try:
            # Prepare cache and throttle timing
            pair_str, since_ms, cached_df = self.prepare_cache_and_throttle(
                pair, timeframe, since_ms
            )

            # Determine fetch limit: full or incremental
            fetch_limit = 10 if since_ms > 0 else limit
            candles = fetch_ohlcv_nodereal(self.api_key, PAIR_ADDRESS, fetch_limit)

            if not candles:
                self.logger.warning(
                    "No OHLCV data fetched for %s/%s, checking cache", pair_str, timeframe
                )
                if cached_df is not None and not cached_df.empty:
                    last_candle = cached_df.iloc[-1]
                    interval_seconds = self.timeframe_to_seconds(timeframe)
                    synthetic_candles = self.create_synthetic_candles(
                        last_candle,
                        last_candle["date"] + pd.Timedelta(seconds=interval_seconds),
                        pd.Timestamp.utcnow(),
                        interval_seconds,
                    )
                    if synthetic_candles:
                        df = ohlcv_to_dataframe(
                            synthetic_candles,
                            timeframe,
                            pair_str,
                            fill_missing=True,
                            drop_incomplete=True,
                        )
                        df = (
                            pd.concat([cached_df, df])
                            .drop_duplicates(subset="date")
                            .sort_values("date")
                        )
                        self.logger.info(
                            "Generated %d synthetic candles for %s/%s",
                            len(synthetic_candles),
                            pair_str,
                            timeframe,
                        )
                        # Persist and return
                        return self._persist_and_return(df, pair_str, timeframe)
                return self._get_fallback_candle(timeframe, pair_str)

            # Interpolate if timeframe < 1h
            if self.timeframe_to_seconds(timeframe) < 3600:
                raw_cap = 50 if since_ms == 0 else 10
                recent = candles[-raw_cap:]
                candles = self.interpolate_ohlcv(recent, timeframe)
                self.logger.debug("Interpolated to %d candles for %s", len(candles), timeframe)

            # Build DataFrame
            df = ohlcv_to_dataframe(
                candles, timeframe, pair_str, fill_missing=True, drop_incomplete=True
            )

            # Merge, filter, validate, append synthetic, trim to 200
            df = self.process_and_filter_dataframe(df, cached_df, since_ms, pair_str, timeframe)

            # Persist to disk and return final DataFrame
            return self._persist_and_return(df, pair_str, timeframe)

        except requests.HTTPError as e:
            if e.response.status_code == 429:
                return self.rate_limit_error_handler(cached_df, timeframe, pair_str)
            self.logger.error(
                "HTTP error fetching OHLCV for %s/%s: %s", pair_str, timeframe, str(e)
            )
            return self._get_fallback_candle(timeframe, pair_str)
        except Exception as e:
            self.logger.error("Error fetching OHLCV for %s/%s: %s", pair_str, timeframe, str(e))
            self.logger.error("Traceback: %s", traceback.format_exc())
            return pd.DataFrame()

    def _persist_and_return(self, df: pd.DataFrame, pair_str: str, timeframe: str) -> pd.DataFrame:
        """Helper to trim, cache in memory, write to disk, and return."""
        # Keep only last 200 candles
        df = df.tail(200)
        cache_key = (pair_str, timeframe, "spot")
        self.latest_ohlcv[cache_key] = df

        # Persist to CSV
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
        """
        Fetches OHLCV data for a given pair and timeframe.

        Args:
            pair (str or tuple): Trading pair
            (e.g., "IMT/BNB" or ("IMT/BNB", timeframe, candle_type)).
            timeframe (str, optional): Timeframe for the candles (e.g., "1m", "5m", "1h").
            Defaults to config timeframe.
            since (int, optional): Start time in milliseconds since epoch. Defaults to None.
            limit (int, optional): Maximum number of candles to fetch. Defaults to None.

        Returns:
            pd.DataFrame: A pandas DataFrame containing OHLCV data.
        """
        # Handle tuple input from DataProvider
        if isinstance(pair, tuple):
            pair_str = pair[0]  # Extract pair (e.g., "IMT/BNB")
            if len(pair) > 1 and pair[1] and timeframe is None:
                timeframe = pair[1]  # Use timeframe from tuple if provided
                self.logger.debug(f"Using timeframe {timeframe} from tuple {pair}")
        else:
            pair_str = pair
        if timeframe is None:
            timeframe = self.config.get("timeframe", "1h")
            self.logger.info(
                f"No timeframe provided for {pair_str}, using default from config: {timeframe}"
            )
        self.logger.debug(
            f"Calling klines: pair={pair_str}, timeframe={timeframe}, since={since}, limit={limit}"
        )
        try:
            self.validate_timeframes(timeframe)
            since_ms = since if since is not None else 0
            limit = min(limit or cast(int, self._ft_has_default["ohlcv_candle_limit"]), 200)
            return self.get_ohlcv(pair_str, timeframe, since_ms, limit)
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

    def ws_connection_reset(self) -> None:
        """Resets the WebSocket connection."""
        self.logger.info("WebSocket reset requested")
        if self._exchange_ws:
            try:
                self._exchange_ws.close()
            except Exception as e:
                self.logger.error(f"Error closing WebSocket: {str(e)}")
            self._exchange_ws = None
        try:
            self._configure_ws()
        except Exception as e:
            self.logger.error(f"Failed to reset WebSocket: {str(e)}")

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

    def get_price(self) -> float:
        try:
            amt = self.w3.to_wei(1, "ether")
            out = self.router.functions.getAmountsOut(amt, [IMMORTALITY_ADDR, WBNB_ADDR]).call()
            price = out[1] / 10**18
            # **Cache it on success**
            self._last_price = price
            self.logger.debug(f"Fetched price: {price} BNB/IMT")
            return price
        except Exception as e:
            self.logger.error(f"Price fetch error: {str(e)}")
            # **Fall back to last known good price if available**
            if self._last_price is not None:
                self.logger.warning(
                    f"Using last known price {self._last_price} due to fetch failure"
                )
                return self._last_price
            # **If no cache, escalate the error**
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
            amt_wei = self.w3.to_wei(amount, "ether")
            out = self.router.functions.getAmountsOut(amt_wei, [WBNB_ADDR, IMMORTALITY_ADDR]).call()
            # min_out = int(out[-1] * 0.9 * 0.95)
            min_out = int(out[-1] * (1 - self.slippage_tolerance))
            fn = self.router.functions.swapExactETHForTokensSupportingFeeOnTransferTokens(
                min_out, [WBNB_ADDR, IMMORTALITY_ADDR], self.wallet, int(time.time()) + 120
            )
            tx = send_tx(self.w3, fn, self.wallet, self.private_key, value=amt_wei)
            self._last_buy_amount = out[-1] / (
                10 ** self.token.functions.decimals().call()
            )  # Store IMT amount
            return {"order_id": tx, "pair": pair, "amount": amount, "price": rate}
        except Exception as e:
            self.logger.error(f"Buy order failed for {pair}: {str(e)}")
            raise ExchangeError(f"Buy order failed: {str(e)}")

    @retrier
    def sell(
        self, pair: str, amount: float, rate: float, time_in_force: str = "gtc", **kwargs
    ) -> dict:
        try:
            # Use last buy amount if SELL_IMT_QUANTITY is None
            units = (
                int(self._last_buy_amount * (10 ** self.token.functions.decimals().call()))
                if SELL_IMT_QUANTITY is None
                else int(SELL_IMT_QUANTITY * (10 ** self.token.functions.decimals().call()))
            )
            balance = self.token.functions.balanceOf(self.wallet).call()
            if balance < units:
                raise InsufficientFundsError(
                    "Insufficient IMT balance: "
                    f"{balance / (10 ** self.token.functions.decimals().call())} IMT available"
                )
            if self.token.functions.allowance(self.wallet, PANCAKESWAP_ROUTER_ADDR).call() < units:
                send_tx(
                    self.w3,
                    self.token.functions.approve(PANCAKESWAP_ROUTER_ADDR, units),
                    self.wallet,
                    self.private_key,
                )
                time.sleep(RPC_SYNC_DELAY_SECONDS)
            out = self.router.functions.getAmountsOut(units, [IMMORTALITY_ADDR, WBNB_ADDR]).call()
            # min_bnb = int(out[-1] * 0.95)
            min_bnb = int(out[-1] * (1 - self.slippage_tolerance))
            fn = self.router.functions.swapExactTokensForETHSupportingFeeOnTransferTokens(
                units, min_bnb, [IMMORTALITY_ADDR, WBNB_ADDR], self.wallet, int(time.time()) + 120
            )
            tx = send_tx(self.w3, fn, self.wallet, self.private_key)
            return {"order_id": tx, "pair": pair, "amount": amount, "price": rate}
        except Exception as e:
            self.logger.error(f"Sell order failed for {pair}: {str(e)}")
            raise ExchangeError(f"Sell order failed: {str(e)}")

    def get_balances(self) -> dict:
        """
        Fetch account balances for BNB and IMT.

        Returns:
            dict: Dictionary of balances in Freqtrade-compatible format.
                  In dry run mode, returns a fake BNB balance equal to stake_amount.
        """
        # —————————————————————————————————————————————————————————————
        # In dry run mode, pretend we have exactly stake_amount BNB available
        # so Freqtrade can create its entry orders unimpeded.
        if getattr(self, "dry_run", False):
            stake_amt = float(self.config.get("stake_amount", 0.0))
            fake_balances = {
                "BNB": {"free": stake_amt, "used": 0.0, "total": stake_amt},
                "IMT": {"free": 0.0, "used": 0.0, "total": 0.0},
            }
            self.logger.info(f"Dry-run mode: faking BNB balance = {stake_amt}")
            return fake_balances
        # —————————————————————————————————————————————————————————————
        # Live mode: query on chain balances as before
        try:
            bnb_balance_wei = self.w3.eth.get_balance(self.wallet)
            bnb_balance = self.w3.from_wei(bnb_balance_wei, "ether")
            imt_balance_raw = self.token.functions.balanceOf(self.wallet).call()
            imt_balance = imt_balance_raw / 10**IMT_DECIMALS

            balances = {
                "BNB": {"free": float(bnb_balance), "used": 0.0, "total": float(bnb_balance)},
                "IMT": {"free": float(imt_balance), "used": 0.0, "total": float(imt_balance)},
            }
            self.logger.debug(f"Fetched balances: {balances}")
            self.logger.info(f"BNB Balance: {bnb_balance}, IMT Balance: {imt_balance}")
            return balances
        except Exception as e:
            self.logger.error(f"Failed to fetch balances: {str(e)}")
            raise ExchangeError(f"Failed to fetch balances: {str(e)}")

    def get_balance(self) -> dict:
        return self.get_balances()

    def get_conversion_rate(self, currency: str, stake_currency: str) -> float:
        """
        Return the conversion rate from the given currency to the stake currency.
        Used by the RPC balance endpoint to estimate needed stake amounts.
        """
        # In dry run or for unsupported pairs, just return 1.0
        if currency == stake_currency:
            return 1.0

        # For IMT BNB, use your on chain price fetch
        if currency == "IMT" and stake_currency == "BNB":
            try:
                rate = self.get_price()
                self.logger.debug(f"Conversion rate IMT→BNB: {rate}")
                return rate
            except Exception as e:
                self.logger.error(f"Failed to get conversion rate: {e}")
                return 1.0

        # Fallback for any other currencies
        self.logger.warning(
            f"Conversion rate from {currency} to {stake_currency} not supported, defaulting to 1.0"
        )
        return 1.0

    def get_rate(self, pair: str, side: str = "buy", **kwargs) -> float:
        """
        Return the current market rate for the given pair.
        Freqtrade calls this during entry validation (get_valid_enter_price_and_stake).
        """
        # Now get_price either returns a real price or raises
        try:
            # We only support IMT/BNB, and `get_price()` fetches that price (BNB per IMT).
            rate = self.get_price()
            self.logger.debug(f"get_rate() called for {pair}, side={side}: {rate}")
            return rate
        except ExchangeError as e:
            self.logger.error(f"Failed to get rate for {pair}: {e}")
            # Bubble up so the RPC layer can handle missing price,
            # instead of returning zero and triggering a ZeroDivisionError
            raise

    def get_min_pair_stake_amount(self, pair: str, *args, **kwargs) -> float:
        # def get_min_pair_stake_amount(self, pair: str) -> float:
        """
        Return the minimum stake amount (in BNB) for the given pair.
        Freqtrade uses this to validate the minimum required BNB per trade.
        """
        # If stake_amount is configured globally, use that
        stake_amount = self.config.get("stake_amount")
        if stake_amount is not None:
            try:
                return float(stake_amount)
            except (ValueError, TypeError):
                self.logger.warning(
                    f"Invalid stake_amount in config: {stake_amount}, defaulting to 0"
                )
        # Fallback: require at least a tiny amount
        default_min = 0.0001
        self.logger.debug(f"No valid stake_amount found, using default min stake {default_min}")
        return default_min

    def get_max_pair_stake_amount(self, pair: str, *args, **kwargs) -> float:
        """
        Return the maximum stake amount (in BNB) for the given pair.
        Accepts extra args/kwargs from Freqtrade without error.
        """
        # If a global stake_amount is configured, use that as max too
        stake_amount = self.config.get("stake_amount")
        if stake_amount is not None:
            try:
                return float(stake_amount)
            except (ValueError, TypeError):
                self.logger.warning(
                    f"Invalid stake_amount in config: {stake_amount}, defaulting to unlimited"
                )
        # Fallback: no enforced max (use a very large number)
        max_default = float("inf")
        self.logger.debug("No valid stake_amount found, using no max limit")
        return max_default

    def get_order(self, order_id: str, pair: str) -> dict:
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
        # inspect `taker_or_maker` here
        return 0.005  # 0.5% default fee

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
        self.logger.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CREATE ORDER")
        params = params or {}
        otype = ordertype.lower()
        # Convert limit orders to market
        if otype == "limit":
            self.logger.warning(f"Converting limit order to market for {pair}")
            otype = "market"
        if otype != "market":
            raise OperationalException(
                f"Order type {ordertype} not supported; only 'market' or 'limit' allowed"
            )

        time_in_force = params.pop("time_in_force", "gtc")

        # ——— Dry run branch ———
        if getattr(self, "dry_run", False):
            self.logger.info(
                f"Dry run: simulating {side} order for {pair}, amount={amount}, price={price}"
            )
            current_price = self.get_price()
            fee_rate = self.get_fee(pair, fee_type="taker")
            cost = amount * current_price
            fee_cost = cost * fee_rate
            ts = int(time.time() * 1000)
            return {
                "id": f"dry_run_{ts}_{side}",
                "symbol": pair,
                "type": "market",
                "side": side,
                "price": current_price,
                "amount": amount,
                "filled": amount,
                "remaining": 0.0,
                "status": "closed",
                "cost": amount * current_price,
                "fee": {
                    "cost": fee_cost,
                    "currency": self.config.get("stake_currency", "BNB"),
                    "rate": fee_rate,
                    "type": "taker",
                },
                "timestamp": ts,
                "datetime": pd.to_datetime(ts, unit="ms").isoformat(),
                "info": {"dry_run": True},
            }
        # ——— Live mode branch ———
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
        """
        Return the precision (decimal places) for the amount of the base asset.
        Freqtrade uses this when rounding order amounts.
        """
        return DECIMAL_PLACES

    def get_precision_price(self, pair: str) -> int:
        """
        Return the precision for the price of the quote asset.
        """
        return DECIMAL_PLACES  # Or adjust as needed for BNB

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
