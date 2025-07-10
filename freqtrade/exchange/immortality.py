import asyncio
import json
import logging
import threading
import time
import traceback
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
IMMORTALITY_ADDR = Web3.to_checksum_address("0x2bF2141eD175f3236903cF07de33D7324871802D")
PAIR_ADDRESS = Web3.to_checksum_address("0xfA56E9AbcaA45207bE5E43cF475Ee061768CA915")  # IMT/BNB
MIN_INTERVAL = 600.0  # seconds between NodeReal calls
NODEREAL_FREE_URL = "https://open-platform.nodereal.io/{api_key}/pancakeswap-free/graphql"

# Trading parameters
IMT_DECIMALS = 8
BUY_BNB_AMOUNT = 0.004  # Match config.json stake_amount
SELL_IMT_QUANTITY = None  # Dynamically set in sell method based on buy amount or 10_000_000
RPC_SYNC_DELAY_SECONDS = 7
CACHE_DIR = "user_data/data/immortality"
CACHE_MAX_AGE_SECONDS = 90 * 24 * 3600  # 90 days

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


def fetch_ohlcv_nodereal(
    api_key: str, pair_address: str, limit: int, since_ms: int = 0
) -> list[list]:
    """Fetches OHLCV data from NodeReal's PancakeSwap GraphQL API."""
    query = f"""
    {{
      pairHourDatas(
        first: {limit},
        orderBy: hourStartUnix,
        orderDirection: desc,
        where: {{ pair: "{pair_address.lower()}", hourStartUnix_gt: {since_ms // 1000} }}
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
        logging.debug(f"Sending GraphQL query to {url}: {query}")
        resp = requests.post(url, json={"query": query}, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", {}).get("pairHourDatas", [])
        logging.debug(f"NodeReal API raw response: {data}")
    except requests.RequestException as e:
        logging.error(
            f"Failed to fetch OHLCV from NodeReal: {str(e)}, "
            f"Response: {resp.text if 'resp' in locals() else 'No response'}"
        )
        return []
    except ValueError as e:
        logging.error(
            f"Failed to parse NodeReal response: {str(e)}, "
            f"Response: {resp.text if 'resp' in locals() else 'No response'}"
        )
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
                    f"Invalid reserves for {pair_address} at {ts}: reserve0={r0}, "
                    f"reserve1={r1}, skipping"
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
        "ohlcv_candle_limit": 1000,
        "order_time_in_force": ["gtc"],
        "stoploss_on_exchange": False,
        "ws_enabled": True,
        "ws_auto_reconnect": True,
        "ws_reconnect_interval": 30,
        "watch_ohlcv": True,
    }

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
        self.api_key_value = None  # Cache HTTP API key
        self.ws_api_key_value = None  # Cache WebSocket API key
        self.w3 = Web3(Web3.HTTPProvider(BSC_RPC_URL.format(api_key=self.api_key)))
        if not self.w3.is_connected():
            raise OperationalException("Cannot connect to BSC RPC")
        self.latest_ohlcv: dict[tuple[str, str, str], pd.DataFrame] = {}
        self._last_call_time: int = 0
        self._min_interval: float = MIN_INTERVAL
        self.candle_builders: dict[tuple[str, str], CandleBuilder] = {}
        # Create cache directory
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Ensured cache directory exists: {CACHE_DIR}")
        try:
            self._configure_ws()
            self.logger.info("Immortality exchange initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize WebSocket: {str(e)}")
            raise OperationalException(f"Immortality initialization failed: {str(e)}")

    def _get_cache_path(self, pair: str, timeframe: str) -> str:
        """Returns the file path for the OHLCV cache."""
        pair_safe = pair.replace("/", "_")
        cache_path = str(Path(CACHE_DIR) / f"{pair_safe}_{timeframe}.csv")
        self.logger.debug(f"Cache path for {pair}/{timeframe}: {cache_path}")
        return cache_path

    def _load_cached_ohlcv(self, pair: str, timeframe: str) -> pd.DataFrame:
        """Loads cached OHLCV data from file if available and valid."""
        cache_path = self._get_cache_path(pair, timeframe)
        if not Path(cache_path).exists():
            self.logger.debug(f"No cache file found at {cache_path}")
            return pd.DataFrame()
        try:
            df = pd.read_csv(cache_path)
            df["date"] = pd.to_datetime(df["date"], utc=True)
            # Validate cache
            now_ms = int(time.time() * 1000)
            cutoff_ms = now_ms - (CACHE_MAX_AGE_SECONDS * 1000)
            if df.empty or df["date"].max().timestamp() * 1000 < cutoff_ms:
                self.logger.debug(f"Cache at {cache_path} is empty or too old")
                return pd.DataFrame()
            self.logger.debug(f"Loaded {len(df)} candles from cache at {cache_path}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load cache from {cache_path}: {str(e)}")
            return pd.DataFrame()

    def _save_cached_ohlcv(self, pair: str, timeframe: str, df: pd.DataFrame) -> None:
        """Saves OHLCV data to cache file."""
        cache_path = self._get_cache_path(pair, timeframe)
        try:
            df.to_csv(cache_path, index=False)
            self.logger.debug(f"Saved {len(df)} candles to cache at {cache_path}")
        except Exception as e:
            self.logger.error(f"Failed to save cache to {cache_path}: {str(e)}")

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
        key = exchange_conf.get("nodereal_api_key", "").strip()
        self.logger.info(f"NodeReal HTTP API key retrieved: {key!r}")
        if not key:
            self.logger.error(
                "Missing 'nodereal_api_key' in config.json. "
                "Please add it to the 'exchange' section."
            )
            raise OperationalException("NodeReal API key is required for OHLCV data retrieval.")
        self.logger.debug(f"Using NodeReal HTTP API key: {key}")
        self.api_key_value = key
        return key

    @property
    def ws_api_key(self) -> str:
        """Retrieve NodeReal WebSocket API key from configuration, falling back to HTTP key."""
        if self.ws_api_key_value is not None:
            return self.ws_api_key_value
        exchange_conf = (
            self.exchange_config
            if getattr(self, "exchange_config", None)
            else self.config.get("exchange", {})
        )
        key = exchange_conf.get("nodereal_ws_api_key", "").strip()
        if not key:
            key = self.api_key  # Fallback to HTTP API key
            self.logger.info(
                "No 'nodereal_ws_api_key' found in config.json, "
                "using 'nodereal_api_key' for WebSocket"
            )
        self.logger.info(f"NodeReal WebSocket API key retrieved: {key!r}")
        if not key:
            self.logger.error(
                "Missing WebSocket API key and no fallback 'nodereal_api_key' in config.json."
            )
            raise OperationalException("NodeReal WebSocket API key is required.")
        self.logger.debug(f"Using NodeReal WebSocket API key: {key}")
        self.ws_api_key_value = key
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
                "Missing 'key' (wallet address) in config.json. "
                "Please add it to the 'exchange' section."
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
                "Missing 'secret' (private key) in config.json. "
                "Please add it to the 'exchange' section."
            )
            raise OperationalException("Private key is required for trading.")
        self.logger.debug(f"Using private key: {secret}")
        return secret

    def _configure_ws(self, websocket_url: str | None = None):
        """Configures the WebSocket connection for real-time data."""
        ws_url = websocket_url or BSC_WS_URL.format(ws_api_key=self.ws_api_key)
        max_retries = 5
        attempt = 0
        while attempt < max_retries:
            try:
                self.logger.info(
                    f"Attempting WebSocket connection (attempt {attempt + 1}/{max_retries}) "
                    f"to {ws_url}"
                )
                self._exchange_ws = websocket.WebSocketApp(
                    ws_url,
                    on_open=self._on_ws_open,
                    on_message=self._on_ws_message,
                    on_error=self._on_ws_error,
                    on_close=self._on_ws_close,
                )
                self._ws_thread = threading.Thread(
                    target=lambda: self._exchange_ws.run_forever(
                        ping_interval=30,  # Send ping every 30 seconds
                        ping_timeout=10,  # Wait 10 seconds for pong
                    ),
                    daemon=True,
                )
                self._ws_thread.start()
                # Wait briefly to confirm connection
                time.sleep(2)
                if (
                    self._exchange_ws
                    and self._exchange_ws.sock
                    and self._exchange_ws.sock.connected
                ):
                    self.logger.info(f"WebSocket connection initialized to {ws_url}")
                    return
                else:
                    self.logger.warning("WebSocket connection not established, retrying")
            except Exception as e:
                self.logger.error(f"WebSocket connection failed: {str(e)}")
            # Clean up before retry
            if self._exchange_ws:
                try:
                    self._exchange_ws.close()
                except Exception as e:
                    self.logger.error(f"Error closing WebSocket during retry: {str(e)}")
                self._exchange_ws = None
            attempt += 1
            if attempt < max_retries:
                backoff = min(2**attempt, 16)  # Exponential backoff: 2s, 4s, 8s, 16s
                self.logger.info(f"Waiting {backoff}s before retry")
                time.sleep(backoff)
        self.logger.error(
            f"Max retries ({max_retries}) reached for WebSocket connection to {ws_url}. "
            "Real-time updates disabled."
        )
        self._exchange_ws = None

    def _on_ws_open(self, ws):
        """Handles WebSocket connection opening."""
        if not ws or not ws.sock or not ws.sock.connected:
            self.logger.error("WebSocket opened but socket is invalid or closed")
            self.ws_connection_reset()
            return
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
            self.ws_connection_reset()

    def _on_ws_message(self, ws, message):
        """Processes incoming WebSocket messages (Swap events)."""
        if not ws or not ws.sock or not ws.sock.connected:
            self.logger.error("WebSocket message received but socket is invalid or closed")
            self.ws_connection_reset()
            return
        try:
            data = json.loads(message)
            if "params" not in data or "result" not in data["params"]:
                return
            log = data["params"]["result"]
            topics = log.get("topics", [])
            if topics[0] != "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822":
                return
            self._process_swap_event(log)
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {str(e)}")
            self.ws_connection_reset()

    def _process_swap_event(self, log: dict) -> None:
        """Extracts swap event data and updates candles."""
        data_hex = log["data"]
        amount0In = int(data_hex[2:66], 16) / 10**8  # IMT
        amount1Out = int(data_hex[130:194], 16) / 10**18  # BNB
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
            if finalized_candle and tf == self.config.get("timeframe", "1h"):
                self.logger.debug(f"Finalized candle for {pair}/{tf}: {finalized_candle}")
                df = pd.DataFrame(
                    [
                        [
                            finalized_candle.timestamp,
                            finalized_candle.open,
                            finalized_candle.high,
                            finalized_candle.low,
                            finalized_candle.close,
                            finalized_candle.volume,
                        ]
                    ],
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                cached_df = self._load_cached_ohlcv(pair, tf)
                if not cached_df.empty:
                    df = (
                        pd.concat([cached_df, df])
                        .drop_duplicates(subset="date")
                        .sort_values("date")
                    )
                self._save_cached_ohlcv(pair, tf, df)

    def _on_ws_error(self, ws, error):
        """Handles WebSocket errors."""
        self.logger.error(f"WebSocket error: {str(error)}")
        self.ws_connection_reset()

    def _on_ws_close(self, ws, close_status_code, close_msg):
        """Handles WebSocket closure."""
        self.logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.ws_connection_reset()

    def _run_ws_forever(self):
        """Runs the WebSocket connection in a separate thread."""
        try:
            if self._exchange_ws:
                self._exchange_ws.run_forever(
                    ping_interval=30,
                    ping_timeout=10,
                )
        except Exception as e:
            self.logger.error(f"WebSocket thread crashed: {str(e)}")
            self.ws_connection_reset()

    async def watch_ohlcv(
        self, pair: str, timeframe: str, limit: int | None = None
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
                self.logger.info(f"Successful init of {pair}/{timeframe} with historic candle")

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
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error in watch_ohlcv for {pair}/{timeframe}: {str(e)}")
                await asyncio.sleep(0.5)

    def refresh_latest_ohlcv(self, pairs: list[str]) -> None:
        """Refreshes the latest OHLCV data, including real-time candles."""
        for item in pairs:
            try:
                pair = item[0] if isinstance(item, tuple) else item
                timeframe = (
                    item[1] if isinstance(item, tuple) else self.config.get("timeframe", "1h")
                )
                candle_type = item[2] if isinstance(item, tuple) and len(item) > 2 else "spot"

                ohlcv = self.get_ohlcv(pair, timeframe, limit=1000)
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

                self.latest_ohlcv[key] = ohlcv
                self.logger.info(f"Refreshed ohlcv for {pair}/{timeframe}, candles: {len(ohlcv)}")
            except Exception as e:
                self.logger.error(f"Failed to refresh OHLCV for {pair}/{timeframe}: {str(e)}")

    def interpolate_ohlcv(self, raw: list[list], timeframe: str) -> list[list]:
        """Interpolates hourly OHLCV data to a specified target timeframe."""
        target_seconds = self.timeframe_to_seconds(timeframe)
        if target_seconds >= 3600:
            return raw

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("date", inplace=True)

        df = (
            df.resample(f"{target_seconds // 60}T")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .interpolate(method="linear")
            .fillna(method="ffill")
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

    def _fetch_and_process_ohlcv(
        self, pair_str: str, timeframe: str, since_ms: int, limit: int
    ) -> pd.DataFrame:
        """Fetches and processes OHLCV data from NodeReal API."""
        try:
            fetch_limit = min(limit, 100)  # Limit API calls to avoid rate limits
            candles = fetch_ohlcv_nodereal(
                self.api_key, PAIR_ADDRESS, fetch_limit, since_ms=since_ms
            )
            if not candles:
                self.logger.warning(
                    f"No OHLCV data fetched for {pair_str}/{timeframe}, falling back to get_price"
                )
                try:
                    current_price = self.get_price()
                except ExchangeError as e:
                    self.logger.error(f"Failed to fetch price: {str(e)}")
                    current_price = 0.00000001  # Default price to prevent empty DataFrame
                ts = int(time.time() * 1000)
                candles = [
                    [
                        ts,
                        current_price,
                        current_price * 1.001,
                        current_price * 0.999,
                        current_price,
                        0.0,
                    ]
                ]

            # Interpolate if timeframe is less than 1 hour
            if self.timeframe_to_seconds(timeframe) < 3600:
                candles = self.interpolate_ohlcv(candles, timeframe)
                self.logger.debug(f"Interpolated to {len(candles)} candles for {timeframe}")

            df = ohlcv_to_dataframe(
                candles, timeframe, pair_str, fill_missing=True, drop_incomplete=True
            )
            return df
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV for {pair_str}/{timeframe}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    def _filter_and_cache_ohlcv(
        self,
        pair_str: str,
        timeframe: str,
        df: pd.DataFrame,
        cached_df: pd.DataFrame,
        since_ms: int,
    ) -> pd.DataFrame:
        """Filters, caches, and validates OHLCV data."""
        # Combine with cached data
        if not cached_df.empty:
            df = pd.concat([cached_df, df]).drop_duplicates(subset="date").sort_values("date")
            self.logger.debug(f"Combined {len(cached_df)} cached and {len(df)} new candles")

        # Filter by since_ms
        if since_ms:
            df = df[df["date"].astype("int64") // 10**6 >= since_ms]

        # Filter out old data (90 days)
        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - (CACHE_MAX_AGE_SECONDS * 1000)
        df = df[(df["date"].astype("int64") // 10**6) >= cutoff_ms]

        # Save cached data
        if not df.empty:
            self._save_cached_ohlcv(pair_str, timeframe, df)

        # Log a sample of the OHLCV DataFrame
        self.logger.debug(
            f"OHLCV DataFrame sample for {pair_str}/{timeframe}:\n{df.head(5).to_string()}"
        )

        # Compare latest close price with get_price
        try:
            if not df.empty:
                latest_close = df.iloc[-1]["close"]
                current_price = self.get_price()
                if current_price:
                    price_diff = abs(latest_close - current_price) / current_price * 100
                    self.logger.debug(
                        f"Price validation for {pair_str}/{timeframe}: "
                        f"Latest DataFrame close={latest_close}, get_price={current_price}, "
                        f"difference={price_diff:.2f}%"
                    )
        except ExchangeError as e:
            self.logger.error(f"Price validation failed: {str(e)}")

        self.logger.info(
            f"Retrieved {len(df)} candles for {pair_str}/{timeframe} "
            f"(cached: {len(cached_df)}, new: {len(df)})"
        )
        return df

    def get_ohlcv(
        self, pair: str | tuple, timeframe: str, since_ms: int = 0, limit: int = 1000
    ) -> pd.DataFrame:
        """Fetches historical OHLCV data, using cache if available."""
        # Handle tuple input from DataProvider
        if isinstance(pair, tuple):
            pair_str = pair[0]  # Extract pair (e.g., "IMT/BNB")
            self.logger.debug(f"Received invalid tuple pair {pair}, using pair_str={pair_str}")
        else:
            pair_str = pair
        self.logger.debug(
            f"Fetching OHLCV for pair={pair_str}, timeframe={timeframe}, "
            f"since_ms={since_ms}, limit={limit}"
        )
        if pair_str != "IMT/BNB":
            raise OperationalException(f"Pair {pair_str} not supported")

        api_key = self.api_key
        if not api_key:
            self.logger.error(
                "NodeReal API key not provided in config.json. "
                "Please add 'nodereal_api_key' to config.json."
            )
            raise OperationalException("NodeReal API key is required for OHLCV data retrieval.")

        # Load cached data
        cached_df = self._load_cached_ohlcv(pair_str, timeframe)
        last_cached_ms = 0
        if not cached_df.empty:
            last_cached_ms = int(cached_df["date"].max().timestamp() * 1000)
            self.logger.debug(
                f"Last cached timestamp: {last_cached_ms} ({cached_df['date'].max()})"
            )

        # Throttle API calls
        now_s = time.time()
        if now_s - self._last_call_time < self._min_interval:
            to_sleep = self._min_interval - (now_s - self._last_call_time)
            self.logger.info(f"Throttling NodeReal call; sleeping {to_sleep:.1f}s")
            time.sleep(to_sleep)
        self._last_call_time = int(time.time())

        # Fetch and process new data
        df = self._fetch_and_process_ohlcv(pair_str, timeframe, last_cached_ms, limit)

        # Filter, cache, and validate
        df = self._filter_and_cache_ohlcv(pair_str, timeframe, df, cached_df, since_ms)
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
            pair (str or tuple): The trading pair...
            (e.g., "IMT/BNB" or ("IMT/BNB", timeframe, candle_type)).
            timeframe (str, optional): Timeframe for the candles (e.g., "1m", "5m", "1h").
            since (int, optional): Start time in milliseconds since epoch.
            limit (int, optional): Maximum number of candles to fetch.

        Returns:
            pd.DataFrame: A pandas DataFrame containing OHLCV data.
        """
        # Handle tuple input
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
                f"No timeframe specified for {pair_str}, using default from config: {timeframe}"
            )
        self.logger.debug(
            f"Calling klines for pair={pair_str}, timeframe={timeframe}, "
            f"since={since}, limit={limit}"
        )
        try:
            self.validate_timeframes(timeframe)
            since_ms = since or 0
            limit = limit or 1000
            return self.get_ohlcv(pair_str, timeframe, since_ms, limit)
        except Exception as e:
            self.logger.error(f"Error processing klines for {pair_str}/{timeframe}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    def validate_timeframes(self, timeframe: str) -> None:
        """Validates that the timeframe is supported."""
        supported_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        if timeframe not in supported_timeframes:
            raise OperationalException(f"Timeframe {timeframe} not supported")

    @staticmethod
    def timeframe_to_seconds(timeframe: str) -> int:
        """Converts timeframe string to seconds."""
        units = {"m": 60, "h": 3600, "d": 86400}
        num = int(timeframe[:-1])
        unit = timeframe[-1]
        return num * units[unit]

    def ws_connection_reset(self):
        """Resets WebSocket connection."""
        self.logger.info("WebSocket reset requested")
        if self._exchange_ws:
            try:
                self._exchange_ws.close()
            except Exception as e:
                self.logger.error(f"Error closing WebSocket: {str(e)}")
            finally:
                self._exchange_ws = None
        # Delay to prevent rapid reconnection loops
        time.sleep(2)
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

    def get_proxy_currency(self) -> str:
        return self.config["stake_currency"]

    def get_markets(self):
        amount8 = 8  # Define amount8 for precision
        return {
            "IMT/BNB": {
                "name": "IMT/BNB",
                "id": "IMT/BNB",
                "symbol": "IMT/BNB",
                "base": "IMT",
                "quote": "BNB",
                "active": True,
                "spot": True,
                "precision": {"amount": amount8, "price": 8},
                "limits": {
                    "amount": {"min": 0.001, "max": 1000000},
                    "price": {"min": 0.00000001, "max": 1000000},
                },
            }
        }

    def reload_markets(self) -> None:
        self.logger.info("Reload markets called — no action required for Immortality.")

    def market_is_tradable(self, market: dict[str, Any]) -> bool:
        return market.get("active", False) and market.get("spot", False)

    def fetch_positions(self) -> list[dict]:
        """Returns an empty list as spot trading does not use positions."""
        self.logger.debug("Fetching positions called — returning empty list for spot trading")
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
            tokens = out[1] / 10**18  # BNB decimals
            self.logger.debug(f"Fetched price: {tokens:.2f} BNB/IMT")
            return tokens  # BNB/IMT price
        except Exception as e:
            self.logger.error(f"Price fetch error: {str(e)}")
            raise ExchangeError(f"Failed to fetch price: {str(e)}")

    def get_ticker(self, pair: str, refresh: bool | None = None) -> dict:
        try:
            price = self.get_price()
            return {"bid": price, "ask": price, "last": price}
        except ExchangeError as e:
            self.logger.error(f"Ticker fetch error: {str(e)}")
            raise ExchangeError(f"Failed to fetch ticker: {str(e)}")

    @retrier
    def buy(
        self, pair: str, amount: float, rate: float, time_in_force: str = "gtc", **kwargs
    ) -> dict:
        try:
            amt_wei = self.w3.to_wei(amount, "ether")
            out = self.router.functions.getAmountsOut(amt_wei, [WBNB_ADDR, IMMORTALITY_ADDR]).call()
            min_out = int(out[-1] * 0.9 * 0.95)
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
                    f"Insufficient IMT balance: "
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
            min_bnb = int(out[-1] * 0.95)
            fn = self.router.functions.swapExactTokensForETHSupportingFeeOnTransferTokens(
                units, min_bnb, [IMMORTALITY_ADDR, WBNB_ADDR], self.wallet, int(time.time()) + 120
            )
            tx = send_tx(self.w3, fn, self.wallet, self.private_key)
            return {"order_id": tx, "pair": pair, "amount": amount, "price": rate}
        except Exception as e:
            self.logger.error(f"Sell order failed for {pair}: {str(e)}")
            raise ExchangeError(f"Sell order failed: {str(e)}")

    def get_balances(self) -> dict:
        try:
            bnb_bal = self.w3.eth.get_balance(self.wallet) / 10**18
            imt_bal = self.token.functions.balanceOf(self.wallet).call() / (
                10 ** self.token.functions.decimals().call()
            )
            return {"BNB": bnb_bal, "IMT": imt_bal}
        except Exception as e:
            self.logger.error(f"Failed to fetch balances: {str(e)}")
            raise ExchangeError(f"Failed to fetch balances: {str(e)}")

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

    def get_fee(self, pair: str, fee_type: str) -> float:
        return 0.005

    @property
    def markets(self):
        return self.get_markets()
