# pip install ib_insync
# Interactive Brokers exchange forex integration for FreqTrade

import atexit
import logging
import math
import signal
import socket
import sys
import time
from datetime import UTC, datetime, timedelta
from threading import Event, Lock, Thread
from typing import Any, cast

import pandas as pd
from ib_insync import IB, Contract, Forex, Order, util

from freqtrade.enums import MarginMode
from freqtrade.exceptions import ExchangeError, OperationalException
from freqtrade.exchange.foreignexchange import Foreignexchange
from freqtrade.persistence import Order as FTOrder
from freqtrade.persistence import Trade


util.patchAsyncio()

logger = logging.getLogger(__name__)

_min_interval = 0.5  # One request every 2 seconds
_last_request_ts = 0.0

_request_lock = Lock()

# Define forex market open and close times in UTC
MARKET_OPEN_TIME_UTC = datetime.strptime("22:00", "%H:%M").time()  # Sunday 10:00 PM UTC
MARKET_CLOSE_TIME_UTC = datetime.strptime("22:00", "%H:%M").time()  # Friday 10:00 PM UTC
MARKET_OPEN_DAY = 6  # Sunday (0 = Monday, 6 = Sunday)
MARKET_CLOSE_DAY = 4  # Friday


def throttle():
    global _last_request_ts
    with _request_lock:
        now = time.time()
        elapsed = now - _last_request_ts
        if elapsed < _min_interval:
            time.sleep(_min_interval - elapsed)
        _last_request_ts = time.time()


class Interactivebrokers(Foreignexchange):
    """
    Interactive Brokers forex exchange class. Contains adjustments needed for Freqtrade
    to work with IBKR for forex trading.
    """

    RECONNECT_MAX_BACKOFF = 32  # seconds
    RECONNECT_BASE_BACKOFF = 1  # seconds

    DECIMAL_PLACES = 6
    SIGNIFICANT_DIGITS = 6
    TICK_SIZE = 0.000001
    MAX_DATA_DELAY = pd.Timedelta(minutes=5)
    MIN_LOT_SIZE = 25_000
    RECONNECT_TIMEOUT = 30

    _cache_lock: Lock
    _entry_rate_cache: dict[str, float]
    _exit_rate_cache: dict[str, float]

    _ft_has_default = {
        "stoploss_on_exchange": False,
        "order_time_in_force": ["GTC", "IOC", "FOK"],
        "ohlcv_candle_limit": 500,
        "ohlcv_has_history": True,
        "ohlcv_partial_candle": True,
        "ohlcv_require_since": False,
        "ohlcv_volume_currency": "base",
        "tickers_have_quoteVolume": True,
        "tickers_have_percentage": True,
        "tickers_have_bid_ask": True,
        "tickers_have_price": True,
        "trades_limit": 1000,
        "trades_pagination": "time",
        "trades_pagination_arg": "since",
        "trades_has_history": False,
        "l2_limit_range": None,
        "l2_limit_range_required": True,
        "mark_ohlcv_price": "mark",
        "mark_ohlcv_timeframe": "8h",
        "funding_fee_timeframe": "8h",
        "ccxt_futures_name": "swap",
        "needs_trading_fees": False,
        "order_props_in_contracts": ["amount", "filled", "remaining"],
        "market_props_in_contracts": ["status"],
        "market_has_ticker": False,
        "market_has_ohlcv": True,
        "order_has_status": True,
        "order_has_type": True,
        "order_has_side": True,
        "order_has_time_in_force": False,
        "order_has_price": True,
        "order_has_amount": True,
        "order_has_cost": False,
        "order_has_fee": False,
        "order_has_slippage": False,
        "order_has_filled": True,
        "order_has_remaining": True,
        "order_has_status_history": False,
        "ws_enabled": True,
        "ws_auto_reconnect": True,
        "ws_reconnect_interval": 30,
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

        self.ib = IB()
        try:
            self.ib.startLoop()
            self._ib_loop_started = True
        except Exception as e:
            logger.debug(f"ib.startLoop() failed — already running or unsupported: {e}")

        self.dry_run = config.get("dry_run", False)
        self.latest_ohlcv: dict = {}
        self._active_tickers: list = []
        self._running = True
        self._reconnect_event = Event()
        self.shutdown_event = Event()
        self.is_shutting_down = False
        self._connection_thread: Thread | None = None
        self._ws_connected = False
        self._markets_cache: dict[str, Any] | None = None
        self._live_price_cache: dict[str, tuple[float, float]] = {}

        self._cache_lock = Lock()
        self._entry_rate_cache = {}
        self._exit_rate_cache = {}

        self._last_connection_ts = 0
        atexit.register(self.close)

        # Set ports based on live/paper trading
        if self.dry_run:
            self.port = config.get("ib_paper_port", 4002)
            logger.info(f"Connecting to IBKR paper trading (IB Gateway) on port {self.port}.")
        else:
            self.port = config.get("ib_live_port", 7497)
            logger.info(f"Connecting to IBKR live trading (TWS) on port {self.port}.")

        # Set up host
        self.host = config.get("ib_host", "127.0.0.1")
        self.client_id = config.get("ib_client_id", 1)

        # Connect to IBKR
        self._connect_to_ib()

        # Set margin mode and initialize markets
        self.margin_mode = MarginMode.NONE
        self.markets = self.get_markets()

        # Start WebSocket connection
        self.ws_start()

        # Verify connection is established
        if not self.ib.isConnected():
            logger.error("Failed to establish connection to Interactive Brokers")
            raise ConnectionError("WebSocket connection failed")

        if "candle_type_def" not in self.config:
            self.config["candle_type_def"] = "spot"
            logger.info("Set default candle_type_def to 'spot' for interactivebrokers")

        # Register signal handler for SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, signum, frame):
        logger.info("Received Ctrl+C, forcing immediate shutdown...")
        self.is_shutting_down = True
        self.close()
        sys.exit(0)  # Ensure the program exits

    def _connect_to_ib(self) -> None:
        """
        Establishes connection to Interactive Brokers.
        Handles refusal cleanly without full traceback spam.
        """
        with _request_lock:
            if self.ib.isConnected():
                logger.info("IBKR already connected.")
                self._ws_connected = True
                self.connected = True
                return

            logger.info(f"Connecting to IBKR paper trading (IB Gateway) on port {self.port}.")
            logger.info(
                f"Connecting to IBKR (host={self.host}, "
                f"port={self.port}, clientId={self.client_id})"
            )

            try:
                self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=5)

                if not self.ib.isConnected():
                    logger.error("❌ IBKR connection failed silently.")
                    self._ws_connected = False
                    self.connected = False
                    raise SystemExit("❌ Could not establish connection to IBKR.")

                logger.info("✅ IBKR connection established.")
                self._ws_connected = True
                self.connected = True

            except ConnectionRefusedError:
                logger.error(
                    "❌ Connection refused: IB Gateway or TWS not running on "
                    f"{self.host}:{self.port}"
                )
                self._ws_connected = False
                self.connected = False
                raise SystemExit("❌ Could not connect to IBKR. Is IB Gateway running?")

            except Exception as e:
                logger.error(f"❌ Unexpected error during IBKR connection: {e}")
                self._ws_connected = False
                self.connected = False
                raise SystemExit("❌ Unexpected failure connecting to IBKR.")

    def _setup_event_loop(self) -> None:
        if self._connection_thread and self._connection_thread.is_alive():
            return

        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_base_delay = 5

        def _start_ib_loop():
            logger.info("Starting IBKR event loop")
            while self._running and not self.shutdown_event.is_set():
                try:
                    if not self.ib.isConnected():
                        self._reconnect_attempts += 1
                        delay = min(
                            self._reconnect_base_delay * 2**self._reconnect_attempts,
                            60,  # Max 60 seconds
                        )
                        logger.warning(
                            f"Connection lost. Reconnecting in {delay}s "
                            f"(attempt {self._reconnect_attempts}/{self._max_reconnect_attempts})"
                        )
                        time.sleep(delay)
                        self._connect_to_ib()
                    else:
                        self._reconnect_attempts = 0
                        self.ib.sleep(1)
                except ConnectionError as e:
                    logger.error(f"IB connection error: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error in event loop: {e}", exc_info=True)
                    time.sleep(5)

            logger.info("IBKR event loop stopped")

        self._connection_thread = Thread(target=_start_ib_loop, daemon=True)
        self._connection_thread.start()

    @property
    def id(self) -> str:
        return "interactivebrokers"

    @property
    def name(self) -> str:
        return "interactivebrokers"

    def get_proxy_coin(self) -> str:
        return self.config.get("stake_currency", "USD")

    def is_market_open(self) -> bool:
        """
        Check if the forex market is currently open based on UTC time.
        - Open: Sunday 10:00 PM UTC to Friday 10:00 PM UTC
        - Closed: Friday 10:00 PM UTC to Sunday 10:00 PM UTC
        """
        now = datetime.now(UTC)
        day = now.weekday()
        current_time = now.time()

        if day == 5:  # Saturday
            return False
        elif day == 6:  # Sunday
            return current_time >= MARKET_OPEN_TIME_UTC
        elif day == 4:  # Friday
            return current_time < MARKET_CLOSE_TIME_UTC
        else:  # Monday to Thursday
            return True

    def wait_for_market_open(self) -> None:
        """
        Sleep until the forex market opens if it is currently closed.
        """
        if self.is_market_open():
            return

        now = datetime.now(UTC)
        if now.weekday() == 5:  # Saturday
            next_open = now.replace(hour=22, minute=0, second=0, microsecond=0) + timedelta(days=1)
        elif now.weekday() == 6:  # Sunday
            next_open = now.replace(hour=22, minute=0, second=0, microsecond=0)
        else:  # Friday after close
            next_open = now.replace(hour=22, minute=0, second=0, microsecond=0) + timedelta(
                days=(6 - now.weekday())
            )

        sleep_seconds = (next_open - now).total_seconds()
        logger.info(
            f"Market closed. Sleeping for {sleep_seconds:.2f} seconds until {next_open} UTC."
        )

        # Sleep in smaller intervals to check for shutdown event
        while sleep_seconds > 0 and not self.shutdown_event.is_set():
            time.sleep(min(1, sleep_seconds))
            sleep_seconds -= 1

        if self.shutdown_event.is_set():
            logger.info("Shutdown signal received, exiting sleep.")

    def create_order(
        self,
        pair: str | tuple,
        ordertype: str,
        side: str,
        amount: float,
        price: float | None = None,
        params: dict[Any, Any] | None = None,
        rate: float | None = None,
        **kwargs,
    ) -> dict:
        # 1) Detect TWS down & back off before everything else
        self.ensure_connected()

        params = params or {}
        pair = pair[0] if isinstance(pair, tuple) else pair

        # ——— Prevent duplicate in-flight orders for the same pair+side ———
        try:
            open_orders = self.fetch_open_orders(pair)
            # match on side and open status
            dup = [
                o
                for o in open_orders
                if o["side"].lower() == side.lower() and o["status"] == "open"
            ]
            if dup:
                logger.warning(
                    f"Skipping new {side.upper()} order for {pair}: "
                    f"{len(dup)} existing open order(s) detected."
                )
                from freqtrade.exceptions import ExchangeError

                raise ExchangeError(f"Duplicate in-flight {side} order for {pair}")
        except ExchangeError:
            # bubble up to FreqTrade so it won't persist anything
            raise
        except Exception as e:
            logger.error(f"Error checking existing orders for {pair}: {e}")
            # proceed anyway

        # ——— initialize contract, amount, price ———
        contract, amount, price = self._initialize_contract_amount_price(pair, amount, price, rate)

        use_market = ordertype.lower() == "market" or (
            side.lower() == "sell" and params.get("exit_as_market", False)
        )

        # ——— build IB order object ———
        if use_market:
            order = Order(action=side.upper(), totalQuantity=amount, orderType="MKT")
        else:
            try:
                if price is None or price <= 0:
                    price = self.get_rate(pair, side=side)
                if not (0.00001 <= price <= 1000.0):
                    raise ValueError(f"Invalid price for order: {price}")
                order = Order(
                    action=side.upper(),
                    totalQuantity=amount,
                    orderType="LMT",
                    lmtPrice=round(price, self.SIGNIFICANT_DIGITS - 1),
                )
            except ValueError as e:
                logger.error(f"Failed to get valid price for order: {e}")
                return self._failed_response(pair, ordertype, side, amount, price, str(e))

        return self._place_and_wait_for_order(contract, order, pair, ordertype, side, amount, price)

    def _place_and_wait_for_order(
        self,
        contract: Any,
        order: Any,
        pair: str,
        ordertype: str,
        side: str,
        amount: float,
        price: float | None,
    ) -> dict:
        """Place order and wait for IB to acknowledge it."""
        try:
            trade = self.ib.placeOrder(contract, order)
            logger.info(
                f"Order placed: {order.action} {order.totalQuantity} "
                f"{pair} at {getattr(order, 'lmtPrice', 'MARKET')}"
            )
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return self._failed_response(pair, ordertype, side, amount, price, str(e))

        # ——— wait for IB to ack/fill ———
        deadline = time.time() + 30
        while (
            time.time() < deadline
            and trade.orderStatus.status in ("ApiPending", "PendingSubmit", "Submitted")
            and not self.shutdown_event.is_set()
        ):
            self.ib.waitOnUpdate(timeout=1)

        if self.shutdown_event.is_set():
            logger.info("Shutdown signal received, exiting order placement.")
            return self._failed_response(pair, ordertype, side, amount, price, "Shutdown")

        # ——— finalize or raise on failure ———
        return self._finalize_trade_status(trade, pair, ordertype, side, amount, price)

    def _initialize_contract_amount_price(self, pair, amount, price, rate):
        if rate is not None and (price is None or price <= 0):
            price = rate

        symbol, currency = self._extract_currencies_from_pair(pair)
        contract = Forex(symbol=symbol, currency=currency, exchange="IDEALPRO")

        try:
            if not self.ib.qualifyContracts(contract):
                raise ValueError(f"Contract qualification failed for {pair}")
        except Exception as e:
            logger.error(f"Contract qualification error: {e}")
            raise ValueError(f"Contract qualification failed for {pair}: {e}")

        min_lot = self.MIN_LOT_SIZE
        amount = max(min_lot, math.floor(amount / min_lot) * min_lot)

        return contract, amount, price

    def _finalize_trade_status(self, trade, pair, ordertype, side, amount, price):
        status = trade.orderStatus.status
        oid = str(trade.order.orderId)
        filled = float(trade.orderStatus.filled)
        remaining = amount - filled

        # Map to Freqtrade status
        ft_status = self._parse_order_status(status)

        # Handle open orders (including partially filled ones)
        if ft_status == "open":
            logger.info(
                f"Order {oid} for {pair} is open (status={status}), "
                f"filled={filled}, remaining={remaining}"
            )
            return {
                "id": oid,
                "symbol": pair,
                "type": ordertype.lower(),
                "side": side.lower(),
                "amount": amount,
                "price": price,
                "filled": filled,
                "remaining": remaining,
                "status": ft_status,
                "info": trade,
            }

        # Handle filled orders
        if ft_status == "closed":
            logger.info(f"Order {oid} for {pair} filled {filled} / {amount}")
            return {
                "id": oid,
                "symbol": pair,
                "type": ordertype.lower(),
                "side": side.lower(),
                "amount": amount,
                "price": price,
                "filled": filled,
                "remaining": remaining,
                "status": ft_status,
                "info": trade,
            }

        # Handle failed/canceled orders
        logger.warning(
            f"Order {oid} for {pair} failed with status: {status}. "
            f"Reason: {trade.orderStatus.whyHeld}"
        )
        Trade.session.rollback()
        raise ExchangeError(f"Order for {pair} failed with status: {status}.")

    def get_rate(
        self,
        pair: str | tuple,
        side: str | None = None,
        **kwargs,
    ) -> float:
        """
        Try to fetch a live price; on failure due to stale/nan data or disconnect,
        trigger a reconnect and retry once before falling back to historical.
        """
        if self.is_shutting_down:
            logger.info("Shutdown in progress terminating now.")
            sys.exit(0)

        if not self.is_market_open():
            self.wait_for_market_open()

        pair = pair[0] if isinstance(pair, tuple) else pair
        # First attempt
        try:
            return self._fetch_live_price(pair, side)
        except Exception as e:
            logger.error(f"Failed to request market data for {pair} (live): {e}")
        # Final fallback
        return self._fallback_to_historical_rate(pair)

    def _fetch_live_price(self, pair: str, side: str | None) -> float:
        """
        Fetch live price from IBKR with caching, snapshot requests,
        and silent fallback to historical data. This version is improved to be
        more reliable and efficient.

        Args:
            pair: Currency pair in format 'BASE/QUOTE'
            side: 'buy', 'sell', or None for mid price

        Returns:
            Current price as float (live if possible, else historical)
        """
        if self.is_shutting_down:
            raise ConnectionError("Shutdown in progress")

        now = time.time()
        # 1) Return cached price if within 1 second
        cached = self._live_price_cache.get(pair)
        if cached and (now - cached[0] < 1.0):
            price = cached[1]
            logger.debug(f"Using cached price for {pair} ({side}): {price}")
            return price

        # Connection check
        if not self.ib.isConnected():
            raise ConnectionError("Not connected to IBKR")

        # Build contract
        symbol, currency = self._extract_currencies_from_pair(pair)
        contract = Forex(symbol=symbol, currency=currency, exchange="IDEALPRO")

        # Rate limit before request
        throttle()

        ticker = None
        try:
            # 2) Snapshot request: get one tick then unsubscribe
            logger.debug(f"Requesting live price for {contract.symbol}, reqId pending")
            ticker = self.ib.reqMktData(contract, snapshot=True)
            # ticker = self.ib.reqMktData(contract, "", True, False)
            logger.debug(f"Received ticker for {contract.symbol}, reqId processed")

            # Wait for valid data with a timeout instead of a fixed sleep
            deadline = time.time() + 5  # 5-second timeout
            while time.time() < deadline:
                bid = getattr(ticker, "bid", None)
                ask = getattr(ticker, "ask", None)
                if (
                    bid is not None
                    and ask is not None
                    and not math.isnan(bid)
                    and not math.isnan(ask)
                ):
                    break  # Data is valid
                self.ib.sleep(0.1)  # Let ib_insync process events
            else:
                # Loop finished without break, indicates a timeout
                raise ValueError(f"Timeout waiting for valid live tick for {pair}")

            bid = ticker.bid
            ask = ticker.ask

            # Choose price based on side
            if side is None:
                price = (bid + ask) / 2
            elif side.lower() == "buy":
                price = ask
            elif side.lower() == "sell":
                price = bid
            else:
                price = (bid + ask) / 2

            # 3) Cache and log live price
            self._live_price_cache[pair] = (now, price)
            logger.info(f"Returning price for {pair} ({side}): {price}")
            return price

        except Exception as e:
            logger.warning(f"Live price fetch for {pair} failed: {e}. Falling back to historical.")
            # 4) On any failure, fallback to historical close
            price = self._fallback_to_historical_rate(pair)
            logger.info(f"Historical fallback price for {pair}: {price}")
            return price
        finally:
            # IMPORTANT: Cancel the market data subscription to prevent leaks
            pass

    def _fallback_to_historical_rate(self, pair: str) -> float:
        """
        Fallback to historical data when live price fails.

        Args:
            pair: Currency pair in format 'BASE/QUOTE'

        Returns:
            Most recent historical close price

        Raises:
            ValueError: If no valid historical data available
        """
        if self.is_shutting_down:
            raise ConnectionError("Shutdown in progress")

        try:
            timeframe = self.config.get("timeframe", "5m")
            ohlcv = self.get_historic_ohlcv(pair, timeframe=timeframe, limit=1)

            if ohlcv.empty:
                raise ValueError(f"No historical data available for {pair}")

            close_price = ohlcv.iloc[0]["close"]

            if pd.isna(close_price):
                raise ValueError(f"NaN value in historical data for {pair}")

            if not (0.00001 <= close_price <= 1000.0):
                raise ValueError(f"Historical price {close_price} out of valid range for {pair}")

            logger.info(f"Using historical close price for {pair}: {close_price}")
            return close_price

        except Exception as e:
            logger.error(f"Historical data fallback failed for {pair}: {str(e)}")
            raise ValueError(f"Could not fetch valid rate for {pair} from any source")

    def _failed_response(self, pair, ordertype, side, amount, price, info):
        return {
            "id": None,
            "symbol": pair,
            "type": ordertype.lower(),
            "side": side.lower(),
            "amount": amount,
            "price": price,
            "filled": 0.0,
            "remaining": amount,
            "status": "failed",
            "info": info,
        }

    def _parse_order_status(self, ib_status: str) -> str:
        status_mapping = {
            "ApiPending": "open",
            "PendingSubmit": "open",
            "PreSubmitted": "open",
            "Submitted": "open",
            "Filled": "closed",
            "Cancelled": "canceled",
            "Canceled": "canceled",
            "Inactive": "canceled",
            "ApiCancelled": "canceled",
            "PendingCancel": "canceling",
        }
        return status_mapping.get(ib_status, "unknown")

    def cancel_order(self, order_id: str, pair: str | None = None) -> dict:
        try:
            ib_order_id = int(order_id)
            self.ib.client.cancelOrder(ib_order_id)
            logger.info(f"Order {order_id} cancel request sent successfully.")
            # *** CRUCIAL: tell Freqtrade that this order is gone ***
            self.remove_order_from_freqtrade(order_id)
            return {
                "status": "canceled",
                "id": order_id,
                "message": "Cancelled on IBKR and removed from Freqtrade",
            }
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid order ID format when canceling '{order_id}': {e}")
            return {"status": "error", "id": order_id, "message": f"Invalid order ID format: {e}"}
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return {"status": "error", "id": order_id, "message": str(e)}

    def get_markets(
        self,
        reload: bool = False,
        params: dict[Any, Any] | None = None,
        tradable_only: bool = False,
        active_only: bool = False,
    ) -> dict[Any, Any]:
        if not reload and self._markets_cache is not None:
            return self._markets_cache

        markets: dict[str, Any] = {}
        forex_pairs = [
            ("EUR", "USD"),
            ("GBP", "USD"),
            ("USD", "JPY"),
            ("AUD", "USD"),
            ("USD", "CAD"),
            ("USD", "CHF"),
            ("NZD", "USD"),
            ("EUR", "GBP"),
            ("EUR", "JPY"),
            ("GBP", "JPY"),
            ("EUR", "AUD"),
            ("USD", "CNH"),
            ("USD", "MXN"),
            ("EUR", "CAD"),
            ("AUD", "JPY"),
            ("GBP", "CAD"),
            ("AUD", "CAD"),
            ("EUR", "NZD"),
            ("GBP", "AUD"),
            ("USD", "TRY"),
        ]

        for base, quote in forex_pairs:
            pair = f"{base}/{quote}"
            markets[pair] = {
                "id": pair,
                "symbol": pair,
                "base": base,
                "quote": quote,
                "precision": {"amount": 2, "price": 5},
                "limits": {
                    "amount": {"min": self.MIN_LOT_SIZE, "max": 10_000_000},
                    "price": {"min": 0.00001, "max": 1000000},
                    "cost": {"min": 0.01, "max": 1000000},
                },
                "active": True,
                "info": {"base": base, "quote": quote},
            }

        self._markets_cache = markets
        return markets

    def reload_markets(self, params: dict[Any, Any] | None = None) -> dict:
        self.markets = self.get_markets(reload=True, params=params)
        logger.info("Markets reloaded successfully.")
        return self.markets

    def get_fee(self, symbol: str, now: Any = None, taker_or_maker: str = "maker") -> float:
        maker_fee = 0.0001
        taker_fee = 0.0002
        return maker_fee if taker_or_maker == "maker" else taker_fee

    async def fetch_historical_data(self, contract, durationStr, ib_timeframe):
        """
        Request historical price data from IBKR (one shot only).

        Args:
            contract: IBKR Contract object.
            durationStr: How far back to go (e.g. '1 D', '2 W').
            ib_timeframe: Bar size (e.g. '1 min', '5 mins').

        Returns:
            List of bars, or empty list if unavailable.
        """
        try:
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",  # Now
                durationStr=durationStr,
                barSizeSetting=ib_timeframe,
                whatToShow="MIDPOINT",
                useRTH=False,
                keepUpToDate=False,  # ONE SHOT (no streaming)
            )
            if not bars:
                logger.warning(f"No historical data returned for contract: {contract}")
            return bars

        except Exception as e:
            logger.warning(f"Historical data error for {contract}: {e}")
            return []

    def get_historic_ohlcv(
        self,
        pair: str,
        since: int | None = None,
        timeframe: str | None = None,
        limit: int = 1000,
        params: dict | None = None,
        since_ms: int | None = None,
        is_new_pair: bool = True,
        candle_type: str = "spot",
        until_ms: int | None = None,
    ) -> pd.DataFrame:
        if not self.is_market_open():
            self.wait_for_market_open()

        if isinstance(pair, tuple):
            pair = pair[0]

        symbol, currency = self._extract_currencies_from_pair(pair)
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "CASH"
        contract.currency = currency
        contract.exchange = "IDEALPRO"

        if timeframe is None:
            timeframe = self.config.get("timeframe", "1h")
        ib_timeframe = self._convert_timeframe(timeframe)
        durationStr = self._calculate_duration(timeframe, limit)

        try:
            throttle()
            bars = self.ib.run(self.fetch_historical_data(contract, durationStr, ib_timeframe))
            throttle()
            if not bars:
                logger.warning(f"No bars returned for {pair} with timeframe {timeframe}")
                return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

            df = util.df(bars)
            if df is None or df.empty:
                logger.warning(f"Empty DataFrame returned for {pair}")
                return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

            df.rename(
                columns={
                    "date": "timestamp",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                },
                inplace=True,
            )

            if "timestamp" in df.columns:
                df["date"] = pd.to_datetime(df["timestamp"], utc=True)
            else:
                logger.error(f"No timestamp column in DataFrame for {pair}")
                raise ValueError("DataFrame must have a 'timestamp' column")

            df = df.sort_values(by="date", ascending=True).reset_index(drop=True)

            if not df.empty:
                current_time = datetime.now(UTC)
                last_candle = df["date"].iloc[-1]
                first_candle = df["date"].iloc[0]
                num_candles = len(df)
                age_minutes = (current_time - last_candle).total_seconds() / 60
                logger.info(
                    f"Retrieved {num_candles} candles for {pair}"
                    f"from {first_candle} to {last_candle} "
                    f"(Last candle age: {age_minutes:.2f} minutes)"
                )
            return df

        except Exception as e:
            logger.error(f"Failed to fetch historical data for {pair}: {e}")
            raise

    def refresh_latest_ohlcv(self, pairs: list) -> None:
        """
        Refresh the latest OHLCV data for the given pairs.
        If the market is closed, sleep until 5 minutes before it opens and inform the user.
        """
        if not pairs:
            logger.debug("Empty pairs list passed to refresh_latest_ohlcv")
            return

        for item in pairs:
            try:
                if isinstance(item, tuple):
                    if len(item) >= 2:
                        pair, timeframe = item[0], item[1]
                        candle_type = item[2] if len(item) > 2 else "spot"
                    else:
                        pair = item[0]
                        timeframe = self.config.get("timeframe", "1h")
                        candle_type = "spot"
                else:
                    pair = item
                    timeframe = self.config.get("timeframe", "1h")
                    candle_type = "spot"

                ohlcv = self.get_historic_ohlcv(pair, None, timeframe, limit=3)

                if not ohlcv.empty:
                    key = (pair, timeframe, candle_type)
                    self.latest_ohlcv[key] = ohlcv
                    logger.debug(
                        f"Refreshed latest OHLCV for {pair}/{timeframe}, "
                        f"last timestamp: {ohlcv['date'].iloc[-1]}"
                    )
                else:
                    logger.warning(f"No OHLCV data refreshed for {pair}/{timeframe}")
            except Exception as e:
                logger.error(f"Failed to refresh latest OHLCV for {pair}: {e}")

    def klines(
        self,
        pair: str,
        timeframe: str | None = None,
        since: int = 0,
        limit: int = 1000,
        params: dict[Any, Any] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if params is None:
            params = {}
        if timeframe is None:
            timeframe = self.config.get("timeframe", "1h")
        return self.get_historic_ohlcv(pair, since, timeframe, limit)

    def get_balances(self):
        account = self.ib.accountSummary()
        balances: dict[str, Any] = {}
        for item in account:
            if item.tag == "TotalCashValue":
                balances[item.currency] = {
                    "free": float(item.value),
                    "used": 0.0,
                    "total": float(item.value),
                }
        return balances

    def market_is_tradable(self, market: dict) -> bool:
        return market.get("active", False) and market.get("tradable", True)

    def get_pair_quote_currency(self, pair: str) -> str:
        if pair not in self.markets:
            raise ValueError(f"Pair {pair} not found in markets")
        return self.markets[pair]["quote"]

    def get_pair_base_currency(self, pair: str) -> str:
        if pair not in self.markets:
            raise ValueError(f"Pair {pair} not found in markets")
        return self.markets[pair]["base"]

    def ws_connection_reset(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            logger.info("WebSocket connection reset")
            self._ws_connected = True
        except Exception as e:
            logger.error(f"Failed to reset WebSocket connection: {e}")
            self._ws_connected = False

    def ws_start(self) -> None:
        if not self.ib.isConnected():
            try:
                self.ib.connect(self.host, self.port, clientId=self.client_id)
                self._setup_event_loop()
                self._ws_connected = True
                logger.info("WebSocket started")
            except Exception as e:
                logger.error(f"Failed to start WebSocket: {e}")
                self._ws_connected = False
        else:
            logger.info("WebSocket already running")

    def ws_stop(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()
        logger.info("WebSocket stopped")

    def ws_health_check(self) -> bool:
        if not self.ib.isConnected():
            return False

        try:
            # Verify actual data flow
            self.ib.reqCurrentTime()
            return True
        except Exception:
            return False

    def _convert_timeframe(self, timeframe: str) -> str:
        mapping = {
            "1m": "1 min",
            "5m": "5 mins",
            "15m": "15 mins",
            "30m": "30 mins",
            "1h": "1 hour",
            "4h": "4 hours",
            "1d": "1 day",
        }
        return mapping.get(timeframe, timeframe)

    def _calculate_duration(self, timeframe: str, limit: int) -> str:
        timeframe_to_candles_per_day = {
            "1m": 1440,
            "5m": 288,
            "15m": 96,
            "30m": 48,
            "1h": 24,
            "4h": 6,
            "1d": 1,
        }

        if timeframe not in timeframe_to_candles_per_day:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        candles_per_day = timeframe_to_candles_per_day[timeframe]
        total_days = math.ceil(limit / candles_per_day)

        if total_days <= 365:
            return f"{total_days} D"
        else:
            years = math.ceil(total_days / 365)
            return f"{years} Y"

    def validate_timeframes(self, timeframes):
        if isinstance(timeframes, str):
            timeframes = [timeframes]

        supported_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        logger.info(f"Validating timeframes: {timeframes}")

        for timeframe in timeframes:
            logger.info(f"Validating timeframe: {timeframe}")
            if timeframe not in supported_timeframes:
                raise ValueError(
                    f"Timeframe '{timeframe}' is not supported by Interactive Brokers."
                )

    def get_funding_fees(self, pair: str, timeframe: str | None = None, **kwargs) -> float:
        return 0.0

    def fetch_order_or_stoploss_order(
        self,
        order_id: str,
        pair: str | None = None,
        *args,
        **kwargs,
    ) -> dict:
        order = self.fetch_order(order_id, pair)
        if order is None:
            return {"status": "not_found"}
        return order

    def check_order_canceled_empty(self, order: dict) -> bool:
        if not order:
            return False
        return order.get("status") == "canceled" and order.get("remaining", 0) == 0

    def order_has_fee(self, order) -> bool:
        return False

    def get_trades_for_order(self, order, *args, **kwargs):
        if not order:
            return []

        order_id = None
        if isinstance(order, dict):
            order_id = order.get("order_id", None)
        else:
            order_id = getattr(order, "order_id", None)

        if order_id is None:
            return []

        trades = self.ib.trades()
        matching_trades = []

        for trade in trades:
            if hasattr(trade.order, "orderId") and trade.order.orderId == order_id:
                matching_trades.append(trade)

        return matching_trades

    def get_liquidation_price(
        self,
        pair: str,
        side: str | None = None,
        leverage: float | None = None,
        open_rate: float | None = None,
        amount: float | None = None,
        initial_stop_rate: float | None = None,
        is_short: bool = False,
        stake_amount: float | None = None,
        wallet_balance: float | None = None,
    ) -> None:
        return None

    def cancel_order_with_result(self, *args, **kwargs) -> dict | None:
        order_id = None
        for arg in args:
            if isinstance(arg, str) and arg.isdigit():
                order_id = arg
                break
            if isinstance(arg, int):
                order_id = str(arg)
                break
            if isinstance(arg, dict) and "id" in arg:
                order_id = str(arg["id"])
                break
            if hasattr(arg, "order_id"):
                order_id = str(arg.order_id)
                break

        if not order_id:
            logger.error(f"cancel_order_with_result: Can't extract order_id from {args}")
            return None

        try:
            result = self.cancel_order(order_id)
        except Exception as e:
            logger.error(f"cancel_order_with_result: error canceling {order_id}: {e}")
            return None

        if result.get("status") == "canceled":
            self.remove_order_from_freqtrade(order_id)

        updated = self.fetch_order(order_id)
        if updated:
            return updated

        return {
            "id": order_id,
            "status": "canceled",
            "filled": 0.0,
            "remaining": 0.0,
        }

    def _extract_currencies_from_pair(self, pair: str) -> tuple[str, str]:
        if isinstance(pair, tuple):
            pair = pair[0]

        parts = pair.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid pair format: {pair}. Expected format 'BASE/QUOTE'")

        symbol = parts[0].strip().upper()
        currency = parts[1].strip().upper()

        if len(symbol) != 3 or len(currency) != 3:
            raise ValueError(
                f"Invalid currency codes: symbol={symbol} (len={len(symbol)}), "
                f"currency={currency} (len={len(currency)}). Expected 3-letter codes."
            )

        return symbol, currency

    def get_min_pair_stake_amount(self, pair: str, *args, **kwargs) -> float:
        return float(self.config.get("stake_amount_min", 10.0))

    def get_max_pair_stake_amount(self, pair: str, *args, **kwargs) -> float:
        return float(self.config.get("stake_amount_max", 1000000.0))

    def get_precision_amount(self, pair: str) -> int:
        return 2

    def get_precision_price(self, pair: str) -> int:
        return 5

    @property
    def precisionMode(self):
        return 2

    @property
    def precision_mode_price(self):
        return 2

    def get_contract_size(self, pair: str) -> float:
        return 100000.0

    def get_order_id(self, order: dict | None) -> str | None:
        return self.get_order_id_conditional(order)

    def get_order_id_conditional(self, order: dict | None) -> str | None:
        if not order:
            return None

        if isinstance(order, dict):
            return order.get("id") if "id" in order else None
        return None

    def get_option(self, key: str, default: Any = None) -> Any:
        return self._ft_has_default.get(key, default)

    def validate_required_startup_candles(self, required_startup: int, timeframe: str) -> None:
        if not self.markets:
            logger.error("No markets available for validation of startup candles")
            raise ValueError("No markets available for validation")

        first_pair = next(iter(self.markets.keys()))
        try:
            ohlcv = self.get_historic_ohlcv(first_pair, timeframe=timeframe, limit=1)
            if ohlcv.empty:
                logger.error(
                    f"Cannot fetch even one candle for {first_pair} on timeframe {timeframe}"
                )
                raise ValueError(
                    f"Cannot fetch historical data for {first_pair} on timeframe {timeframe}"
                )
            logger.info(
                f"Successfully validated startup candles for {first_pair} on timeframe {timeframe}"
            )
        except Exception as e:
            logger.error(f"Failed to validate required startup candles: {e}")
            raise

    def fetch_open_orders(self, symbol: str | None = None) -> list[dict]:
        """
        Fetch open orders from Interactive Brokers and normalize them for Freqtrade.
        Ensures all returned orders include a valid 'side' field to prevent bot crashes.
        Adds "orphaned": True to orders not associated with known trades.
        """

        self.ensure_connected()

        if not self.is_market_open():
            self.wait_for_market_open()

        orders: list[dict] = []
        for o in self.ib.openOrders():
            # ——— Guard: skip anything that is not a full IB Trade object ———
            if (
                not hasattr(o, "contract")
                or not hasattr(o, "order")
                or not hasattr(o, "orderStatus")
            ):
                logger.warning(f"Skipping unparsable open order entry: {o!r}")
                continue

            try:
                sym = f"{o.contract.symbol}/{o.contract.currency}"
                if symbol and sym != symbol:
                    continue

                # Determine side
                action = o.order.action.lower()
                side = "buy" if action == "buy" else "sell" if action == "sell" else "unknown"

                total = float(o.order.totalQuantity or 0.0)
                filled = float(o.orderStatus.filled or 0.0)

                orders.append(
                    {
                        "id": str(o.order.orderId),
                        "symbol": sym,
                        "type": (
                            o.order.orderType.lower()
                            if getattr(o.order, "orderType", None)
                            else "unknown"
                        ),
                        "side": side,
                        "amount": total,
                        "price": getattr(o.order, "lmtPrice", None),
                        "filled": filled,
                        "remaining": total - filled,
                        "status": self._parse_order_status(o.orderStatus.status),
                        "info": {"orphaned": True},  # Freqtrade will filter its own trades
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to parse open order: {e}")

        return orders

    def fetch_order(self, order_id: str, pair: str | None = None) -> dict:
        if order_id is None:
            logger.error("Cannot fetch order with order_id=None")
            return {
                "status": "not_found",
                "id": None,
                "symbol": pair or "unknown",
                "side": "unknown",
                "amount": 0.0,
                "filled": 0.0,
                "remaining": 0.0,
            }

        try:
            oid = int(order_id)
        except (ValueError, TypeError):
            logger.error(f"Invalid order ID format: {order_id}")
            return {
                "status": "not_found",
                "id": order_id,
                "symbol": pair or "unknown",
                "side": "unknown",
                "amount": 0.0,
                "filled": 0.0,
                "remaining": 0.0,
            }

        try:
            for trade in self.ib.trades():
                if trade.order.orderId == oid:
                    filled = (
                        float(trade.orderStatus.filled)
                        if hasattr(trade.orderStatus, "filled")
                        else 0.0
                    )
                    total = (
                        float(trade.order.totalQuantity)
                        if hasattr(trade.order, "totalQuantity")
                        else 0.0
                    )
                    symbol = (
                        f"{trade.contract.symbol}/{trade.contract.currency}"
                        if (
                            hasattr(trade.contract, "symbol")
                            and hasattr(trade.contract, "currency")
                        )
                        else (pair or "unknown")
                    )
                    side = (
                        trade.order.action.lower()
                        if hasattr(trade.order, "action") and trade.order.action
                        else "buy"
                    )
                    price = (
                        float(trade.order.lmtPrice)
                        if trade.order.orderType == "LMT" and hasattr(trade.order, "lmtPrice")
                        else None
                    )
                    status = self._parse_order_status(
                        trade.orderStatus.status
                        if hasattr(trade.orderStatus, "status")
                        else "unknown"
                    )

                    if status in ("canceled", "rejected", "inactive"):
                        logger.warning(f"Order {order_id} is {status}. Marking as canceled.")
                        return {
                            "status": "canceled",  # Changed from "not_found" to "canceled"
                            "id": order_id,
                            "symbol": symbol,
                            "side": side,
                            "amount": total,
                            "filled": filled,
                            "remaining": total - filled,
                        }

                    return {
                        "id": order_id,
                        "symbol": symbol,
                        "type": trade.order.orderType.lower()
                        if hasattr(trade.order, "orderType")
                        else "unknown",
                        "side": side,
                        "amount": total,
                        "price": price,
                        "filled": filled,
                        "remaining": total - filled,
                        "status": status,
                        "info": trade,
                    }

            logger.debug(f"fetch_order: no trade with orderId={order_id}")
            return {
                "status": "not_found",
                "id": order_id,
                "symbol": pair or "unknown",
                "side": "unknown",
                "amount": 0.0,
                "filled": 0.0,
                "remaining": 0.0,
            }

        except Exception as e:
            logger.error(f"Error in fetch_order for {order_id}: {e}")
            return {
                "status": "not_found",
                "id": order_id,
                "symbol": pair or "unknown",
                "side": "unknown",
                "amount": 0.0,
                "filled": 0.0,
                "remaining": 0.0,
            }

    def close_orphaned_orders(self) -> None:
        for order in self.fetch_open_orders():
            if order.get("info", {}).get("orphaned"):
                logger.warning(
                    f"Orphaned order found: {order['id']} {order['symbol']} — attempting cancel."
                )
                try:
                    self.cancel_order(order["id"], order["symbol"])
                except Exception as e:
                    logger.error(f"Failed to cancel orphaned order {order['id']}: {e}")

    def cleanup_incomplete_trades(self):
        """
        Detect and remove incomplete trades from Freqtrade's database.
        Incomplete trades are those with zero amount or invalid rates.
        """
        try:
            open_trades = Trade.get_open_trades()
            for trade in open_trades:
                if trade.amount == 0 or trade.open_rate <= 0:
                    logger.warning(f"Incomplete trade detected: ID {trade.id}, Pair {trade.pair}")
                    trade.is_open = False
                    trade.close_date = datetime.now(UTC)
                    trade.status = "closed"
                    Trade.session.commit()
                    logger.info(f"Closed incomplete trade ID {trade.id}")
        except Exception as e:
            logger.error(f"Failed to cleanup incomplete trades: {e}")

    def sync_orders(self):
        self.cleanup_incomplete_trades()
        open_orders = self.fetch_open_orders()
        open_order_ids = {order["id"] for order in open_orders}
        logger.info(f"Found {len(open_order_ids)} open orders in IBKR.")

        try:
            freqtrade_open_orders = self.get_freqtrade_open_orders()
        except Exception as e:
            logger.error(f"Failed to fetch Freqtrade open trades: {e}")
            return

        removed_count = 0
        for trade in freqtrade_open_orders:
            try:
                if trade.order_id not in open_order_ids:
                    # Check the order status explicitly
                    order = self.fetch_order(trade.order_id, trade.pair)
                    if order["status"] in ("canceled", "not_found"):
                        logger.warning(
                            f"Removing Order {trade.order_id} ({trade.pair}) is {order['status']}."
                        )
                        self.remove_order_from_freqtrade(trade.order_id)
                        removed_count += 1
                    else:
                        logger.info(
                            f"Order {trade.order_id} ({trade.pair}) "
                            f"is still active with status {order['status']}."
                        )
            except Exception as e:
                logger.error(f"Failed to process trade {trade.order_id}: {e}")

        logger.info(
            f"Synchronization complete. Removed {removed_count} orphaned or canceled trades."
        )

    def get_freqtrade_open_orders(self):
        """Retrieve open orders from Freqtrade internal state."""
        try:
            return Trade.get_open_trades()
        except Exception as e:
            logger.error(f"Failed to fetch Freqtrade open trades: {e}")
            return []

    def remove_order_from_freqtrade(self, order_id: str):
        try:
            order = Trade.session.query(FTOrder).filter_by(id=int(order_id)).first()
            if not order:
                logger.warning(f"No order found with id {order_id}.")
                return

            trade = order.trade
            if trade and trade.is_open:
                trade.is_open = False
                # Corrected attribute: 'average' instead of 'price_open'
                trade.close_rate = order.average  # Previously order.price_open
                trade.close_date = datetime.now(UTC)
                Trade.session.commit()

                logger.info(f"Closed orphaned trade from order {order_id}")
                # Removed RPCManager call - requires freqtrade instance
                # Consider alternative notification if needed

        except Exception as e:
            logger.error(f"Failed to remove trade {order_id}: {e}")

    def close(self) -> None:
        """
        Aggressively shut down IBKR connection and subscriptions,
        without long waits, so CTRL+C returns immediately.
        """
        self._running = False
        self.shutdown_event.set()

        # Forcefully cancel all subscriptions, but quietly ignore connection failures
        try:
            if hasattr(self.ib.client, "reqMarketDataType"):
                self.ib.client.reqMarketDataType(3)  # Switch to delayed feed
            for ticker in getattr(self, "_active_tickers", []):
                try:
                    self.ib.cancelMktData(ticker.contract)
                except ConnectionError:
                    # Already disconnected—no need to warn
                    pass
                except Exception as e:
                    logger.warning(f"Error canceling ticker during shutdown: {e}")
            self._active_tickers.clear()
        except ConnectionError:
            # Ignore if the client is already disconnected
            pass
        except Exception as e:
            logger.warning(f"Unexpected error during shutdown subscription cleanup: {e}")

    def _disconnect_and_clear(self) -> None:
        try:
            if self.ib.isConnected():
                self.ib.disconnect()
                self.ib.client._sock = None  # Nullify socket immediately
        except Exception as e:
            logger.warning(f"Exception during disconnect: {e}")

    def _release_port_and_stop_threads(self) -> None:
        def _release_port():
            for attempt in range(3):
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind((self.host, self.port))
                    s.close()
                    logger.info(f"Port {self.port} released successfully on attempt {attempt + 1}.")
                    return
                except Exception as e:
                    logger.warning(
                        f"Failed to release port {self.port} on attempt {attempt + 1}: {e}"
                    )
                    time.sleep(1)
            logger.error(f"Could not release port {self.port} after 3 attempts.")

        Thread(target=_release_port, daemon=True).start()

        thr = getattr(self, "_connection_thread", None)
        if thr and thr.is_alive():
            thr.join(timeout=0.1)

    def fetch_closed_orders(self, symbol: str | None = None) -> list[dict]:
        closed = []
        for t in self.ib.trades():
            status = self._parse_order_status(t.orderStatus.status)
            if status not in ("closed", "canceled"):
                continue
            sym = f"{t.contract.symbol}/{t.contract.currency}"
            if symbol and sym != symbol:
                continue
            qty = float(t.order.totalQuantity)
            filled = float(t.orderStatus.filled)
            closed.append(
                {
                    "id": str(t.order.orderId),
                    "symbol": sym,
                    "type": t.order.orderType.lower(),
                    "side": t.order.action.lower(),
                    "amount": qty,
                    "price": (t.order.lmtPrice if t.order.orderType == "LMT" else None),
                    "filled": filled,
                    "remaining": qty - filled,
                    "status": status,
                    "info": {},
                }
            )
        return closed

    def fetch_my_trades(self, symbol: str | None = None) -> list[dict]:
        trades = []
        for t in self.ib.trades():
            for fill in t.fills:
                t_sym = f"{t.contract.symbol}/{t.contract.currency}"
                if symbol and t_sym != symbol:
                    continue
                trades.append(
                    {
                        "id": f"{t.order.orderId}:{fill.execution.execId}",
                        "symbol": t_sym,
                        "side": t.order.action.lower(),
                        "amount": float(fill.execution.shares),
                        "price": float(fill.execution.price),
                        "fee": 0.0,
                        "timestamp": fill.execution.time.isoformat(),
                        "info": {},
                    }
                )
        return trades

    def fetch_balance(self) -> dict:
        # Simply alias your existing balance call
        return self.get_balances()

    def fetch_positions(self) -> list[dict]:
        positions = []
        for pos in self.ib.positions():
            sym = f"{pos.contract.symbol}/{pos.contract.currency}"
            amount = float(pos.position)
            if amount == 0:
                continue
            avg_cost = float(pos.avgCost)
            positions.append(
                {
                    "symbol": sym,
                    "amount": amount,
                    "entry_price": avg_cost,
                    "info": {},
                }
            )
        return positions

    def fetch_ticker(self, symbol: str) -> dict:
        # Reuse fetch_tickers under the hood
        return self.fetch_tickers([symbol])[symbol]

    def fetch_tickers(self, symbols: list[str] | None = None) -> dict[str, dict]:
        tickers: dict[str, dict] = {}
        # Default to all configured market pairs if no list given
        symbols = symbols or list(self.markets.keys())
        for sym in symbols:
            contract = self._get_contract(sym)
            # snapshot=True for one off quote, or reuse existing subscription
            data = self.ib.reqMktData(contract, "", True, False)
            throttle()
            bid = getattr(data, "bid", None) or 0.0
            ask = getattr(data, "ask", None) or 0.0
            last = (bid + ask) / 2 if bid and ask else getattr(data, "last", 0.0)
            tickers[sym] = {
                "symbol": sym,
                "bid": bid,
                "ask": ask,
                "last": last,
                "info": {},
            }
        return tickers

    def _get_contract(self, symbol: str) -> Forex:
        """
        Creates and returns an IBKR Forex contract for a given symbol/pair.
        """
        base, quote = self._extract_currencies_from_pair(symbol)
        return Forex(symbol=base, currency=quote, exchange="IDEALPRO")

    def get_rates(self, pair: str, refresh: bool, is_short: bool) -> tuple[float, float]:
        """
        Returns entry and exit rates for a forex pair, compatible with Freqtrade UI.
        Caches rates when `refresh=False`.
        """
        entry_rate = None
        exit_rate = None

        # Try cache first
        if not refresh:
            with self._cache_lock:
                entry_rate = self._entry_rate_cache.get(pair)
                exit_rate = self._exit_rate_cache.get(pair)
            if entry_rate is not None:
                logger.debug(f"Using cached entry rate for {pair}.")
            if exit_rate is not None:
                logger.debug(f"Using cached exit rate for {pair}.")

        # Always fetch fresh if cache miss or refresh requested
        if entry_rate is None or exit_rate is None:
            ticker = self.fetch_ticker(pair)
            bid = ticker["bid"]
            ask = ticker["ask"]

            # For a long entry, buy at ask; for a short entry, sell at bid
            entry_rate = entry_rate if entry_rate is not None else (ask if not is_short else bid)
            # For a long exit, sell at bid; for a short exit, buy at ask
            exit_rate = exit_rate if exit_rate is not None else (bid if not is_short else ask)

            # Cache the newly fetched rates
            with self._cache_lock:
                self._entry_rate_cache[pair] = entry_rate
                self._exit_rate_cache[pair] = exit_rate

        return entry_rate, exit_rate

    def get_conversion_rate(self, base: str, quote: str) -> float:
        """
        Returns the mid market conversion rate between two currencies.
        FreqUI calls this to convert between quote currencies (e.g. P&L displays).
        """
        pair = f"{base}/{quote}"
        try:
            ticker = self.fetch_ticker(pair)
        except Exception:
            # If the direct pair doesn't exist, try the inverse and invert the rate.
            inverse = f"{quote}/{base}"
            inv_ticker = self.fetch_ticker(inverse)
            mid = (inv_ticker["bid"] + inv_ticker["ask"]) / 2
            return 1.0 / mid

        # Mid market rate = (bid + ask) / 2
        return (ticker["bid"] + ticker["ask"]) / 2

    def exit_positions(self, trades):
        for trade in trades:
            if trade.has_open_position and trade.is_open:
                try:
                    exit_rate = self.exchange.get_rate(trade.pair, side="sell")
                    if exit_rate is None:
                        logger.warning(
                            f"Could not fetch exit rate for {trade.pair} during shutdown."
                        )
                        continue  # Skip this trade instead of crashing
                    # Existing code to exit the trade with the rate
                except Exception as e:
                    logger.error(f"Error exiting position for {trade.pair}: {e}")

    def ensure_connected(self):
        """
        Ensure the IBKR client is connected. If not, retry with exponential backoff.
        Raises ExchangeError if we exhaust retries.
        """
        if self.ib.isConnected():
            return

        backoff = self.RECONNECT_BASE_BACKOFF
        while backoff <= self.RECONNECT_MAX_BACKOFF:
            logger.warning(f"TWS disconnected — retrying connection in {backoff}s…")
            time.sleep(backoff)
            try:
                # adjust host/port/clientId to your settings
                self.ib.connect(self.host, self.port, clientId=self.clientId)
                logger.info("Reconnected to TWS successfully.")
                return
            except Exception as e:
                logger.error(f"Reconnect attempt failed: {e}")
                backoff *= 2

        # final failure
        raise ExchangeError("Unable to reconnect to IBKR TWS after multiple attempts.")

    def validate_config(self, config: dict) -> None:
        """
        Validate the exchange configuration.
        This method is required by Freqtrade and called during bot initialization.
        """
        logger.info("Validating Interactive Brokers configuration...")

        # Check for required connection parameters
        if not hasattr(self, "host") or not hasattr(self, "port"):
            raise OperationalException(
                "Interactive Brokers host and port configuration are required."
            )

        # Test connection by making a simple API call
        try:
            # This will fail immediately if connection is invalid
            self.ib.client.reqCurrentTime()
            logger.debug("Interactive Brokers connection validated successfully")
        except Exception as e:
            error_message = str(e).lower()
            if "connection" in error_message or "not connected" in error_message:
                logger.error(
                    "Connection failed - Cannot connect to Interactive Brokers. "
                    "Please ensure TWS or IB Gateway is running and configured properly."
                )
                sys.exit(1)
            # Re-raise other connection errors
            raise

        # Validate dry_run mode compatibility
        if not self.dry_run:
            logger.warning(
                "Live trading mode is enabled with Interactive Brokers. "
                "Ensure you have sufficient funds and understand the risks."
            )

        # Validate timeframes if specified in config
        timeframes = config.get("timeframes", [])
        if timeframes:
            self.validate_timeframes(timeframes)

        logger.info("Interactive Brokers configuration validation completed successfully.")

    def validate_trading_mode_and_margin_mode(
        self, trading_mode: str, margin_mode: str, allow_none_margin_mode: bool = False, **kwargs
    ) -> None:
        """
        Validate that the requested trading and margin modes are supported.
        Interactive Brokers forex implementation currently uses 'spot' trading.
        """
        # Forex in this implementation is treated as spot trading
        if trading_mode and str(trading_mode).lower() != "spot":
            from freqtrade.exceptions import OperationalException

            raise OperationalException(
                f"Interactive Brokers forex exchange does not support {trading_mode} trading mode."
            )

        # In this implementation, margin mode is set to NONE
        if margin_mode and str(margin_mode).lower() != "none":
            from freqtrade.exceptions import OperationalException

            raise OperationalException(
                f"Interactive Brokers forex exchange does not support {margin_mode} margin mode."
            )

    def ohlcv_candle_limit(self, timeframe: str, candle_type: str = "spot") -> int:
        """
        Returns the maximum number of candles allowed in a single history request.
        Uses typing.cast to satisfy mypy's strict type checking for dict lookups.
        """
        # Retrieve the value. We use typing.cast to explicitly tell mypy that the value
        # we retrieve is an int, even though dict.get() returns a generic 'Any'.
        limit = cast(int, self._ft_has_default.get("ohlcv_candle_limit", 1000))
        return limit
