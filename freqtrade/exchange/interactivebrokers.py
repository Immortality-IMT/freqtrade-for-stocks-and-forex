# pip install ib_insync
# Interactive Brokers exchange integration for FreqTrade

import asyncio
import logging
import math
import time
from datetime import datetime, timezone
from threading import Event, Lock, Thread
from typing import Any

import pandas as pd
from ib_insync import IB, Contract, Forex, Order, util
from sqlalchemy.engine import Engine

from freqtrade.enums import MarginMode
from freqtrade.exchange.foreignexchange import Foreignexchange
from freqtrade.persistence.trade_model import Trade


logger = logging.getLogger(__name__)


class Interactivebrokers(Foreignexchange):
    """
    Interactive Brokers forex exchange class. Contains adjustments needed for Freqtrade
    to work with IBKR for forex trading.
    """

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
        self.dry_run = config.get("dry_run", False)
        self.latest_ohlcv: dict = {}
        self._running = True
        self._reconnect_event = Event()
        self._connection_thread: Thread | None = None
        self._ws_connected = False
        self._markets_cache: dict[str, Any] | None = None

        self._cache_lock = Lock()
        self._entry_rate_cache = {}
        self._exit_rate_cache = {}

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

    def _connect_to_ib(self) -> None:
        max_retries = 3
        retry_count = 0
        retry_delay = 5

        while retry_count < max_retries:
            try:
                if self.ib.isConnected():
                    self.ib.disconnect()

                logger.info(
                    f"Connecting to IBKR on {self.host}:{self.port} (clientId={self.client_id})"
                )
                self.ib.connect(self.host, self.port, clientId=self.client_id)

                if self.ib.isConnected():
                    logger.info(f"Successfully connected to IBKR on port {self.port}.")
                    util.startLoop()
                    self._setup_event_loop()
                    self._ws_connected = True
                    return
                else:
                    logger.warning("Connection attempt returned without error but not connected")
                    retry_count += 1
            except ConnectionRefusedError as e:
                logger.error(f"Connection refused while connecting to IBKR: {e}")
                retry_count += 1
            except Exception as e:
                logger.error(f"Unexpected error while connecting to IBKR: {e}")
                retry_count += 1

            if retry_count < max_retries:
                logger.info(
                    f"Retrying connection in {retry_delay} seconds..."
                    f"(Attempt {retry_count + 1}/{max_retries})"
                )
                time.sleep(retry_delay)

        logger.error("Failed to connect to IBKR after multiple attempts")
        logger.error("Please ensure TWS or IB Gateway is running with API connections enabled")
        raise ConnectionError("Could not connect to Interactive Brokers")

    def _setup_event_loop(self) -> None:
        if self._connection_thread is not None and self._connection_thread.is_alive():
            logger.info("Event loop thread is already running")
            return

        self._loop = asyncio.new_event_loop()

        def _start_ib_loop():
            asyncio.set_event_loop(self._loop)
            logger.info("Starting IBKR event loop")

            while self._running:
                try:
                    if self.ib.isConnected():
                        self.ib.sleep(1)
                    else:
                        if self._reconnect_event.wait(timeout=5):
                            self._reconnect_event.clear()
                            try:
                                self._connect_to_ib()
                            except Exception as e:
                                logger.error(f"Error during reconnection: {e}")
                except Exception as e:
                    logger.error(f"Error in IBKR event loop: {e}")
                    time.sleep(1)

            logger.info("IBKR event loop stopped")

        self._connection_thread = Thread(target=_start_ib_loop, daemon=True)
        self._connection_thread.start()
        logger.info("Started IBKR event loop in background thread.")

    @property
    def id(self) -> str:
        return "interactivebrokers"

    @property
    def name(self) -> str:
        return "interactivebrokers"

    def get_proxy_coin(self) -> str:
        return self.config.get("stake_currency", "USD")

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
        params = params or {}
        pair = pair[0] if isinstance(pair, tuple) else pair

        contract, amount, price = self._initialize_contract_amount_price(pair, amount, price, rate)

        use_market = ordertype.lower() == "market" or (
            side.lower() == "sell" and params.get("exit_as_market", False)
        )

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

        try:
            trade = self.ib.placeOrder(contract, order)
            logger.info(
                f"Order placed: {order.action} {order.totalQuantity} "
                f"{pair} at {getattr(order, 'lmtPrice', 'MARKET')}"
            )
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return self._failed_response(pair, ordertype, side, amount, price, str(e))

        deadline = time.time() + 30
        while time.time() < deadline and trade.orderStatus.status in (
            "ApiPending",
            "PendingSubmit",
            "Submitted",
        ):
            self.ib.waitOnUpdate(timeout=1)

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

        if status == "Inactive":
            logger.warning(
                f"Order for {pair} was rejected: INACTIVE. Reason: {trade.orderStatus.whyHeld}"
            )
            return self._failed_response(pair, ordertype, side, amount, price, trade.orderStatus)

        if status not in ("PreSubmitted", "Submitted", "Filled"):
            logger.warning(
                f"Order for {pair} failed with status: {status}.Reason: {trade.orderStatus.whyHeld}"
            )
            return self._failed_response(pair, ordertype, side, amount, price, trade.orderStatus)

        oid = str(trade.order.orderId)
        filled = float(trade.orderStatus.filled)
        remaining = float(amount - filled)

        logger.info(
            f"Order {oid} for {pair} placed successfully. Status: {status}, filled: {filled}"
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
            "status": self._parse_order_status(status),
            "info": trade,
        }

    def get_rate(
        self,
        pair: str | tuple,
        side: str | None = None,
        **kwargs,
    ) -> float:
        pair = pair[0] if isinstance(pair, tuple) else pair
        try:
            return self._fetch_live_price(pair, side)
        except Exception as e:
            logger.error(f"Failed to request market data for {pair}: {e}")

        return self._fallback_to_historical_rate(pair)

    def _fetch_live_price(self, pair: str, side: str | None) -> float:
        symbol, currency = pair.split("/")
        contract = Forex(
            symbol=symbol.strip().upper(), currency=currency.strip().upper(), exchange="IDEALPRO"
        )

        ticker = self.ib.reqMktData(contract)
        self.ib.sleep(0.5)
        if ticker and ticker.bid > 0.00001 and ticker.ask > 0.00001:
            if side is None:
                price = (ticker.bid + ticker.ask) / 2
            elif side.lower() == "buy":
                price = ticker.ask
            elif side.lower() == "sell":
                price = ticker.bid
            else:
                price = (ticker.bid + ticker.ask) / 2

            if not (0.00001 <= price <= 1000.0):
                raise ValueError(f"Price out of valid forex range: {price}")

            logger.info(f"Returning price for {pair} ({side}): {price}")
            return price

        raise ValueError(f"Invalid market data for {pair}: bid={ticker.bid}, ask={ticker.ask}")

    def _fallback_to_historical_rate(self, pair: str) -> float:
        try:
            timeframe = self.config.get("timeframe", "5m")
            ohlcv = self.get_historic_ohlcv(pair, timeframe=timeframe, limit=1)
            if not ohlcv.empty:
                close_price = ohlcv.iloc[0]["close"]
                if pd.notna(close_price) and close_price > 0.00001:
                    if not (0.00001 <= close_price <= 1000.0):
                        raise ValueError(f"Invalid historical close price: {close_price}")
                    logger.info(f"Returning historical close price for {pair}: {close_price}")
                    return close_price
                logger.warning(f"Invalid historical close price for {pair}: {close_price}")
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {pair}: {e}")
        raise ValueError(f"Could not fetch valid rate for {pair}")

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

    def fetch_order(
        self,
        order_id: str,
        pair: str | None = None,
    ) -> dict | None:
        try:
            oid = int(order_id)
        except ValueError:
            logger.error(f"Invalid order ID format: {order_id}")
            return None

        try:
            for trade in self.ib.trades():
                if trade.order.orderId == oid:
                    filled = float(trade.orderStatus.filled)
                    total = float(trade.order.totalQuantity)
                    remaining = total - filled

                    if hasattr(trade.contract, "symbol") and hasattr(trade.contract, "currency"):
                        symbol = f"{trade.contract.symbol}/{trade.contract.currency}"
                    else:
                        symbol = pair if pair else "UNKNOWN/UNKNOWN"

                    price: float | None = None

                    if trade.order.orderType == "LMT":
                        price = float(trade.order.lmtPrice)
                    elif hasattr(trade, "fills") and trade.fills:
                        total_cost = sum(
                            fill.execution.price * fill.execution.shares for fill in trade.fills
                        )
                        total_shares = sum(fill.execution.shares for fill in trade.fills)
                        price = total_cost / total_shares if total_shares > 0 else None
                    else:
                        price = None
                    return {
                        "id": order_id,
                        "symbol": symbol,
                        "type": trade.order.orderType.lower(),
                        "side": trade.order.action.lower(),
                        "amount": total,
                        "price": price,
                        "filled": filled,
                        "remaining": remaining,
                        "status": self._parse_order_status(trade.orderStatus.status),
                        "info": trade,
                    }

            logger.debug(f"fetch_order: no trade with orderId={order_id}")
            return None

        except Exception as e:
            logger.error(f"Error in fetch_order for {order_id}: {e}")
            return None

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

    def cancel_order(self, order_id: str) -> None:
        try:
            self.ib.cancelOrder(int(order_id))
            logger.info(f"Order {order_id} cancel request sent successfully.")
        except ValueError as e:
            logger.error(f"Invalid order ID format when canceling {order_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")

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
        try:
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",  # Current time
                durationStr=durationStr,
                barSizeSetting=ib_timeframe,
                whatToShow="MIDPOINT",
                useRTH=False,
                keepUpToDate=True,
            )
            if not bars:
                logger.warning(
                    f"No historical data returned for contract:"
                    f"{contract.symbol}/{contract.currency}"
                )
                return []
            return bars
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise

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
            bars = self.ib.run(self.fetch_historical_data(contract, durationStr, ib_timeframe))

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
                current_time = datetime.now(timezone.utc)
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

    def close(self) -> None:
        self.ib.disconnect()
        logger.info("Disconnected from IBKR.")

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
        return self.ib.isConnected() and self._ws_connected

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
            if isinstance(arg, dict) and "id" in arg:
                order_id = arg["id"]
                break
            if hasattr(arg, "order_id"):
                order_id = arg.order_id
                break

        if not order_id:
            logger.error(f"cancel_order_with_result: Can't extract order_id from {args}")
            return None

        try:
            self.cancel_order(order_id)
        except Exception as e:
            logger.error(f"cancel_order_with_result: error canceling {order_id}: {e}")

        updated = self.fetch_order(order_id)
        if updated:
            updated["status"] = updated.get("status", "canceled")
            return updated

        return {
            "id": order_id,
            "status": "canceled",
            "filled": 0.0,
            "remaining": 0.0,
        }

    def is_market_open(self):
        now = datetime.now(timezone.utc)
        if now.weekday() == 4 and now.hour >= 22:
            return False
        if now.weekday() >= 5:
            return False
        return True

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
        self.ib.reqOpenOrders()
        orders = []
        for o in self.ib.openOrders():
            sym = f"{o.contract.symbol}/{o.contract.currency}"
            if symbol and sym != symbol:
                continue
            filled = float(o.orderStatus.filled)
            total = float(o.order.totalQuantity)
            orders.append(
                {
                    "id": str(o.order.orderId),
                    "symbol": sym,
                    "type": o.order.orderType.lower(),
                    "side": o.order.action.lower(),
                    "amount": total,
                    "price": getattr(o.order, "lmtPrice", None),
                    "filled": filled,
                    "remaining": total - filled,
                    "status": self._parse_order_status(o.orderStatus.status),
                    "info": {},
                }
            )
        return orders

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
                    "info": {},  # strip non serializable objects
                }
            )
        return positions

    def fetch_ticker(self, symbol: str) -> dict:
        # Reuse fetch_tickers under the hood
        return self.fetch_tickers([symbol])[symbol]

    def fetch_tickers(self, symbols: list[str] | None = None) -> dict[str, dict]:
        tickers = {}
        symbols = symbols or [p["symbol"] for p in self.fetch_positions()]

        for sym in symbols:
            contract = self._get_contract(sym)
            data = self.ib.reqMktData(contract, "", False, False)

            # Wait briefly for data (optional)
            # time.sleep(0.1)

            bid = data.bid
            ask = data.ask
            last = (bid + ask) / 2 if bid and ask else data.last

            # Sanity check: IBKR sometimes returns nan — bail early
            if any(math.isnan(v) for v in [bid, ask, last]):
                raise ValueError(
                    f"Invalid market data for {sym}: bid={bid}, ask={ask}, last={last}"
                )

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

    def disconnect(self) -> None:
        """
        Gracefully disconnect from IBKR and dispose of all SQLAlchemy connections
        via the ORM sessions bound engine to avoid pool exhaustion.
        """
        try:
            logger.info("Disconnecting from IBKR...")
            # Stop your IBKR WS loop (if any)
            try:
                self.ws_stop()
            except Exception as e:
                logger.warning(f"Failed to stop websocket: {e}")
            # Disconnect IB connection
            if self.ib.isConnected():
                self.ib.disconnect()
            logger.info("Disconnected from IBKR.")
        finally:
            # Tear down all pooled DB connections via the sessions engine
            try:
                bind = Trade.session.get_bind()
                if isinstance(bind, Engine):
                    bind.dispose()
                    logger.info("Disposed of engine via Trade.session.")
                else:
                    logger.warning("Session bind is not an Engine, skipping dispose().")
            except Exception:
                logger.exception(
                    "Could not dispose DB engine via Trade.session; pool may still exhaust."
                )

    def stop(self) -> None:
        """
        Hook called by Freqtrade when the bot is stopping.
        Ensures proper IBKR disconnection and DB cleanup.
        """
        self.disconnect()
