# pip install ib_insync
# Interactive Brokers exchange integration for FreqTrade

import asyncio
import logging
import math
import time
from datetime import datetime, timezone
from threading import Event, Thread
from typing import Any

import pandas as pd
from ib_insync import IB, Contract, Forex, Order, util

from freqtrade.enums import MarginMode
from freqtrade.exchange.foreignexchange import Foreignexchange


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
        "ws_reconnect_interval": 300,
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
        self._markets_cache: dict[str, Any] = {}

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
        pair: str,
        ordertype: str,
        side: str,
        amount: float,
        price: float | None = None,
        params: dict[Any, Any] | None = None,
        rate: float | None = None,
        **kwargs,
    ) -> dict:
        params = params or {}

        if rate is not None and (price is None or price <= 0):
            price = rate

        try:
            contract = self._create_and_qualify_contract(pair)
        except ValueError as e:
            logger.error(f"Contract qualification failed: {e}")
            return {"id": None, "status": "failed", "info": str(e)}

        amount = self._adjust_amount_for_min_lot(amount)

        # Use the helper method
        use_market = self._should_use_market_order(ordertype, side, params)

        if use_market:
            order = self._create_order_object(side, amount, None, ordertype)
        else:
            try:
                if price is None or price <= 0:
                    price = self.get_rate(pair, side=side)

                if not (0.00001 <= price <= 1000.0):
                    raise ValueError(f"Invalid price for order: {price}")

                order = self._create_order_object(side, amount, price, ordertype)
            except ValueError as e:
                logger.error(f"Failed to get valid price for order: {e}")
                return {"id": None, "status": "failed", "info": str(e)}

        try:
            trade = self.ib.placeOrder(contract, order)
            logger.info(
                f"Order placed: {order.action} {order.totalQuantity} {pair}"
                f" at {getattr(order, 'lmtPrice', 'MARKET')}"
            )
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {"id": None, "status": "failed", "info": str(e)}

        return self._wait_for_order_status(trade)

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

                    if trade.order.orderType == "LMT":
                        price = (
                            float(trade.order.lmtPrice)
                            if trade.order.lmtPrice is not None
                            else None
                        )
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
    ) -> dict:
        if hasattr(self, "_markets_cache") and not reload:
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

    def fetch_positions(self) -> list:
        positions = self.ib.positions()
        formatted_positions: list[dict[str, Any]] = []
        for pos in positions:
            formatted_positions.append(
                {
                    "symbol": f"{pos.contract.symbol}/{pos.contract.currency}",
                    "amount": float(pos.position),
                    "side": "long" if pos.position > 0 else "short",
                    "leverage": 1.0,
                    "contracts": abs(pos.position),
                    "contractSize": 1,
                    "unrealizedPnl": float(pos.unrealizedPNL),
                    "info": pos,
                }
            )
        return formatted_positions

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

    def get_rate(
        self,
        pair: str,
        side: str | None = None,
        is_short: bool | None = None,
        refresh: bool = False,
        **kwargs,
    ) -> float:
        if isinstance(pair, tuple):
            pair = pair[0]

        market_price = self._get_market_data_price(pair, side)
        if market_price is not None:
            logger.info(f"Returning market price for {pair} ({side}): {market_price}")
            return market_price

        timeframe = self.config.get("timeframe", "5m")
        historical_price = self._get_historical_price(pair, timeframe)
        if historical_price is not None:
            logger.info(f"Returning historical close price for {pair}: {historical_price}")
            return historical_price

        raise ValueError(f"Could not fetch valid rate for {pair}")

    def get_min_pair_stake_amount(self, pair: str, leverage: float = 1.0, *args, **kwargs) -> float:
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

    def _create_and_qualify_contract(self, pair: str) -> Contract:
        symbol, currency = self._extract_currencies_from_pair(pair)
        contract = Forex(symbol=symbol, currency=currency, exchange="IDEALPRO")
        if not self.ib.qualifyContracts(contract):
            raise ValueError(f"Contract qualification failed for {pair}")
        return contract

    def _adjust_amount_for_min_lot(self, amount: float) -> float:
        min_lot = self.MIN_LOT_SIZE
        return max(min_lot, math.floor(amount / min_lot) * min_lot)

    def _create_order_object(
        self, side: str, amount: float, price: float | None, ordertype: str
    ) -> Order:
        if ordertype.lower() == "market":
            return Order(action=side.upper(), totalQuantity=amount, orderType="MKT")
        else:
            if price is None:
                raise ValueError("Price must be provided for limit orders")
            return Order(
                action=side.upper(),
                totalQuantity=amount,
                orderType="LMT",
                lmtPrice=round(price, self.SIGNIFICANT_DIGITS - 1),
            )

    def _wait_for_order_status(self, trade) -> dict:
        deadline = time.time() + 30
        while time.time() < deadline and trade.orderStatus.status in (
            "ApiPending",
            "PendingSubmit",
            "Submitted",
        ):
            self.ib.waitOnUpdate(timeout=1)

        status = trade.orderStatus.status
        filled = float(trade.orderStatus.filled)
        remaining = float(trade.order.totalQuantity - filled)

        if status == "Inactive":
            return {
                "id": None,
                "status": "rejected",
                "info": trade.orderStatus,
            }

        if status not in ("PreSubmitted", "Submitted", "Filled"):
            return {
                "id": None,
                "status": "failed",
                "info": trade.orderStatus,
            }

        oid = str(trade.order.orderId)
        return {
            "id": oid,
            "status": self._parse_order_status(status),
            "filled": filled,
            "remaining": remaining,
            "info": trade,
        }

    def _get_market_data_price(self, pair: str, side: str | None = None) -> float | None:
        parts = pair.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid pair format: {pair}. Expected format 'BASE/QUOTE'")

        symbol, currency = parts[0].strip().upper(), parts[1].strip().upper()
        contract = Forex(symbol=symbol, currency=currency, exchange="IDEALPRO")

        try:
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

                if 0.00001 <= price <= 1000.0:
                    return price
                else:
                    logger.warning(f"Invalid price from market data: {price}")
        except Exception as e:
            logger.error(f"Failed to request market data for {pair}: {e}")
        return None

    def _get_historical_price(self, pair: str, timeframe: str) -> float | None:
        try:
            ohlcv = self.get_historic_ohlcv(pair, timeframe=timeframe, limit=1)
            if not ohlcv.empty:
                close_price = ohlcv.iloc[0]["close"]
                if pd.notna(close_price) and close_price > 0.00001:
                    if 0.00001 <= close_price <= 1000.0:
                        return close_price
                    logger.warning(f"Invalid historical close price: {close_price}")
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {pair}: {e}")
        return None

    def _should_use_market_order(self, ordertype: str, side: str, params: dict) -> bool:
        return ordertype.lower() == "market" or (
            side.lower() == "sell" and params.get("exit_as_market", False)
        )

    def _get_valid_price(self, pair: str, side: str) -> float:
        price = self.get_rate(pair, side=side)
        if not (0.00001 <= price <= 1000.0):
            raise ValueError(f"Invalid price: {price}")
        return price

    def _place_and_wait_for_order(self, contract: Contract, order: Order) -> dict:
        try:
            trade = self.ib.placeOrder(contract, order)
            logger.info(f"Order placed: {order.action} {order.totalQuantity} {contract.symbol}")
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {"id": None, "status": "failed", "info": str(e)}

        return self._wait_for_order_status(trade)
