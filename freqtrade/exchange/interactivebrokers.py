# pip install ib_insync
# Interactive brokers

import logging
import math
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

    DECIMAL_PLACES = 6  # Forex typically uses 5 decimal places
    SIGNIFICANT_DIGITS = 6
    TICK_SIZE = 0.000001  # Minimum price movement for forex
    MAX_DATA_DELAY = pd.Timedelta(minutes=5)  # Allowed data delay during market hours

    _ft_has_default = {
        "stoploss_on_exchange": False,
        "order_time_in_force": ["GTC", "IOC", "FOK"],
        "ohlcv_candle_limit": 500,
        "ohlcv_has_history": True,
        "ohlcv_partial_candle": True,
        "ohlcv_require_since": False,  # IBKR doesn't use 'since' directly in reqHistoricalData
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
        # Initialize IBKR connection
        self.ib = IB()
        self.dry_run = config.get("dry_run", False)
        self.latest_ohlcv: dict = {}
        # Determine the port based on dry_run
        if self.dry_run:
            self.port = 4002  # IB Gateway paper trading port
            logger.info("Connecting to IBKR paper trading (IB Gateway).")
        else:
            self.port = 7497  # TWS live trading port
            logger.info("Connecting to IBKR live trading (TWS).")
        # Connect to TWS/IB Gateway
        try:
            self.ib.connect("127.0.0.1", self.port, clientId=1)
            if self.ib.isConnected():
                logger.info(f"Successfully connected to IBKR on port {self.port}.")
        except ConnectionRefusedError as e:
            logger.error(f"Unexpected error while connecting to IBKR: {e}")
            logger.error("Failed to connect to IBKR. Is TWS from Interactive Brokers running?")
            logger.error("Get TWS: interactivebrokers.com/en/trading/tws-updatable-latest.php")
            exit(1)

        self.margin_mode = MarginMode.NONE

        self.ws_start()
        self.markets = self.get_markets()
        if not self.ib.isConnected():
            raise ConnectionError("WebSocket connection failed")

    @property
    def id(self) -> str:
        return "interactivebrokers"

    @property
    def name(self) -> str:
        return "interactivebrokers"

    def get_proxy_coin(self) -> str:
        """
        Return the stake currency (proxy coin) used for trading.
        This is typically the quote currency of the trading pair (e.g., USD for EUR/USD).
        """
        return self.config.get("stake_currency", "USD")

    def create_order(
        self,
        pair: str,
        ordertype: str,
        side: str,
        amount: float,
        price: float | None = None,
        params: dict[Any, Any] | None = None,
        **kwargs,
    ) -> dict:
        # Build the contract from the pair
        parts = pair.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid pair format: {pair}. Expected format 'BASE/QUOTE'")
        symbol, currency = parts[0].strip().upper(), parts[1].strip().upper()
        contract = Forex(symbol=symbol, currency=currency, exchange="IDEALPRO")

        if ordertype == "limit":
            # If no valid price is provided, attempt to fetch one using get_rate.
            if price is None or price <= 0:
                logger.info(
                    f"No valid price for {pair}, call get_rate to get price for side {side}."
                )
                price = self.get_rate(pair, side=side)
            if price is None or price <= 0:
                logger.error(
                    f"Attempted to create limit order for {pair} with invalid price: {price}"
                )
                raise ValueError(f"Limit order for {pair} requires a valid price > 0.")
            formatted_price = round(price, 5)
            order = Order(
                action=side, totalQuantity=round(amount), orderType="LMT", lmtPrice=formatted_price
            )
        elif ordertype == "market":
            order = Order(action=side, totalQuantity=round(amount), orderType="MKT")
        else:
            raise ValueError(f"Unsupported order type: {ordertype}")

        # Place the order via IBKR API
        trade = self.ib.placeOrder(contract, order)
        # Return a CCXT-style order dict
        return {
            "id": str(trade.order.orderId),
            "symbol": pair,
            "type": ordertype.lower(),
            "side": side.lower(),
            "amount": amount,
            "price": price,
            "filled": 0.0,
            "remaining": amount,
            "status": "open",
            "info": trade.order,
        }

    def fetch_order(
        self,
        order_id: str,
        pair: str | None = None,  # Changed from Optional[str]
    ) -> dict | None:
        """Fetch order from IBKR and format as CCXT order."""
        trades = self.ib.trades()
        for trade in trades:
            if str(trade.order.orderId) == order_id:
                # Convert IBKR trade to CCXT-style order dict
                filled = trade.orderStatus.filled
                remaining = trade.orderStatus.remaining
                return {
                    "id": str(trade.order.orderId),
                    "symbol": f"{trade.contract.symbol}/{trade.contract.currency}",
                    "type": trade.order.orderType.lower(),
                    "side": "buy" if trade.order.action == "BUY" else "sell",
                    "amount": trade.order.totalQuantity,
                    "price": trade.order.lmtPrice if trade.order.orderType == "LMT" else None,
                    "filled": filled,
                    "remaining": remaining,
                    "status": self._parse_order_status(trade.orderStatus.status),
                    "info": trade,
                }
        return None

    def _parse_order_status(self, ib_status: str) -> str:
        """Map IBKR order status to CCXT status."""
        status_mapping = {
            "ApiPending": "open",
            "PendingSubmit": "open",
            "PreSubmitted": "open",
            "Submitted": "open",
            "Filled": "closed",
            "Cancelled": "canceled",
            "Inactive": "canceled",
        }
        return status_mapping.get(ib_status, "unknown")

    def cancel_order(self, order_id: str) -> None:
        """Cancel an order on IBKR."""
        self.ib.cancelOrder(order_id)
        logger.info(f"Order {order_id} canceled successfully.")

    def get_markets(
        self,
        reload: bool = False,
        params: dict[Any, Any] | None = None,
        tradable_only: bool = False,
        active_only: bool = False,
    ):
        """Return a dictionary of available markets."""
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
                    "amount": {"min": 0.01, "max": 1000000},
                    "price": {"min": 0.00001, "max": 1000000},
                    "cost": {"min": 0.01, "max": 1000000},
                },
                "active": True,
                "info": {"base": base, "quote": quote},
            }
        return markets

    def reload_markets(self, params: dict[Any, Any] | None = None) -> dict:
        """
        Reload the markets data and update self.markets.
        :param params: Optional parameters for market fetching.
        :return: Updated markets dictionary.
        """
        self.markets = self.get_markets(reload=True, params=params)
        logger.info("Markets reloaded successfully.")
        return self.markets

    def get_fee(self, symbol: str, now: Any = None, taker_or_maker: str = "maker") -> float:
        """Get the fee structure for a symbol."""
        return 0.0001 if taker_or_maker == "maker" else 0.0002

    def get_historic_ohlcv(
        self,
        pair: str,
        since: int | None = None,
        timeframe: str | None = None,
        limit: int = 1000,
        params: dict[Any, Any] | None = None,
        since_ms: int | None = None,
        is_new_pair: bool = True,
        candle_type: str = "spot",
        until_ms: int | None = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from IBKR.
        :param pair: The trading pair (e.g., 'EUR/USD').
        :param timeframe: Timeframe string (e.g., '1h', '5m').
        :param since: Start timestamp for data retrieval.
        :param limit: Maximum number of candles to fetch.
        :param params: Additional parameters.
        :param since_ms: Start timestamp in milliseconds.
        :param is_new_pair: Whether the pair is new.
        :param candle_type: Type of candle (e.g., 'spot').
        :param until_ms: End timestamp in milliseconds.
        :return: DataFrame containing OHLCV data.
        """
        if isinstance(pair, tuple):
            pair = pair[0]
        parts = pair.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid pair format: {pair}. Expected format 'BASE/QUOTE'")
        symbol = parts[0].strip().upper()
        currency = parts[1].strip().upper()
        full_pair = symbol + currency
        logger.info("Constructing Forex contract for pair: %s", full_pair)
        if len(symbol) != 3 or len(currency) != 3 or len(full_pair) != 6:
            raise ValueError(
                f"Invalid currency codes: symbol={symbol} (len={len(symbol)}), "
                f"currency={currency} (len={len(currency)}). "
                f"Combined: '{full_pair}' (len={len(full_pair)}). "
                "Expected 3-letter codes for each."
            )
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "CASH"
        contract.currency = currency
        contract.exchange = "IDEALPRO"
        if timeframe is None:
            timeframe = self.config.get("timeframe", "1h")
        ib_timeframe = self._convert_timeframe(timeframe)
        durationStr = self._calculate_duration(timeframe, limit)
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=durationStr,
            barSizeSetting=ib_timeframe,
            whatToShow="MIDPOINT",
            useRTH=True,
        )
        if bars is None:
            raise ValueError("No historical data returned from IBKR")
        df = util.df(bars)
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
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        elif "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"])
        return df

    def refresh_latest_ohlcv(self, pairs: list) -> None:
        """
        Refresh the latest OHLCV data for the given pairs.
        Freqtrade may call this method with a tuple in the form:
            (pair, timeframe, candle_type)
        """
        for item in pairs:
            if isinstance(item, tuple):
                pair, timeframe, candle_type = item
            else:
                pair = item
                timeframe = self.config.get("timeframe", "1h")
            try:
                ohlcv = self.get_historic_ohlcv(pair, None, timeframe, 1)  # Passing None for since
                self.latest_ohlcv[pair] = ohlcv
                logger.info("Refreshed latest OHLCV for %s", pair)
            except Exception as e:
                logger.error("Failed to refresh latest OHLCV for %s: %s", pair, e)

    def klines(
        self,
        pair: str,
        timeframe: str | None = None,
        since: int = 0,
        limit: int = 1000,
        params: dict[Any, Any] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Retrieve OHLCV data for a given trading pair and timeframe.
        :param pair: Trading pair (e.g., 'EUR/USD')
        :param timeframe: Timeframe string (e.g., '5m', '1h'). Defaults to the config timeframe.
        :param since: Timestamp from which to retrieve data. Defaults to 0.
        :param limit: Maximum number of candles to retrieve.
        :param params: Additional parameters if needed. Defaults to an empty dict.
        :param kwargs: Extra keyword arguments passed by Freqtrade.
        :return: DataFrame containing OHLCV data.
        """
        if params is None:
            params = {}
        if timeframe is None:
            timeframe = self.config.get("timeframe", "1h")
        return self.get_historic_ohlcv(pair, since, timeframe, limit)

    def get_balances(self):
        """Retrieve account balances."""
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
        """Fetch open positions."""
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
        """Disconnect from IBKR."""
        self.ib.disconnect()
        logger.info("Disconnected from IBKR.")

    def market_is_tradable(self, market: dict) -> bool:
        """
        Check if a market is tradable by verifying if the pair is active and available.
        """
        return market.get("active", False)

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
        self.ib.connect("127.0.0.1", self.port, clientId=1)
        logger.info("WebSocket connection reset")

    def ws_start(self) -> None:
        if not self.ib.isConnected():
            self.ib.connect("127.0.0.1", self.port, clientId=1)
        logger.info("WebSocket started")

    def ws_stop(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()
        logger.info("WebSocket stopped")

    def ws_health_check(self) -> bool:
        return self.ib.isConnected()

    def _convert_timeframe(self, timeframe: str) -> str:
        mapping = {
            "1m": "1 min",
            "5m": "5 mins",
            "15m": "15 mins",
            "30m": "30 mins",
            "1h": "1 hour",
            "2h": "2 hours",
            "3h": "3 hours",
            "4h": "4 hours",
            "8h": "8 hours",
            "1d": "1 day",
        }
        if timeframe not in mapping:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        return mapping[timeframe]

    def _calculate_duration(self, timeframe: str, limit: int) -> str:
        if timeframe.endswith("m"):
            minutes = int(timeframe[:-1])
            days_per_candle = minutes / (60 * 24)
        elif timeframe.endswith("h"):
            hours = int(timeframe[:-1])
            days_per_candle = hours / 24
        elif timeframe.endswith("d"):
            days_per_candle = 1
        else:
            days_per_candle = 1
        total_days = math.ceil(limit * days_per_candle)
        if total_days > 365:
            years = math.ceil(total_days / 365)
            return f"{years} Y"
        else:
            return f"{total_days} D"

    @property
    def precisionMode(self):
        return 4

    @property
    def precision_mode_price(self):
        return 4

    def get_precision_price(self, pair):
        return 2

    def validate_required_startup_candles(self, startup_candle_count, timeframe):
        pass

    def validate_timeframes(self, timeframes):
        """
        Validate the timeframes supported by the exchange.

        :param timeframes: List of timeframes to validate.
        """
        if isinstance(timeframes, str):
            timeframes = [timeframes]

        supported_timeframes = ["1m", "5m", "15m", "1h", "1d"]
        logger.info(f"Validating timeframes: {timeframes}")
        for timeframe in timeframes:
            logger.info(f"Validating timeframe: {timeframe}")
            if timeframe not in supported_timeframes:
                raise ValueError(
                    f"Timeframe '{timeframe}' is not supported by Interactive Brokers."
                )

    def get_option(self, option: str, default: Any = None) -> Any:
        """
        Retrieve the value for a given option from the _ft_has_default dictionary.
        :param option: The option key to retrieve.
        :param default: Default value if the option is not found.
        :return: The value of the option or the default.
        """
        return self._ft_has_default.get(option, default)

    def get_max_leverage(self, pair: str, stake_amount: float) -> float:
        """
        Get the maximum leverage available for a given trading pair and stake amount.
        :param pair: The trading pair (e.g., 'EUR/USD').
        :param stake_amount: The amount of stake currency.
        :return: Maximum leverage available.
        """
        return 1.0  # Assuming a leverage of 1 (no leverage)

    def get_min_pair_stake_amount(self, pair: str, *args, **kwargs) -> float:
        """
        Get the minimum stake amount required for a given trading pair.
        """
        return 10.0  # or your custom logic for minimum stake

    def get_max_pair_stake_amount(self, pair: str, *args, **kwargs) -> float:
        """
        Get the maximum stake amount allowed for a given trading pair.
        """
        return 1000000.0  # matches the limit defined in get_markets()

    def get_valid_price_and_stake(self, row, pair, leverage):
        propose_rate = row["close"]
        stake_amount = self.calculate_stake_amount(pair, propose_rate, leverage)
        min_stake_amount = self.exchange.get_min_pair_stake_amount(
            pair
        )  # Removed leverage argument here
        return propose_rate, stake_amount, leverage, min_stake_amount

    def get_contract_size(self, pair: str) -> float:
        """
        Returns the contract size (lot size) for a given Forex pair.
        Many Forex brokers use 100,000 units as a standard lot size.
        """
        return 100000.0

    def get_precision_amount(self, pair):
        return 2

    def get_rate(
        self,
        pair: str,
        side: str | None = None,
        is_short: bool | None = None,
        refresh: bool = False,
        **kwargs,
    ) -> float:
        """
        Get the current exchange rate for a given trading pair.
        If side is provided ('buy' or 'sell'), it returns the
        ask or bid price respectively; otherwise, returns the
        midpoint price.

        :param pair: The trading pair (e.g., 'EUR/USD').
        :param side: Optional parameter, 'buy' or 'sell' to get ask or bid price.
        :param is_short: Optional parameter indicating if the position is short.
        :param refresh: Optional parameter to force refreshing the market data.
        :return: The current exchange rate as a float.
        :raises ValueError: If no data is available for the pair.
        """
        # if isinstance(pair, tuple):
        #    pair = pair[0]

        # Ensure the pair is split properly and standardized
        # symbol, currency = [s.strip().upper() for s in pair.split('/')]
        # Now 'symbol' and 'currency' should both be exactly 3 characters for standard forex pairs.
        # contract = Forex(symbol=symbol, currency=currency)

        parts = pair.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid pair format: {pair}. Expected format 'BASE/QUOTE'")
        symbol, currency = parts[0].strip().upper(), parts[1].strip().upper()
        contract = Forex(symbol=symbol, currency=currency, exchange="IDEALPRO")

        if self.ib.isConnected() and side is not None:
            # Request real-time market data for the pair
            try:
                market_data = self.ib.reqMktData(contract)
                if market_data is not None:
                    if side.lower() == "buy":
                        price = market_data.ask
                        if price and price > 0:
                            logger.info(f"Returning ask price for {pair}: {price}")
                            return price
                        else:
                            logger.warning(f"Invalid ask price received for {pair}: {price}")
                    elif side.lower() == "sell":
                        price = market_data.bid
                        if price and price > 0:
                            logger.info(f"Returning bid price for {pair}: {price}")
                            return price
                        else:
                            logger.warning(f"Invalid bid price received for {pair}: {price}")
            except Exception as e:
                logger.error(f"Failed to request market data for {pair}: {e}")

        # Fallback to historical data for midpoint price
        timeframe = self.config.get("timeframe", "5m")
        ohlcv = self.get_historic_ohlcv(
            pair, None, timeframe=timeframe, limit=1
        )  # Passing None for since
        if not ohlcv.empty:
            close_price = ohlcv.iloc[0]["close"]
            if close_price and close_price > 0:
                logger.info(f"Returning historical close price for {pair}: {close_price}")
                return close_price
            else:
                logger.warning(
                    f"Historical data returned invalid close price for {pair}: {close_price}"
                )
        raise ValueError(f"Could not fetch current rate for {pair}")

    def get_funding_fees(self, pair: str, timeframe: str | None = None, **kwargs) -> float:
        """
        Return the funding fee for a given trading pair.
        For Forex trading, funding fees may be negligible or not applicable,
        so returning 0.0 is appropriate.
        """
        return 0.0

    def fetch_order_or_stoploss_order(
        self,
        order_id: str,
        pair: str | None = None,
        *args,
        **kwargs,
    ) -> dict:
        """
        Fetch order details from IBKR, including stoploss orders.

        :param order_id: The order ID to fetch.
        :param pair: The trading pair (optional).
        :param args: Additional positional arguments (ignored).
        :param kwargs: Additional keyword arguments (ignored).
        :return: Order details or a dictionary indicating the order status.
        """
        order = self.fetch_order(order_id, pair)
        if order is None:
            return {"status": "not_found"}
        return order

    def check_order_canceled_empty(self, order: dict) -> bool:
        """
        Checks if the order is canceled and has no remaining quantity.
        This is relevant for some exchanges that might return partially filled
        and then canceled orders as "canceled" even if they have a filled amount.

        :param order: Order dictionary as returned by `fetch_order`
        :return: True if the order is canceled and empty (no remaining quantity), False otherwise.
        """
        if not order:
            return False  # Or maybe True, depending on how empty orders are represented
        return order.get("status") == "canceled" and order.get("remaining", 0) == 0

    def order_has_fee(self, order) -> bool:
        """
        Return whether the order object includes fee information.
        For IBKR, fees might not be available, so we return False.
        """
        return False

    def get_trades_for_order(self, order, *args, **kwargs):
        """Retrieve trades associated with the given order, ignoring extra arguments."""
        if not order:
            return []

        # Extract the order_id from the order parameter
        order_id = None
        if isinstance(order, dict):
            order_id = order.get("order_id", None)
        else:
            order_id = getattr(order, "order_id", None)

        if order_id is None:
            return []

        # Query IB for trades
        trades = self.ib.trades()
        matching_trades = []

        for trade in trades:
            if hasattr(trade.order, "orderId") and trade.order.orderId == order_id:
                matching_trades.append(trade)

        return matching_trades

    def get_order_id_conditional(self, order: dict | None) -> str | None:
        """Extract order ID from the order object or dictionary."""
        if not order:
            return None
        if isinstance(order, dict):
            return order.get("order_id") if "order_id" in order else None
        return None

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
        """
        Return the liquidation price for a given trade.
        Since this is spot forex (or treated as such),
        liquidation price is not applicable.
        Therefore, we return None.
        Corrected syntax error by providing default values
        for all arguments after the first default.
        """
        return None
