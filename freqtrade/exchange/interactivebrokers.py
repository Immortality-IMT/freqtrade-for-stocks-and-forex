# pip install ib_insync
# Interactive brokers

import logging
import math
from typing import Any

import pandas as pd
from ib_insync import IB, Contract, Forex, Order, util

from freqtrade.exchange.foreignexchange import Foreignexchange


logger = logging.getLogger(__name__)


class Interactivebrokers(Foreignexchange):
    """
    Interactive Brokers forex exchange class. Contains adjustments needed for Freqtrade
    to work with IBKR for forex trading.
    """

    DECIMAL_PLACES = 5  # Forex typically uses 5 decimal places
    SIGNIFICANT_DIGITS = 6
    TICK_SIZE = 0.00001  # Minimum price movement for forex
    MAX_DATA_DELAY = pd.Timedelta(minutes=5)  # Allowed data delay during market hours

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
        self.ib.connect("127.0.0.1", self.port, clientId=1)
        if self.ib.isConnected():
            logger.info(f"Successfully connected to IBKR on port {self.port}.")
        else:
            logger.error("Failed to connect to IBKR.")
        self.ws_start()
        self.markets = self.get_markets()
        if not self.ib.isConnected():
            raise ConnectionError("WebSocket connection failed")

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
    ) -> int:
        if params is None:
            params = {}
        """Create an order on IBKR."""
        symbol = pair.split("/")[0]
        currency = pair.split("/")[1]
        # Define forex contract
        contract = Forex(symbol, currency)
        if ordertype == "market":
            order = Order(action=side, totalQuantity=amount, orderType="MKT")
        elif ordertype == "limit":
            order = Order(action=side, totalQuantity=amount, orderType="LMT", lmtPrice=price)
        else:
            raise ValueError(f"Unsupported order type: {ordertype}")
        trade = self.ib.placeOrder(contract, order)
        return trade.order.orderId

    def cancel_order(self, order_id: str) -> None:
        """Cancel an order on IBKR."""
        self.ib.cancelOrder(order_id)
        logger.info(f"Order {order_id} canceled successfully.")

    def fetch_order(self, order_id: str):
        """Fetch order details from IBKR."""
        trades = self.ib.trades()
        for trade in trades:
            if trade.order.orderId == order_id:
                return trade
        return None

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

    def get_historical_ohlcv(
        self,
        pair: str,
        since: int,
        timeframe: str | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from IBKR.
        :param pair: The trading pair (e.g., 'EUR/USD').
        :param timeframe: Timeframe string (e.g., '1h', '5m').
        :param since: Start timestamp for data retrieval.
        :param limit: Maximum number of candles to fetch.
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
                ohlcv = self.get_historical_ohlcv(pair, 0, timeframe, 1)
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
        return self.get_historical_ohlcv(pair, since, timeframe, limit)

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
