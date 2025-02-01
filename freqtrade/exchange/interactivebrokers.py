# pip install ib_insync
# Interactive brokers
import logging

import pandas as pd
from ib_insync import IB, Forex, Order, util

from freqtrade.exchange.foreignexchange import Foreignexchange


logger = logging.getLogger(__name__)


class Interactivebrokers(Foreignexchange):
    """
    Interactive Brokers forex exchange class. Contains adjustments needed for Freqtrade to work
    with IBKR for forex trading.
    """

    DECIMAL_PLACES = 5  # Forex typically uses 5 decimal places
    SIGNIFICANT_DIGITS = 6
    TICK_SIZE = 0.00001  # Minimum price movement for forex
    MAX_DATA_DELAY = pd.Timedelta(minutes=5)  # Allowed data delay during market hours

    _ft_has_default = {
        "stoploss_on_exchange": False,
        "order_time_in_force": ["GTC", "IOC", "FOK"],  # Supported TIF options
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
        "ws_enabled": True,  # Enable WebSocket for real-time updates
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
        self.ib.connect("127.0.0.1", 7497, clientId=1)  # Default TWS/Gateway port

    @property
    def name(self):
        return "interactivebrokers"

    def create_order(
        self,
        pair: str,
        ordertype: str,
        side: str,
        amount: float,
        price: float | None = None,
        params=None,
    ) -> int:
        """Create an order on IBKR."""
        if params is None:
            params = {}

        symbol = pair.split("/")[0]
        currency = pair.split("/")[1]

        # Define forex contract
        contract = Forex(symbol, currency)

        # Define order type
        if ordertype == "market":
            order = Order(action=side, totalQuantity=amount, orderType="MKT")
        elif ordertype == "limit":
            order = Order(action=side, totalQuantity=amount, orderType="LMT", lmtPrice=price)
        else:
            raise ValueError(f"Unsupported order type: {ordertype}")

        # Place order
        trade = self.ib.placeOrder(contract, order)
        return trade.order.orderId  # Return order ID for tracking

    def cancel_order(self, order_id: str):
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

    def get_markets(self, reload=False, params=None, tradable_only=False, active_only=False):
        """Retrieve a list of forex pairs available on IBKR."""
        markets = {}

        # Example forex pairs (you can dynamically fetch these from IBKR)
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
                "spot": True,
                "margin": True,  # Forex is always margin-based
                "active": True,
                "maker": 0.0001,  # Example maker fee
                "taker": 0.0002,  # Example taker fee
                "info": {"base": base, "quote": quote},
                "precision": {"amount": 2, "price": 5},  # Forex precision
                "limits": {
                    "amount": {"min": 0.01, "max": 1000000},  # Min lot size
                    "price": {"min": 0.00001, "max": 1000000},
                    "cost": {"min": 0.01, "max": 1000000},
                },
                "future": False,
                "option": False,
                "linear": True,
                "inverse": False,
                "contractSize": 1,
                "expiry": None,
                "expiry_date": None,
                "strike": None,
                "underlying": None,
                "settle": None,
                "settleDate": None,
                "listing": None,
                "listed": None,
                "market_type": "spot",
            }

        return markets

    def get_fee(self, symbol, now=None, taker_or_maker="maker"):
        """Get the fee structure for a symbol."""
        return 0.0001 if taker_or_maker == "maker" else 0.0002  # Example forex fees

    def get_historical_ohlcv(self, pair: str, timeframe: str, since: int, limit: int = 1000):
        """Fetch historical OHLCV data from IBKR."""
        symbol = pair.split("/")[0]
        currency = pair.split("/")[1]

        # Define forex contract
        contract = Forex(symbol, currency)

        # Fetch historical data
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=f"{limit} D",
            barSizeSetting=timeframe,
            whatToShow="MIDPOINT",
            useRTH=True,
        )

        # Convert to DataFrame
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
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def get_balances(self):
        """Retrieve account balances."""
        account = self.ib.accountSummary()
        balances = {}
        for item in account:
            if item.tag == "TotalCashValue":
                balances[item.currency] = {
                    "free": float(item.value),
                    "used": 0.0,  # Placeholder
                    "total": float(item.value),
                }
        return balances

    def fetch_positions(self):
        """Fetch open positions."""
        positions = self.ib.positions()
        formatted_positions = []
        for pos in positions:
            formatted_positions.append(
                {
                    "symbol": f"{pos.contract.symbol}/{pos.contract.currency}",
                    "amount": float(pos.position),
                    "side": "long" if pos.position > 0 else "short",
                    "leverage": 1.0,  # Placeholder
                    "contracts": abs(pos.position),
                    "contractSize": 1,
                    "unrealizedPnl": float(pos.unrealizedPNL),
                    "info": pos,
                }
            )
        return formatted_positions

    def close(self):
        """Disconnect from IBKR."""
        self.ib.disconnect()
