# alpacastocks.py
# This file contains the implementation for the Alpaca Stocks exchange integration in Freqtrade.
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pyarrow.feather as feather
import requests
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass
from alpaca.trading.requests import GetAssetsRequest, LimitOrderRequest, MarketOrderRequest
from alpaca.trading.stream import TradingStream

from freqtrade.exchange.stockexchange import Stockexchange


logger = logging.getLogger(__name__)


class Alpacastocks(Stockexchange):
    """
    Alpacastocks exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.
    """

    DECIMAL_PLACES = 2
    SIGNIFICANT_DIGITS = 3
    TICK_SIZE = 4
    MAX_DATA_DELAY = pd.Timedelta(minutes=5)  # Allowed data delay during market hours

    PAIRLIST_FILE = "user_data/data/alpacastocks/alpaca_pairs.json"

    _ft_has_default = {
        "stoploss_on_exchange": False,
        "order_time_in_force": ["GTC"],
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
        "ws_enabled": False,
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

        exchange_conf = exchange_config if exchange_config else config.get("exchange", {})

        # Retrieve API key and secret
        self.key = exchange_conf.get("key")
        self.secret = exchange_conf.get("secret")

        if not self.key or not self.secret:
            raise ValueError("API key and secret are required for Alpaca")

        # Determine base URL based on dry_run setting
        self.dry_run = config.get("dry_run", False)

        if self.dry_run:
            logger.info("Connecting to Alpaca paper trading.")
        else:
            logger.info("Connecting to Alpaca live trading.")

        # Initialize TradingClient
        self.trading_client = TradingClient(self.key, self.secret, paper=self.dry_run)

        # Initialize WebSocket client
        self.ws_client = None
        if self._ft_has_default["ws_enabled"]:
            self.setup_websocket()

    @property
    def name(self):
        return "alpacastocks"

    def get_markets(self, reload=False, params=None, tradable_only=False, active_only=False):
        """
        Retrieve a list of markets (trading pairs) available on the exchange.

        :param reload: If True, reload the markets from the exchange.
        :param params: Additional parameters for fetching markets.
        :param tradable_only: If True, return only tradable markets.
        :param active_only: If True, return only active markets.
        """

        pairlist_file_path = Path(self.PAIRLIST_FILE)

        try:
            if not hasattr(self, "_markets") or reload:
                if pairlist_file_path.exists():
                    # Load markets from file
                    with pairlist_file_path.open() as f:
                        self._markets = json.load(f)
                else:
                    try:
                        search_params = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)
                        assets = self.trading_client.get_all_assets(search_params)
                        assets_dict = [dict(item) for item in assets]
                        self._markets = {}
                        for asset in assets_dict:
                            if tradable_only and not asset["tradable"]:
                                continue
                            if active_only and asset["status"] != "active":
                                continue
                            pair = f"{asset['symbol']}/USD"
                            self._markets[pair] = {
                                "id": pair,
                                "symbol": pair,
                                "base": asset["symbol"],
                                "quote": "USD",
                                "spot": True,
                                "margin": False,
                                "active": asset["status"] == "active",
                                "maker": 0.001,
                                "taker": 0.002,
                                "info": asset,
                                "precision": {"amount": 8, "price": 8},
                                "limits": {
                                    "amount": {"min": 0.001, "max": 1000000},
                                    "price": {"min": 0.01, "max": 1000000},
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
                            logger.debug(f"Added pair: {pair}")

                        # logger.info(f"Retrieved markets: {self._markets}")

                        with pairlist_file_path.open("w") as f:
                            json.dump(self._markets, f, default=str)
                        logger.info("Saved market pairs to file.")

                    except Exception as e:
                        error_message = str(e).lower()
                        if "forbidden" in error_message:
                            logger.error(
                                "Authentication failed - Invalid API credentials. "
                                "Please check your Alpaca API key and secret. "
                                "Make sure they are correct and have proper permissions."
                            )
                            sys.exit(1)

            return self._markets
        except Exception as e:
            logger.error(f"Failed to retrieve markets: {str(e)}")
            return {}

    def setup_websocket(self):
        """Initialize WebSocket for real-time updates."""
        self.ws_client = TradingStream(self.key, self.secret, paper=self.dry_run)
        self.ws_client.subscribe_trade_updates(self.handle_trade_update)
        self.ws_client.run()

    def handle_trade_update(self, trade_update):
        """Handle real-time trade updates from WebSocket."""
        logger.info(f"Trade update received: {trade_update}")

    def create_order(
        self,
        pair: str,
        ordertype: str,
        side: str,
        amount: float,
        price: float | None = None,
        params=None,
    ):
        """Create an order on Alpaca."""
        symbol = pair.split("/")[0]
        if params is None:
            params = {}

        try:
            if ordertype == "market":
                # Use notional for market orders to specify USD amount
                order_data = MarketOrderRequest(symbol=symbol, notional=amount, side=side.lower())
            elif ordertype == "limit":
                # For limit orders, amount is in shares (qty)
                order_data = LimitOrderRequest(
                    symbol=symbol, qty=amount, side=side.lower(), limit_price=price
                )
            else:
                raise ValueError(f"Unsupported order type: {ordertype}")

            order = self.trading_client.submit_order(order_data)
            return order.id  # Return order ID for tracking
        except APIError as e:
            logger.error(f"Failed to create order: {e}")
            return None

    def cancel_order(self, order_id: str):
        """Cancel an order on Alpaca."""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Order {order_id} canceled successfully.")
        except APIError as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")

    def fetch_order(self, order_id: str):
        """Fetch order details from Alpaca."""
        try:
            order = self.trading_client.get_order_by_id(order_id)
            return order
        except APIError as e:
            logger.error(f"Failed to fetch order {order_id}: {e}")
            return None

    @property
    def markets(self):
        return self._markets

    def reload_markets(self) -> None:
        """Reload the markets data, forcing a fresh fetch from Alpaca."""
        self.get_markets(reload=True)

    def get_fee(self, symbol, now=None, taker_or_maker="maker"):
        # Replace with the actual fee calculation for the symbol
        fee = {
            "maker": 0.001,
            "taker": 0.001,
        }
        return fee.get(taker_or_maker, 0.001)

    @property
    def precisionMode(self):
        return self.DECIMAL_PLACES

    @property
    def precision_mode_price(self):
        return self.precisionMode

    def validate_required_startup_candles(self, startup_candle_count, timeframe):
        pass

    def get_proxy_coin(self):
        return "USD"

    def get_precision_price(self, pair):
        return 8

    def get_max_leverage(self, pair, stake_amount):
        return 1

    def get_min_pair_stake_amount(self, *args, **kwargs):
        return 1

    def get_max_pair_stake_amount(self, *args, **kwargs):
        return 1000000

    def get_pair_base_currency(self, pair: str) -> str:
        return pair.split("/")[0]

    def ws_connection_reset(self):
        # No-op implementation for Alpaca
        pass

    def get_pair_quote_currency(self, pair):
        return pair.split("/")[1]

    def get_contract_size(self, pair):
        return 1

    def get_precision_amount(self, pair):
        return 8

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
        # Implementation here
        return None

    def update_liquidation_prices(self, trade, row):
        try:
            liquidation_price = self.get_liquidation_price(
                pair=trade.pair,
                current_price=row["close"],
                order_side=trade.order_side,
                amount=trade.amount,
                order_type=None,
                open_rate=row["open"],
                is_short=None,
                stake_amount=None,
                leverage=None,
                wallet_balance=None,
            )
            if liquidation_price is not None:
                trade.liquidation_price = liquidation_price
        except Exception as e:
            logger.error(f"Failed to update liquidation price: {str(e)}")

    def market_is_tradable(self, market):
        """
        Check if the market is tradable.
        :param market: Market dictionary
        :return: True if the market is tradable, False otherwise
        """
        return market["active"] and market["spot"]

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
                raise ValueError(f"Timeframe '{timeframe}' is not supported by Alpaca.")

    def convert_timeframe(self, timeframe):
        """
        Convert Freqtrade timeframe to Alpaca timeframe.

        :param timeframe: Freqtrade timeframe.
        :return: Alpaca timeframe.
        """
        conversion_map = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour", "1d": "1Day"}
        return conversion_map.get(timeframe, timeframe)

    def get_option(self, option, default=None):
        """
        Get an option value from the exchange configuration.

        :param option: The option key.
        :param default: The default value if the option is not found.
        :return: The option value.
        """
        return self._ft_has_default.get(option, default)

    def get_historical_bars(
        self,
        symbols,
        timeframe,
        start,
        end,
        limit=1000,
        adjustment="raw",
        feed="iex",
        currency="USD",
    ):
        """
        Fetch historical bars data from Alpaca API.

        :param symbols: List of stock symbols.
        :param timeframe: Timeframe for the bars.
        :param start: Start date in ISO 8601 format.
        :param end: End date in ISO 8601 format.
        :param limit: Maximum number of data points to return.
        :param adjustment: Corporate action adjustment for the stocks.
        :param feed: Source feed of the data.
        :param currency: Currency of all prices.
        """

        url = "https://data.alpaca.markets/v2/stocks/bars"
        params = {
            "symbols": ",".join(symbols),
            "timeframe": self.convert_timeframe(timeframe),
            "start": start,
            "end": end,
            "limit": limit,
            "adjustment": adjustment,
            "feed": feed,
            "currency": currency,
        }

        # Include start and end only if they are not None
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        headers = {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret}

        logger.debug(f"API Request: GET {url}")
        logger.debug(f"API Headers: {headers}")
        logger.debug(f"API Parameters: {params}")

        # Add timeout parameter to prevent hanging
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code != 200:
            logger.error(f"Failed to fetch historical bars: {response.text}")
            return {}

        data = response.json()
        logger.debug(f"API Response: {data}")
        return data

    def get_historic_ohlcv(
        self,
        pair,
        timeframe,
        since=None,
        limit=10000,
        params=None,
        since_ms=None,
        is_new_pair=True,
        candle_type="spot",
        until_ms=None,
    ):
        try:
            if params is None:
                params = {}

            logger.debug(f"Request parameters: {params}")
            logger.debug(f"since_ms: {since_ms}, until_ms: {until_ms}")

            symbol = pair.split("/")[0]

            # Convert since and until_ms to ISO 8601 format
            if since_ms:
                start = pd.to_datetime(since_ms, unit="ms", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                start = pd.to_datetime("2024-01-01T00:00:00Z", utc=True).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )  # Default start date

            if until_ms:
                end = pd.to_datetime(until_ms, unit="ms", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                # Use current UTC time (or current_time minus a small buffer) as the end time.
                end = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

            bars = self.get_historical_bars(
                [symbol],
                timeframe,
                start,
                end,
                limit,
                params.get("adjustment", "raw"),
                params.get("feed", "iex"),
                params.get("currency", "USD"),
            )

            if not bars or "bars" not in bars or not bars["bars"].get(symbol):
                logger.warning(f"No data available for {pair} {timeframe}")
                return pd.DataFrame()

            data = []
            for bar in bars["bars"][symbol]:
                # Remove unit='s' as 't' is in ISO format
                data.append(
                    {
                        "date": pd.to_datetime(bar["t"], utc=True),
                        "open": bar["o"],
                        "high": bar["h"],
                        "low": bar["l"],
                        "close": bar["c"],
                        "volume": bar["v"],
                    }
                )

            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values(by="date", inplace=True)
            df.reset_index(drop=True, inplace=True)

            logger.debug(f"DataFrame columns: {df.columns}")
            logger.debug(f"First row: {df.head(1)}")

            if df.empty:
                logger.warning(f"No data retrieved for {pair} {timeframe}")
                return pd.DataFrame()

            return df
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return pd.DataFrame()

    def _download_pair_history(self, data, DATETIME_PRINT_FORMAT):
        if data.empty:
            logger.error("DataFrame is empty")
            return "None"
        elif "date" not in data.columns:
            logger.error(f"DataFrame does not contain 'date' column: {data.columns}")
            return "None"
        else:
            return f"{data.iloc[0]['date']:{DATETIME_PRINT_FORMAT}}"

    def save_to_feather(self, df, file_path):
        """
        Save the DataFrame to a Feather file.

        :param df: DataFrame to save.
        :param file_path: Path to save the Feather file.
        """
        try:
            if df.empty:
                logger.warning("DataFrame is empty, not saving to Feather file.")
                return
            feather.write_feather(df, file_path)
            logger.info(f"DataFrame saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save DataFrame to Feather file: {str(e)}")

    def klines(self, pair, timeframe=None, *, candle_type="spot", **kwargs) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a pair and timeframe.

        :param pair: The trading pair, or a tuple containing (pair, timeframe, candle_type).
        :param timeframe: The timeframe (e.g., '1m', '1h', '1d').
        :param candle_type: The type of candles ('spot' for standard candles).
        :return: DataFrame containing OHLCV data.
        """
        try:
            # Initialize pair_str to hold the string pair name
            pair_str = pair
            # Check if pair is provided as a tuple (pair, timeframe, candle_type)
            if isinstance(pair, tuple) and len(pair) == 3:
                # Unpack tuple, but use pair_str for the string pair name
                pair_str, timeframe, candle_type = pair
            else:
                # Ensure timeframe is provided when not using a tuple
                if timeframe is None:
                    raise ValueError("timeframe must be provided if not using a tuple.")
            # Proceed to fetch OHLCV data, now using pair_str
            ohlcv = self.get_historic_ohlcv(pair_str, timeframe)
            return ohlcv.copy()
        except Exception as e:
            logger.error(f"Error fetching klines for {pair}: {str(e)}")
            return pd.DataFrame()

    def refresh_latest_ohlcv(self, pairs=None, timeframe="1h", **kwargs):
        """
        Fetches the latest OHLCV data for the given pairs from Alpaca.
        """
        if not self.is_market_open():
            # If the market is closed, this message will have already been logged by is_market_open.
            # Optionally, you can decide not to proceed with fetching data.
            return {}

        # Continue with fetching data when the market is open
        latest_ohlcv = {}
        current_time = pd.Timestamp.now(tz="UTC")

        for raw_pair in pairs or []:
            if isinstance(raw_pair, tuple) and len(raw_pair) == 3:
                pair_symbol = raw_pair[0]
            else:
                pair_symbol = raw_pair

            try:
                start_time = current_time - pd.Timedelta(minutes=5)
                limit = 1  # Get only the latest candle
                ohlcv_data = self.get_historic_ohlcv(
                    pair=pair_symbol,
                    timeframe=timeframe,
                    since=int(start_time.timestamp() * 1000),
                    until_ms=int(current_time.timestamp() * 1000),
                    limit=limit,
                )
                if not ohlcv_data.empty:
                    latest_ohlcv[pair_symbol] = ohlcv_data
                else:
                    logger.warning(f"No OHLCV data found for pair {pair_symbol} at {current_time}.")
            except Exception as e:
                logger.error(f"Error fetching latest OHLCV for {pair_symbol}: {str(e)}")
                continue

        return latest_ohlcv

    def get_balances(self):
        """Retrieve account balances (already implemented)"""
        try:
            account = self.trading_client.get_account()
            return {
                "USD": {
                    "free": float(account.cash),
                    "used": float(account.cash) - float(account.buying_power),
                    "total": float(account.equity),
                }
            }
        except Exception as e:
            logger.error(f"Balance fetch error: {e}")
            return {}

    def fetch_positions(self, symbols=None, params=None):
        """New positions fetch method"""
        try:
            positions = self.trading_client.get_all_positions()
            return [self._format_position(pos) for pos in positions]
        except Exception as e:
            logger.error(f"Position fetch error: {e}")
            return []

    def _format_position(self, position):
        """Helper to format Alpaca position"""
        qty = float(position.qty)
        return {
            "symbol": f"{position.symbol}/USD",
            "amount": abs(qty),
            "side": "long" if qty > 0 else "short",
            "leverage": 1.0,
            "contracts": abs(qty),
            "contractSize": 1,
            "unrealizedPnl": float(position.unrealized_pl),
            "info": dict(position),
        }

    def is_market_open(self) -> bool:
        """
        Check if the market is open using Alpaca's clock API.

        :return: True if the market is open, False otherwise.
        """
        try:
            clock = self.trading_client.get_clock()
            if not clock.is_open:
                logger.info("Market is closed")
            return clock.is_open
        except Exception as e:
            logger.error(f"Failed to retrieve market clock: {e}")
            # If there is an error, assume the market is closed
            return False
