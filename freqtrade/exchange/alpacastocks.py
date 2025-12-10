import asyncio
import json
import logging
import math
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.feather as feather
import requests
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetAssetsRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
)
from alpaca.trading.stream import TradingStream
from dateutil.parser import isoparse

from freqtrade.exceptions import OperationalException
from freqtrade.exchange.stockexchange import Stockexchange


logger = logging.getLogger(__name__)

_min_interval = 3.0  # throttle it by 1 call every 2 sec to avoid time bans
_last_request_ts = 0.0


def throttle():
    global _last_request_ts
    now = time.time()
    elapsed = now - _last_request_ts
    if elapsed < _min_interval:
        time.sleep(_min_interval - elapsed)
    _last_request_ts = time.time()


class Alpacastocks(Stockexchange):
    """
    Alpacastocks exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.
    """

    DECIMAL_PLACES = 2
    SIGNIFICANT_DIGITS = 3
    TICK_SIZE = 4
    MAX_DATA_DELAY = pd.Timedelta(minutes=1440)  # Allowed data delay during market hours
    PAIRLIST_FILE = "user_data/data/alpacastocks/alpaca_pairs.json"
    _ft_has_default = {
        "stoploss_on_exchange": False,
        "order_time_in_force": ["GTC", "DAY"],
        "ohlcv_candle_limit": 10000,
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
        "needs_trading_fees": True,
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
        "order_has_fee": True,
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
        exchange_conf = exchange_config if exchange_config else config.get("exchange", {})
        self.id = "alpacastocks"

        self._entry_rate_cache: dict[str, float] = {}
        self._exit_rate_cache: dict[str, float] = {}
        self._cache_lock = threading.Lock()

        self.key = exchange_conf.get("key")
        self.secret = exchange_conf.get("secret")
        if not self.key or not self.secret:
            raise ValueError("API key and secret are required for Alpaca")
        self.dry_run = config.get("dry_run", False)
        if self.dry_run:
            logger.info("Connecting to Alpaca paper trading.")
        else:
            logger.info("Connecting to Alpaca live trading.")
        self.trading_client = TradingClient(self.key, self.secret, paper=self.dry_run)
        self.data_client = StockHistoricalDataClient(self.key, self.secret)
        self.ws_client = None
        self._ws_thread = None
        self._last_market_state = None
        if self._ft_has_default["ws_enabled"]:
            self.setup_websocket()

        if "candle_type_def" not in self.config:
            self.config["candle_type_def"] = "spot"
            logger.info("Set default candle_type_def to 'spot' for alpacastocks")

    @property
    def name(self):
        return "alpacastocks"

    def setup_websocket(self):
        if (
            self.ws_client is not None
            and self._ws_thread is not None
            and self._ws_thread.is_alive()
        ):
            logger.debug("WebSocket client already running.")
            return

        try:
            self.ws_client = TradingStream(self.key, self.secret, paper=self.dry_run)
            # Test authentication by trying to subscribe (this will fail if credentials are invalid)
            self.ws_client.subscribe_trade_updates(self.handle_trade_update)
            self._ws_thread = threading.Thread(target=self.ws_client.run, daemon=True)
            self._ws_thread.start()
        except Exception as e:
            error_msg = str(e).lower()
            if (
                "authenticate" in error_msg
                or "unauthorized" in error_msg
                or "forbidden" in error_msg
            ):
                logger.error(
                    "WebSocket authentication failed - Invalid API credentials. "
                    "Please check your Alpaca API key and secret."
                )
                sys.exit(1)
            logger.warning(f"WebSocket setup failed (non-auth error): {e}")

    async def handle_trade_update(self, trade_update):
        logger.info(f"Trade update received: {trade_update}")

    def create_order(self, pair, ordertype, side, amount, price=None, params=None, **kwargs):
        symbol = pair.split("/", 1)[0]
        params = params or {}
        try:
            side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            if ordertype == "market":
                notional = round(amount, 2)
                if notional < 1.0:
                    notional = 1.0
                    logger.warning(f"Adjusting notional to minimum: ${notional:.2f}")
                order_req = MarketOrderRequest(
                    symbol=symbol,
                    notional=notional,
                    side=side_enum,
                    time_in_force=TimeInForce.DAY,
                )

            elif ordertype == "limit":
                if price is None:
                    price = self.get_rate(f"{symbol}/USD")
                    logger.warning(
                        f"No limit price supplied; using current market price {price:.2f}"
                    )

                available = self._get_available_qty(symbol)
                if amount > available:
                    logger.warning(
                        f"Requested {amount} exceeds available qty ({available}). "
                        "Adjusting to available."
                    )
                # Floor to 6 decimal places to ensure qty <= available
                raw_qty = min(amount, available)
                precision = 6
                qty = math.floor(raw_qty * (10**precision)) / (10**precision)
                if qty <= 0:
                    raise OperationalException(f"Available quantity too small ({available}).")

                limit_price = round(price, 2)
                time_in_force = TimeInForce.DAY if abs(qty - int(qty)) > 1e-6 else TimeInForce.GTC
                logger.info(f"Using time_in_force={time_in_force} for order of {qty} shares.")

                order_req = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side_enum,
                    limit_price=limit_price,
                    time_in_force=time_in_force,
                )
            else:
                raise OperationalException(f"Unsupported order type: {ordertype}")

            alpaca_order = self.trading_client.submit_order(order_req)
            raw_qty = alpaca_order.qty
            raw_filled = alpaca_order.filled_qty
            filled = float(raw_filled or 0)
            qty = float(raw_qty) if raw_qty is not None else filled
            remaining = qty - filled

            return {
                "id": str(alpaca_order.id),
                "symbol": f"{symbol}/USD",
                "type": alpaca_order.order_type.value,
                "side": "buy" if alpaca_order.side == OrderSide.BUY else "sell",
                "price": float(alpaca_order.limit_price or 0),
                "amount": qty,
                "filled": filled,
                "remaining": remaining,
                "status": alpaca_order.status.lower(),
                "cost": filled * float(alpaca_order.filled_avg_price or 0),
                "info": dict(alpaca_order),
            }
        except APIError as e:
            logger.error(f"Failed to create order: {e}")
            raise OperationalException(f"Order rejected by Alpaca: {e}")

    def cancel_order(self, order_id: str):
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return {"id": order_id, "status": "canceled", "info": {}}
        except APIError as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return None

    def cancel_order_with_result(self, order_id: str, pair: str, amount: float) -> dict:
        """
        Cancel an order and return the result in a Freqtrade-compatible format.

        :param order_id: The ID of the order to cancel.
        :param pair: The trading pair (e.g., "AAPL/USD").
        :param amount: The amount of the order to cancel.
        :return: A dictionary with order details post cancellation.
        """
        try:
            # Fetch the order to check its status
            alpaca_order = self.trading_client.get_order_by_id(order_id)
            alpaca_status = alpaca_order.status.lower()
            logger.debug(f"Order {order_id} status before cancellation: {alpaca_status}")

            # Define non-cancellable statuses
            non_cancellable_statuses = [
                "filled",
                "canceled",
                "expired",
                "rejected",
                "done_for_day",
                "stopped",
                "suspended",
            ]
            if alpaca_status in non_cancellable_statuses:
                logger.info(
                    f"Order {order_id} is in {alpaca_status} state and cannot be canceled."
                    f"Returning order details."
                )
                raw_qty = alpaca_order.qty
                raw_filled = alpaca_order.filled_qty
                filled = float(raw_filled or 0)
                qty = float(raw_qty) if raw_qty is not None else filled
                remaining = qty - filled
                order_type = alpaca_order.order_type.value.lower()
                taker_or_maker = "taker" if order_type == "market" else "maker"
                fee_rate = self.get_fee(pair, taker_or_maker=taker_or_maker)
                filled_avg_price = float(alpaca_order.filled_avg_price or 0)
                fee_cost = filled * filled_avg_price * fee_rate
                filled_cost = filled * filled_avg_price
                order_side = "buy" if alpaca_order.side == OrderSide.BUY else "sell"
                result = {
                    "id": str(alpaca_order.id),
                    "symbol": pair,
                    "type": order_type,
                    "side": order_side,
                    "price": float(alpaca_order.limit_price or filled_avg_price or 0),
                    "amount": qty,
                    "filled": filled,
                    "remaining": remaining,
                    "status": alpaca_status,
                    "timestamp": pd.to_datetime(alpaca_order.submitted_at)
                    .tz_convert("UTC")
                    .timestamp()
                    * 1000
                    if alpaca_order.submitted_at
                    else None,
                    "datetime": alpaca_order.submitted_at.isoformat()
                    if alpaca_order.submitted_at
                    else None,
                    "cost": filled_cost,
                    "filled_cost": filled_cost,
                    "fee": {
                        "cost": fee_cost,
                        "currency": "USD",
                        "rate": fee_rate,
                    },
                    "info": dict(alpaca_order),
                }
                return result

            # Attempt to cancel the order if it's in a cancellable state
            self.trading_client.cancel_order_by_id(order_id)
            # Fetch the order again to get updated status
            alpaca_order = self.trading_client.get_order_by_id(order_id)
            raw_qty = alpaca_order.qty
            raw_filled = alpaca_order.filled_qty
            filled = float(raw_filled or 0)
            qty = float(raw_qty) if raw_qty is not None else filled
            remaining = qty - filled
            order_type = alpaca_order.order_type.value.lower()
            taker_or_maker = "taker" if order_type == "market" else "maker"
            fee_rate = self.get_fee(pair, taker_or_maker=taker_or_maker)
            filled_avg_price = float(alpaca_order.filled_avg_price or 0)
            fee_cost = filled * filled_avg_price * fee_rate
            filled_cost = filled * filled_avg_price
            order_side = "buy" if alpaca_order.side == OrderSide.BUY else "sell"
            result = {
                "id": str(alpaca_order.id),
                "symbol": pair,
                "type": order_type,
                "side": order_side,
                "price": float(alpaca_order.limit_price or filled_avg_price or 0),
                "amount": qty,
                "filled": filled,
                "remaining": remaining,
                "status": alpaca_order.status.lower(),
                "timestamp": pd.to_datetime(alpaca_order.submitted_at).tz_convert("UTC").timestamp()
                * 1000
                if alpaca_order.submitted_at
                else None,
                "datetime": alpaca_order.submitted_at.isoformat()
                if alpaca_order.submitted_at
                else None,
                "cost": filled_cost,
                "filled_cost": filled_cost,
                "fee": {
                    "cost": fee_cost,
                    "currency": "USD",
                    "rate": fee_rate,
                },
                "info": dict(alpaca_order),
            }
            logger.info(f"Order {order_id} canceled successfully for pair {pair}")
            return result
        except APIError as e:
            if 'order is already in "filled" state' in str(e):
                logger.info(
                    f"Order {order_id} is already filled,"
                    f"skipping cancellation and returning order details."
                )
                alpaca_order = self.trading_client.get_order_by_id(order_id)
                raw_qty = alpaca_order.qty
                raw_filled = alpaca_order.filled_qty
                filled = float(raw_filled or 0)
                qty = float(raw_qty) if raw_qty is not None else filled
                remaining = qty - filled
                order_type = alpaca_order.order_type.value.lower()
                taker_or_maker = "taker" if order_type == "market" else "maker"
                fee_rate = self.get_fee(pair, taker_or_maker=taker_or_maker)
                filled_avg_price = float(alpaca_order.filled_avg_price or 0)
                fee_cost = filled * filled_avg_price * fee_rate
                filled_cost = filled * filled_avg_price
                order_side = "buy" if alpaca_order.side == OrderSide.BUY else "sell"
                result = {
                    "id": str(alpaca_order.id),
                    "symbol": pair,
                    "type": order_type,
                    "side": order_side,
                    "price": float(alpaca_order.limit_price or filled_avg_price or 0),
                    "amount": qty,
                    "filled": filled,
                    "remaining": remaining,
                    "status": alpaca_order.status.lower(),
                    "timestamp": pd.to_datetime(alpaca_order.submitted_at)
                    .tz_convert("UTC")
                    .timestamp()
                    * 1000
                    if alpaca_order.submitted_at
                    else None,
                    "datetime": alpaca_order.submitted_at.isoformat()
                    if alpaca_order.submitted_at
                    else None,
                    "cost": filled_cost,
                    "filled_cost": filled_cost,
                    "fee": {
                        "cost": fee_cost,
                        "currency": "USD",
                        "rate": fee_rate,
                    },
                    "info": dict(alpaca_order),
                }
                return result
            logger.error(f"Failed to cancel order {order_id} for pair {pair}: {e}")
            raise OperationalException(f"Order cancellation failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error while canceling order {order_id}: {e}")
            raise OperationalException(f"Unexpected error during order cancellation: {e}")

    def fetch_order(self, order_id: str, symbol: str, params: dict | None = None) -> dict | None:
        params = params or {}
        if not isinstance(order_id, str) or not order_id.strip():
            logger.warning(f"fetch_order called with invalid order_id: {order_id}")
            return None
        try:
            alpaca_order = self.trading_client.get_order_by_id(order_id)
            logger.debug(
                f"Fetched order {order_id} for {symbol}: status={alpaca_order.status},"
                f"qty={alpaca_order.qty}, filled_qty={alpaca_order.filled_qty}"
            )
        except Exception as e:
            logger.error(f"Error fetching order with ID {order_id}: {e}")
            return None

        raw_qty = alpaca_order.qty
        raw_filled = alpaca_order.filled_qty
        filled = float(raw_filled or 0)
        qty = float(raw_qty) if raw_qty is not None else filled

        # Ensure remaining is 0 if fully filled
        remaining = max(0.0, qty - filled)

        order_type = alpaca_order.order_type.value.lower()
        taker_or_maker = "taker" if order_type == "market" else "maker"
        fee_rate = self.get_fee(symbol, taker_or_maker=taker_or_maker)
        filled_avg_price = float(alpaca_order.filled_avg_price or 0)
        fee_cost = filled * filled_avg_price * fee_rate
        filled_cost = filled * filled_avg_price
        order_side = "buy" if alpaca_order.side == OrderSide.BUY else "sell"

        # Map Alpaca status to Freqtrade status
        alpaca_status = alpaca_order.status.lower()
        status_map = {
            "new": "open",
            "partially_filled": "open",
            "filled": "closed",  # Mark as closed when fully filled
            "done_for_day": "closed",
            "canceled": "canceled",
            "expired": "canceled",
            "rejected": "canceled",
            "pending_cancel": "canceled",
            "pending_replace": "open",
            "replaced": "open",
            "stopped": "canceled",
            "suspended": "canceled",
        }
        freqtrade_status = status_map.get(alpaca_status, "open")

        # Force close if fully filled
        if filled >= qty:
            freqtrade_status = "closed"
            remaining = 0.0

        freqtrade_order = {
            "id": str(alpaca_order.id),
            "symbol": symbol,
            "type": order_type,
            "side": order_side,
            "price": float(alpaca_order.limit_price or filled_avg_price or 0),
            "amount": qty,
            "filled": filled,
            "remaining": remaining,
            "status": freqtrade_status,
            "timestamp": pd.to_datetime(alpaca_order.submitted_at).tz_convert("UTC").timestamp()
            * 1000
            if alpaca_order.submitted_at
            else None,
            "datetime": alpaca_order.submitted_at.isoformat()
            if alpaca_order.submitted_at
            else None,
            "cost": filled_cost,
            "filled_cost": filled_cost,
            "fee": {
                "cost": fee_cost,
                "currency": "USD",
                "rate": fee_rate,
            },
            "info": dict(alpaca_order),
        }

        logger.debug(f"Returning Freqtrade order: {freqtrade_order}")
        return freqtrade_order

    @property
    def markets(self):
        return self._markets

    def reload_markets(self) -> None:
        self.get_markets(reload=True)

    def get_fee(self, symbol, now=None, taker_or_maker="maker"):
        fee = {"maker": 0.001, "taker": 0.002}
        return fee.get(taker_or_maker, 0.001)

    @property
    def precisionMode(self):
        return self.DECIMAL_PLACES

    @property
    def precision_mode_price(self):
        return self.precisionMode

    def validate_required_startup_candles(self, startup_candle_count, timeframe):
        return True

    def get_proxy_coin(self):
        return "USD"

    def get_precision_price(self, pair):
        return 8

    def get_max_leverage(self, pair, stake_amount):
        return 1

    def get_min_pair_stake_amount(
        self,
        pair: str,
        price: float | None = None,
        amount: float | None = None,
        side: str | None = None,
        is_entry: bool = True,
        **kwargs: Any,
    ) -> float:
        return 1.0

    def get_max_pair_stake_amount(self, *args, **kwargs):
        return 1000000

    def get_pair_base_currency(self, pair: str) -> str:
        return pair.split("/")[0]

    def ws_connection_reset(self):
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
        return market.get("tradable", False) and market.get("spot", False)

    def validate_timeframes(self, timeframes):
        if isinstance(timeframes, str):
            timeframes = [timeframes]
        supported_timeframes = ["1m", "5m", "15m", "1h", "1d"]
        logger.info(f"Validating timeframes: {timeframes}")
        for timeframe in timeframes:
            logger.info(f"Validating timeframe: {timeframe}")
            if timeframe not in supported_timeframes:
                raise ValueError(f"Timeframe '{timeframe}' is not supported by Alpaca.")

    def convert_timeframe(self, timeframe):
        conversion_map = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour", "1d": "1Day"}
        return conversion_map.get(timeframe, timeframe)

    def get_option(self, option, default=None):
        return self._ft_has_default.get(option, default)

    def get_historical_bars(
        self,
        symbols: list[str],
        timeframe: str,
        start: str,
        end: str,
        limit: int = 1000,
        adjustment: str = "raw",
        feed: str = "iex",
        currency: str = "USD",
    ) -> dict:
        """
        Fetch *all* bars from Alpaca Data API between start/end by
        paging through `next_page_token`. Returns a dict with a top-level
        'bars' key mapping each symbol to its list of bars.
        """
        url = "https://data.alpaca.markets/v2/stocks/bars"
        params = {
            "symbols": ",".join(symbols),
            "timeframe": self.convert_timeframe(timeframe),
            "limit": limit,
            "adjustment": adjustment,
            "feed": feed,
            "currency": currency,
            "start": start,
            "end": end,
        }
        headers = {
            "APCA-API-KEY-ID": self.key,
            "APCA-API-SECRET-KEY": self.secret,
        }

        all_bars: dict[str, list] = {s: [] for s in symbols}
        page_token: str | None = None

        while True:
            if page_token:
                params["page_token"] = page_token

            throttle()
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            try:
                resp.raise_for_status()
            except requests.HTTPError as e:
                logger.error(f"HTTP error fetching historical bars: {e}")
                break

            data = resp.json()
            # Accumulate bars for each symbol
            for sym, bars in data.get("bars", {}).items():
                all_bars.setdefault(sym, []).extend(bars)

            # Next page?
            page_token = data.get("next_page_token")
            if not page_token:
                break

        logger.info(
            f"Fetched {sum(len(v) for v in all_bars.values())} bars "
            f"for {symbols} from {start} to {end}"
        )

        # **Wrap** under the 'bars' key so Freqtrade can find it:
        return {"bars": all_bars}

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
            if since_ms:
                start = pd.to_datetime(since_ms, unit="ms", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            if until_ms:
                end = pd.to_datetime(until_ms, unit="ms", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
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
            if (
                not bars
                or "bars" not in bars
                or not isinstance(bars["bars"], dict)
                or not bars["bars"].get(symbol)
            ):
                logger.warning(f"No data available for {pair} {timeframe}")
                return pd.DataFrame()
            data = []
            for bar in bars["bars"].get(symbol, []):
                try:
                    data.append(
                        {
                            "date": pd.to_datetime(bar["t"], utc=True),
                            "open": float(bar["o"]),
                            "high": float(bar["h"]),
                            "low": float(bar["l"]),
                            "close": float(bar["c"]),
                            "volume": float(bar["v"]),
                        }
                    )
                except (KeyError, TypeError) as e:
                    logger.warning(f"Invalid bar data for {symbol}: {e}")
                    continue
            df = pd.DataFrame(data)
            if df.empty:
                logger.warning(f"No valid data retrieved for {pair} {timeframe}")
                return pd.DataFrame()
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values(by="date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            logger.debug(f"DataFrame columns: {df.columns}")
            logger.debug(f"First row: {df.head(1)}")
            return df
        except Exception as e:
            logger.error(f"An error occurred fetching OHLCV for {pair}: {e}")
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
        try:
            if df.empty:
                logger.warning("DataFrame is empty, not saving to Feather file.")
                return
            feather.write_feather(df, file_path)
            logger.info(f"DataFrame saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save DataFrame to Feather file: {str(e)}")

    def klines(self, pair, timeframe=None, *, candle_type="spot", **kwargs) -> pd.DataFrame:
        try:
            pair_str = pair
            if isinstance(pair, tuple) and len(pair) == 3:
                pair_str, timeframe, _candle_type = pair  # Prefix with underscore
            else:
                if timeframe is None:
                    raise ValueError("timeframe must be provided if not using a tuple.")
            ohlcv = self.get_historic_ohlcv(pair_str, timeframe)
            return ohlcv.copy()
        except Exception as e:
            logger.error(f"Error fetching klines for {pair}: {str(e)}")
            return pd.DataFrame()

    def refresh_latest_ohlcv(self, pairs=None, timeframe="1h", **kwargs):
        """
        Refresh the latest OHLCV data for the given pairs.
        If the market is closed, sleep until 5 minutes before it opens.
        """
        is_open, time_until_open = self.is_market_open()
        if not is_open:
            if time_until_open > 300:  # More than 5 minutes until open
                sleep_time = time_until_open - 300
                logger.info(
                    f"Market is closed, sleeping for {sleep_time:.1f} seconds "
                    "until 5 minutes before market opens. "
                    "Reminder, JPX (Japan) opens at 8:00 PM "
                    "and the SSE (China) and HKEX open at 9:30 PM"
                )
                time.sleep(sleep_time)
            else:
                logger.info(
                    "Market is closed, but opening in less than 5 minutes. "
                    "Proceeding to fetch data."
                )

        # Now fetch the data
        latest_ohlcv = {}
        for raw_pair in pairs or []:
            if isinstance(raw_pair, tuple) and len(raw_pair) == 3:
                pair_symbol = raw_pair[0]
            else:
                pair_symbol = raw_pair
            try:
                ohlcv_data = self.get_historic_ohlcv(
                    pair=pair_symbol,
                    timeframe=timeframe,
                    limit=1,
                )
                if not ohlcv_data.empty:
                    latest_ohlcv[pair_symbol] = ohlcv_data
                else:
                    logger.warning(
                        f"No OHLCV data retrieved for {pair_symbol} on timeframe {timeframe}"
                    )
            except Exception as e:
                logger.error(f"Error fetching latest OHLCV for {pair_symbol}: {str(e)}")
        return latest_ohlcv

    def get_balances(self):
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
        try:
            throttle()
            positions = self.trading_client.get_all_positions()
            return [self._format_position(pos) for pos in positions]
        except Exception as e:
            logger.error(f"Position fetch error: {e}")
            return []

    def _format_position(self, position):
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

    def is_market_open(self) -> tuple[bool, float]:
        """
        Check if the NYSE market is currently open and return the time until it opens if closed.
        NYSE is open Monday through Friday, 9:30 AM to 4:00 PM Eastern Time.

        Returns:
            tuple: (is_open, time_until_open)
                - is_open: True if market is open, False otherwise
                - time_until_open: seconds until the market opens (0 if open)
        """
        try:
            clock = self.trading_client.get_clock()
            current_time_utc = pd.Timestamp.now(tz="UTC")
            self._last_market_state = clock.is_open
            if not clock.is_open:
                time_until_open = (clock.next_open - current_time_utc).total_seconds()
                hours, remainder = divmod(time_until_open, 3600)
                minutes, _ = divmod(remainder, 60)
                next_open_formatted = clock.next_open.strftime("%Y-%m-%d %H:%M UTC")
                logger.info(
                    f"Market is closed. Next open: {next_open_formatted} "
                    f"({int(hours)}h {int(minutes)}m)"
                )
                return False, time_until_open
            else:
                return True, 0.0
        except Exception as e:
            logger.error(f"Failed to retrieve market clock: {e}")
            return self._last_market_state if self._last_market_state is not None else False, 0.0

    def get_rate(self, pair: str, side: str | None = None, *args, **kwargs) -> float:
        try:
            df = self.klines(pair, timeframe="1m", candle_type="spot")
            if df.empty:
                raise ValueError(f"No price data available for {pair}")
            return float(df["close"].iat[-1])
        except Exception as e:
            logger.error(f"Failed to fetch rate for {pair}: {e}")
            return 0.0

    def get_funding_fees(self, pair: str, amount: float, **kwargs) -> float:
        return 0.0

    def check_order_canceled_empty(self, order: dict) -> bool:
        if not order:
            return True
        status = order.get("status", "").lower()
        return status in ["canceled", "cancelled", "not-found"]

    def fetch_order_or_stoploss_order(
        self, order_id: str, symbol: str, params: dict | None = None, **kwargs
    ) -> dict | None:
        return self.fetch_order(order_id, symbol, params)

    def order_has_fee(self, order: dict) -> bool:
        return True

    def get_trades_for_order(
        self, order_id: str, symbol: str, params: dict | None = None, **kwargs
    ) -> list:
        return []

    def get_order_id_conditional(self, order: dict) -> str:
        return order.get("id", "")

    def handle_order_fee(self, trade, order_obj, order):
        order_id = order_obj.get("id")
        logger.debug(
            f"Alpaca handle_order_fee called for trade {trade.id}, processing order ID: {order_id}"
        )
        fee_info = order_obj.get("fee")
        if not fee_info:
            logger.debug(f"Order {order_id}: No 'fee' info found in order_obj.")
            return None
        fee_cost = fee_info.get("cost", 0.0)
        fee_currency = fee_info.get("currency", self.config.get("stake_currency", "USD"))
        stake_currency = self.config.get("stake_currency", "USD")
        is_open_fee = order_id is not None and str(trade.open_order_id) == str(order_id)
        is_close_fee = order_id is not None and str(trade.close_order_id) == str(order_id)
        if is_open_fee:
            logger.info(
                f"Applying OPEN fee for trade {trade.id}, order {order_id}: "
                f"Cost={fee_cost}, Currency={fee_currency}"
            )
            trade.fee_open = True
            trade.fee_open_cost = fee_cost
            trade.fee_open_currency = fee_currency
            trade.fee_open_stake_currency = stake_currency
        elif is_close_fee:
            logger.info(
                f"Applying CLOSE fee for trade {trade.id}, order {order_id}: "
                f"Cost={fee_cost}, Currency={fee_currency}"
            )
            trade.fee_close = True
            trade.fee_close_cost = fee_cost
            trade.fee_close_currency = fee_currency
            trade.fee_close_stake_currency = stake_currency
        else:
            logger.debug(
                f"Order {order_id} does not match open or close order for trade {trade.id}."
            )
        return None

    def extract_cost_curr_rate(self, *args, **kwargs) -> tuple[float, str, float]:
        logger.debug("Alpacastocks.extract_cost_curr_rate called.")
        logger.debug(f"  args: {args}")
        logger.debug(f"  kwargs: {kwargs}")
        cost = 0.0
        currency = self.config.get("stake_currency", "USD")
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
                logger.warning(f"Could not convert args[2] '{arg_cost}' to float for cost.")
                cost = 0.0
        logger.debug(
            f"extract_cost_curr_rate returning cost={cost}, currency={currency}, rate={rate}"
        )
        return cost, currency, rate

    def get_markets(self, reload=False, params=None, tradable_only=False, active_only=False):
        pairlist_file_path = Path(self.PAIRLIST_FILE)
        reload_needed = self._should_reload_markets(pairlist_file_path, reload)

        if reload_needed:
            self._load_markets_from_api(pairlist_file_path, tradable_only, active_only)
        else:
            self._load_markets_from_file(pairlist_file_path, tradable_only, active_only)

        return self._markets

    def _should_reload_markets(self, file_path: Path, reload_flag: bool) -> bool:
        """
        Determine if markets should be reloaded from the API.
        """
        if reload_flag or not hasattr(self, "_markets"):
            if file_path.exists():
                file_age = time.time() - file_path.stat().st_mtime
                if file_age > 86400:
                    logger.info("Pairlist file is older than 24 hours. Reloading from Alpaca.")
                    return True
                return False
            return True
        return False

    def _load_markets_from_file(self, file_path: Path, tradable_only: bool, active_only: bool):
        """
        Load markets from a local JSON file.
        """
        try:
            with file_path.open("r") as f:
                self._markets = json.load(f)
            logger.debug("Loaded market pairs from file.")
        except Exception as e:
            logger.error(f"Failed to load market pairs from file: {e}")
            # Fallback to API reload with original filter flags
            self._load_markets_from_api(file_path, tradable_only, active_only)

    def _load_markets_from_api(self, file_path: Path, tradable_only: bool, active_only: bool):
        """
        Fetch and process market data from Alpaca API.
        """
        logger.info("Refreshing market pairs from Alpaca API.")
        try:
            search_params = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)
            throttle()
            assets = self.trading_client.get_all_assets(search_params)
            assets_dict = [dict(item) for item in assets]
            self._markets = self._process_assets(assets_dict, tradable_only, active_only)

            with file_path.open("w") as f:
                json.dump(self._markets, f, default=str)
            logger.info("Saved market pairs to file.")

        except APIError as e:
            error_message = str(e).lower()
            # Handle both "forbidden" and "unauthorized" authentication errors
            if "forbidden" in error_message or "unauthorized" in error_message:
                logger.error(
                    "Authentication failed - Invalid API credentials. "
                    "Please check your Alpaca API key and secret."
                )
                sys.exit(1)
            logger.error(f"Error fetching market data from Alpaca: {e}")
            # Re-raise other API errors so they bubble up
            raise
        except Exception as e:
            logger.error(f"Unexpected error while fetching markets: {e}")
            # Re-raise unexpected errors
            raise

    def _process_assets(self, assets_dict: list, tradable_only: bool, active_only: bool):
        """
        Process raw asset data into a market dictionary.
        """
        markets = {}
        for asset in assets_dict:
            if not self._is_asset_included(asset, tradable_only, active_only):
                continue

            pair = f"{asset['symbol']}/USD"
            markets[pair] = {
                "id": pair,
                "symbol": pair,
                "base": asset["symbol"],
                "quote": "USD",
                "spot": True,
                "tradable": asset["tradable"],
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
        return markets

    def _is_asset_included(self, asset: dict, tradable_only: bool, active_only: bool) -> bool:
        """
        Determine if an asset should be included in the market list.
        """
        if tradable_only and not asset.get("tradable", False):
            return False
        if active_only and asset.get("status") != "active":
            return False
        return True

    def fetch_balance(self, params: dict | None = None) -> dict:
        """
        Fetch account balances for Freqtrade strategies and FreqUI.

        Returns a dict of currencies with fields 'free', 'used' and 'total'.
        """
        try:
            # Reuse your existing helper
            balances = self.get_balances()
            return balances
        except Exception as e:
            logger.error(f"Error fetching balance via Alpaca: {e}")
            return {}

    def fetch_ticker(self, pair: str, params: dict | None = None) -> dict:
        """
        Pulls latest quote + trade from Alpaca for `pair` (e.g. "MSFT/USD").
        First tries to use any datetime attributes on the Pydantic models,
        then ISO strings, then finally now() if all else fails.
        """
        symbol = pair.split("/", 1)[0]

        def normalize(obj):
            # If it's a Pydantic model with attrs, keep both forms
            data = obj.dict() if hasattr(obj, "dict") else obj
            return obj, data.get("data", data) if isinstance(data, dict) else data

        try:
            # 1) Fetch quote and trade
            throttle()
            quote_obj, qdata = normalize(
                self.data_client.get_stock_latest_quote(
                    StockLatestQuoteRequest(symbol_or_symbols=symbol)
                )
            )
            throttle()
            trade_obj, tdata = normalize(
                self.data_client.get_stock_latest_trade(
                    StockLatestTradeRequest(symbol_or_symbols=symbol)
                )
            )

            # If wrapped in list, grab first
            if isinstance(qdata, list):
                qdata = qdata[0]
            if isinstance(tdata, list):
                tdata = tdata[0]

            # 2) Timestamp resolution
            ts_dt = None
            # a) Model attribute (datetime)
            for attr in ("timestamp", "timestamp_utc"):
                val = getattr(quote_obj, attr, None) or getattr(trade_obj, attr, None)
                if isinstance(val, datetime):
                    ts_dt = val
                    break

            # b) ISO string in dict
            if ts_dt is None:
                ts_str = (
                    qdata.get("timestamp")
                    or qdata.get("timestamp_utc")
                    or tdata.get("timestamp")
                    or tdata.get("timestamp_utc")
                )
                if ts_str:
                    ts_dt = isoparse(ts_str)

            # c) Fallback to UTC now
            if ts_dt is None:
                ts_dt = datetime.now(timezone.utc)  # noqa: UP017
                logger.warning(
                    f"No timestamp in quote/trade for {symbol}; using now()={ts_dt.isoformat()}"
                )

            ts_ms = int(ts_dt.timestamp() * 1000)

            # 3) Assemble CCXT-style ticker
            return {
                "symbol": pair,
                "timestamp": ts_ms,
                "datetime": ts_dt.isoformat(),
                "high": None,
                "low": None,
                "open": None,
                "close": float(tdata.get("price", 0.0)),
                "bid": float(qdata.get("bid_price", 0.0)),
                "bidVolume": None,
                "ask": float(qdata.get("ask_price", 0.0)),
                "askVolume": None,
                "info": {
                    "quote": qdata,
                    "trade": tdata,
                },
            }

        except Exception:
            logger.exception(f"Error fetching ticker for {symbol} via Alpaca")
            # bubble up real API/connectivity errors
            raise

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ) -> list[list]:
        try:
            bars = self.get_historic_ohlcv(
                pair=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit,
                params=params or {},
            )
            ohlcv = [
                [
                    int(bar["date"].timestamp() * 1000),
                    float(bar["open"]),
                    float(bar["high"]),
                    float(bar["low"]),
                    float(bar["close"]),
                    float(bar["volume"]),
                ]
                for bar in bars.to_dict("records")
            ]
            return ohlcv
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol} via Alpaca: {e}")
            return []

    def fetch_open_orders(
        self,
        symbol: str | None = None,
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ) -> list[dict]:
        try:
            query: dict = {"status": QueryOrderStatus.OPEN}
            if limit:
                query["limit"] = limit
            if params:
                query.update(params)

            request = GetOrdersRequest(**query)
            alpaca_orders = self.trading_client.get_orders(request)

            ft_orders = []
            for o in alpaca_orders:
                created_ts = int(o.submitted_at.timestamp() * 1000)
                qty = float(o.qty or 0)
                filled = float(o.filled_qty or 0)
                remaining = qty - filled
                avg_fill_price = float(o.filled_avg_price or 0)

                ft_orders.append(
                    {
                        "id": str(o.id),
                        "clientOrderId": getattr(o, "client_order_id", None),
                        "timestamp": created_ts,
                        "datetime": self.iso8601(created_ts),
                        "symbol": f"{o.symbol}/USD",
                        "type": o.order_type.value.lower(),
                        "side": o.side.value.lower(),
                        "price": float(o.limit_price) if o.limit_price else None,
                        "amount": qty,
                        "filled": filled,
                        "remaining": remaining,
                        "status": o.status.value.lower(),
                        "cost": filled * avg_fill_price,
                        "info": dict(o),
                    }
                )

            if symbol:
                base = symbol.split("/", 1)[0]
                ft_orders = [o for o in ft_orders if o["symbol"].startswith(base)]

            return ft_orders

        except Exception as e:
            logger.error(f"Error fetching open orders via Alpaca: {e}")
            return []

    def iso8601(self, timestamp: int) -> str:
        return pd.to_datetime(timestamp, unit="ms").isoformat()

    def fetch_trades(
        self,
        pair: str,
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ) -> list[dict]:
        """
        Pulls historical trades from Alpaca and returns them in CCXT/Freqtrade format,
        so FreqUI can render live and backtested order-flow.
        """
        symbol = pair.split("/", 1)[0]
        data_client: StockHistoricalDataClient = (
            self.data_client
        )  # your instantiated Alpaca data client

        # Convert since (ms) to ISO8601, if provided
        start: str | None = None
        if since:
            start = datetime.fromtimestamp(since / 1000, tz=timezone.utc).isoformat()  # noqa: UP017

        # Default limit if not set
        max_trades = limit or 1000

        # Fetch trades (Alpaca returns a .data list of Trade objects)
        throttle()
        api_resp = data_client.get_stock_trades(
            symbol_or_symbols=symbol, start=start, limit=max_trades, **(params or {})
        )

        ccxt_trades = []
        for t in api_resp.data:
            # Alpaca Trade.timestamp is ISO str; parse to ms
            ts = int(isoparse(t.timestamp).timestamp() * 1000)
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()  # noqa: UP017
            ccxt_trades.append(
                {
                    "id": t.trade_id,
                    "timestamp": ts,
                    "datetime": dt,
                    "symbol": symbol,
                    "side": t.taker_side.value.lower(),  # "buy" or "sell"
                    "price": float(t.price),
                    "amount": float(t.size),
                }
            )

        return ccxt_trades

    async def watch_ohlcv(
        self,
        pair: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ):
        """
        Streams live OHLCV bars from Alpaca into Freqtrade/FreqUI.
        Yields lists: [ timestamp(ms), open, high, low, close, volume ].
        """
        # 1) Symbol part
        symbol = pair.split("/", 1)[0]

        # 2) Convert Freqtrade TF (e.g. "1m", "5m") to Alpaca bar_timeframe (e.g. "1Min", "5Min")
        minutes = int(timeframe.rstrip("m"))
        alpaca_tf = f"{minutes}Min"

        # 3) Create the stream client
        stream = TradingStream(self.key, self.secret, paper=self.dry_run)

        # 4) Use an asyncio.Queue to hand bars over to the generator
        queue: asyncio.Queue = asyncio.Queue()

        async def _on_bar(bar):
            ts = int(bar.start * 1000)
            await queue.put(
                [
                    ts,
                    float(bar.open),
                    float(bar.high),
                    float(bar.low),
                    float(bar.close),
                    float(bar.volume),
                ]
            )

        # 5) Subscribe to bar updates
        stream.subscribe_bars(_on_bar, symbol, bar_timeframe=alpaca_tf)

        # 6) Kick off the streaming loop in the background
        self._stream_task = asyncio.create_task(stream._run_forever())

        # 7) Yield bars as they arrive
        while True:
            ohlcv = await queue.get()
            yield ohlcv

    async def watch_ticker(self, pairs: list[str], params: dict | None = None):
        """
        Streams live bid/ask updates from Alpaca into Freqtrade/FreqUI.
        Yields dicts: {
            'symbol': 'MSFT',
            'timestamp': 1234567890123,      # ms
            'bid': 250.12,
            'ask': 250.15
        }
        """
        # 1) Instantiate the stream client
        stream = TradingStream(self.key, self.secret, paper=self.dry_run)

        # 2) Prepare an asyncio queue for quotes
        queue: asyncio.Queue[dict] = asyncio.Queue()

        # 3) Quote callback
        async def _on_quote(quote):
            # Alpaca's quote.timestamp is ISO8601
            ts_ms = int(isoparse(quote.timestamp).timestamp() * 1000)
            await queue.put(
                {
                    "symbol": quote.symbol,
                    "timestamp": ts_ms,
                    "bid": float(quote.bid_price),
                    "ask": float(quote.ask_price),
                }
            )

        # 4) Subscribe to quotes for each pair
        for pair in pairs:
            symbol = pair.split("/", 1)[0]
            stream.subscribe_quotes(_on_quote, symbol)

        # 5) Launch the websocket loop
        self._stream_task = asyncio.create_task(stream._run_forever())

        # 6) Yield quotes continuously
        while True:
            quote_update = await queue.get()
            yield quote_update

    def get_rates(self, pair: str, refresh: bool, is_short: bool) -> tuple[float, float]:
        """
        Returns entry and exit rates for any symbol, compatible with Freqtrade UI.
        Caches rates when `refresh=False`.
        """
        entry_rate = None
        exit_rate = None

        # 1) Try cache first
        if not refresh:
            with self._cache_lock:
                entry_rate = self._entry_rate_cache.get(pair)
                exit_rate = self._exit_rate_cache.get(pair)
            if entry_rate is not None:
                logger.debug(f"Using cached entry rate for {pair}.")
            if exit_rate is not None:
                logger.debug(f"Using cached exit rate for {pair}.")

        # 2) On cache miss or when refresh requested, fetch fresh bid/ask
        if entry_rate is None or exit_rate is None:
            ticker = self.fetch_ticker(pair)
            bid = ticker["bid"]
            ask = ticker["ask"]

            # Long entry buys at ask, short entry sells at bid
            entry_rate = entry_rate if entry_rate is not None else (ask if not is_short else bid)
            # Long exit sells at bid, short exit buys at ask
            exit_rate = exit_rate if exit_rate is not None else (bid if not is_short else ask)

            # 3) Cache them
            with self._cache_lock:
                self._entry_rate_cache[pair] = entry_rate
                self._exit_rate_cache[pair] = exit_rate

        return entry_rate, exit_rate

    def get_conversion_rate(self, base: str, quote: str) -> float:
        """
        Returns the mid market conversion rate between two symbols,
        with full exception safety so FreqUI never sees a missing rate.
        """
        # 0) trivial case
        if base == quote:
            return 1.0

        # 1) try direct pair
        pair = f"{base}/{quote}"
        try:
            ticker = self.fetch_ticker(pair)
            return (ticker["bid"] + ticker["ask"]) / 2
        except Exception:
            logger.debug(f"Direct conversion fetch failed for {pair}, trying inverse.")

        # 2) try inverse pair
        inv_pair = f"{quote}/{base}"
        try:
            inv = self.fetch_ticker(inv_pair)
            mid = (inv["bid"] + inv["ask"]) / 2
            return 1.0 / mid
        except Exception:
            logger.warning(
                f"Inverse conversion fetch also failed for {inv_pair}. Falling back to 1.0"
            )

        # 3) ultimate fallback
        return 1.0

    async def watch_trades(
        self,
        pair: str,
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ):
        """
        Streams live trade ticks from Alpaca into Freqtrade/FreqUI.
        Yields dicts: {
            'id':       '123456789',
            'timestamp': 1234567890123,  # ms
            'datetime': '2025-05-16T12:34:56.789Z',
            'symbol':   'MSFT',
            'side':     'buy' or 'sell',
            'price':    250.12,
            'amount':   100
        }
        """
        symbol = pair.split("/", 1)[0]
        stream = TradingStream(self.key, self.secret, paper=self.dry_run)
        q: asyncio.Queue = asyncio.Queue()

        async def _on_trade(trade):
            ts = int(isoparse(trade.timestamp).timestamp() * 1000)
            # Alpaca Trade.timestamp is ISO str; parse to ms
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()  # noqa: UP017
            await q.put(
                {
                    "id": trade.trade_id,
                    "timestamp": ts,
                    "datetime": dt,
                    "symbol": symbol,
                    "side": trade.taker_side.value.lower(),
                    "price": float(trade.price),
                    "amount": float(trade.size),
                }
            )

        # subscribe to live trades for this symbol
        stream.subscribe_trades(_on_trade, symbol)

        # kick off the websocket
        self._stream_task = asyncio.create_task(stream._run_forever())

        # yield trades as they arrive
        while True:
            yield await q.get()

    def validate_config(self, config: dict) -> None:
        """
        Validate the exchange configuration.
        This method is required by Freqtrade and called during bot initialization.
        """
        logger.info("Validating Alpaca configuration...")

        # Check for required API credentials
        if not self.key or not self.secret:
            raise OperationalException(
                "Alpaca API key and secret are required in the configuration."
            )

        # Test authentication by making a simple API call
        try:
            # This will fail immediately if credentials are invalid
            self.trading_client.get_account()
            logger.debug("Alpaca authentication successful")
        except APIError as e:
            error_message = str(e).lower()
            if "forbidden" in error_message or "unauthorized" in error_message:
                logger.error(
                    "Authentication failed - Invalid API credentials. "
                    "Please check your Alpaca API key and secret."
                )
                sys.exit(1)
            # Re-raise other API errors
            raise

        # Validate dry_run mode compatibility
        if not self.dry_run:
            logger.warning(
                "Live trading mode is enabled. "
                "Ensure you have sufficient funds and understand the risks."
            )

        # Validate timeframes if specified in config
        timeframes = config.get("timeframes", [])
        if timeframes:
            self.validate_timeframes(timeframes)

        logger.info("Alpaca configuration validation completed successfully.")

    def _get_available_qty(self, symbol: str) -> float:
        try:
            positions = self.trading_client.get_all_positions()
            for pos in positions:
                if pos.symbol == symbol:
                    return float(pos.qty)
        except Exception as e:
            logger.error(f"Failed to fetch available qty for {symbol}: {e}")
        return 0.0

    def validate_trading_mode_and_margin_mode(
        self, trading_mode, margin_mode, allow_none_margin_mode: bool = False, **kwargs
    ) -> None:
        """
        Validate that the requested trading and margin modes are supported by the exchange.
        Alpaca stocks implementation currently focuses on spot trading.
        """
        # The logic remains the same:
        if trading_mode and str(trading_mode).lower() != "spot":
            raise OperationalException(
                f"Alpaca Stocks exchange does not support {trading_mode} trading mode."
            )

        # In Alpaca, margin is account-level rather than symbol-level in the crypto sense.
        # Check against 'none' or 'cross'. The 'allow_none_margin_mode' is handled by the caller.
        if margin_mode and str(margin_mode).lower() not in ("none", "cross"):
            raise OperationalException(
                f"Alpaca Stocks exchange does not support {margin_mode} margin mode."
            )

    def ohlcv_candle_limit(self, timeframe: str, candle_type: str = "trade") -> int:
        """
        Calculates the maximum number of candles that can be requested at once.
        This limit is defined by the Alpaca API for historical data requests.
        """
        # 1000 is the common limit for Alpaca bar data requests.
        return 1000
