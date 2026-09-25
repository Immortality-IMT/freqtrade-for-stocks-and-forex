"""Native Alpaca US-equity exchange adapter for the Freqtrade stock fork.

This module intentionally bypasses CCXT.  It translates between the
Freqtrade/CCXT-shaped interface used by the fork and Alpaca's native
trading + market-data APIs.

Supported trading model:
    * US equities only
    * spot / long-only
    * market and limit orders
    * DAY / GTC / OPG / CLS / IOC / FOK TIFs supported by Alpaca

Important design rules:
    * Freqtrade's ``dry_run`` never submits an Alpaca order.
    * Stock balances are exposed as normal spot wallets (USD + shares).
    * Stock market data uses ``StockDataStream``; order updates use
      ``TradingStream``.
    * Missing stock candles are NOT fabricated across market closures.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import requests
from alpaca.common.enums import Sort
from alpaca.common.exceptions import APIError
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.live.stock import StockDataStream
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockLatestTradeRequest,
    StockTradesRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    AssetClass,
    AssetStatus,
    OrderSide,
    QueryOrderStatus,
    TimeInForce,
)
from alpaca.trading.models import Asset
from alpaca.trading.requests import (
    GetAssetsRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
)
from alpaca.trading.stream import TradingStream

from freqtrade.enums import CandleType
from freqtrade.exceptions import (
    ConfigurationError,
    ExchangeError,
    InsufficientFundsError,
    InvalidOrderException,
    OperationalException,
    PricingError,
    TemporaryError,
)
from freqtrade.exchange.stockexchange import Stockexchange


logger = logging.getLogger(__name__)


class Alpacastocks(Stockexchange):
    """Freqtrade adapter for Alpaca US equities."""

    # Freqtrade/CCXT precision mode constants.  This adapter uses TICK_SIZE
    # because Alpaca expresses quantity/price constraints as increments.
    DECIMAL_PLACES = 2
    SIGNIFICANT_DIGITS = 3
    TICK_SIZE = 4
    PAIRLIST_FILE = "user_data/data/alpacastocks/alpaca_pairs.json"

    # This is deliberately kept close to Freqtrade's capability vocabulary,
    # but the adapter does not use CCXT itself.
    _ft_has_default: dict[str, Any] = {
        "stoploss_on_exchange": False,
        "order_time_in_force": ["GTC", "DAY", "IOC", "FOK", "OPG", "CLS"],
        "ohlcv_candle_limit": 10000,
        "ohlcv_has_history": True,
        "ohlcv_partial_candle": True,
        "ohlcv_require_since": False,
        "ohlcv_volume_currency": "base",
        "tickers_have_quoteVolume": False,
        "tickers_have_percentage": False,
        "tickers_have_bid_ask": True,
        "tickers_have_price": True,
        "trades_limit": 1000,
        "trades_pagination": "time",
        "trades_pagination_arg": "since",
        "trades_has_history": True,
        "market_has_ticker": True,
        "market_has_ohlcv": True,
        "order_has_status": True,
        "order_has_type": True,
        "order_has_side": True,
        "order_has_time_in_force": True,
        "order_has_price": True,
        "order_has_amount": True,
        "order_has_cost": True,
        # Alpaca trading API does not put an execution fee into the normal
        # Order response for the simple stock orders this adapter supports.
        "order_has_fee": False,
        "order_has_slippage": False,
        "order_has_filled": True,
        "order_has_remaining": True,
        "order_statuses": {
            "new": "open",
            "pending_new": "open",
            "accepted": "open",
            "accepted_for_bidding": "open",
            "partially_filled": "open",
            "pending_cancel": "open",
            "pending_replace": "open",
            "calculated": "open",
            "held": "open",
            "filled": "closed",
            "canceled": "canceled",
            "cancelled": "canceled",
            "expired": "canceled",
            "rejected": "rejected",
            "stopped": "canceled",
            "suspended": "open",
            # done_for_day is terminal. It may contain a partial fill, so use
            # Freqtrade's canceled state while preserving filled/remaining.
            "done_for_day": "canceled",
            # replaced is normally followed by a new order id and therefore
            # remains open until the replacement is observed.
            "replaced": "canceled",
        },
        "ws_enabled": True,
        "ws_auto_reconnect": True,
        "ws_reconnect_interval": 30,
        "balance_includes_unrealized_pnl": False,
        "contract_size": 1.0,
    }

    _METHOD_MAP = {
        "fetchTicker": "fetch_ticker",
        "fetchTickers": "fetch_tickers",
        "fetchOHLCV": "fetch_ohlcv",
        "fetchTrades": "fetch_trades",
        "fetchOpenOrders": "fetch_open_orders",
        "fetchOrders": "fetch_orders",
        "fetchOrder": "fetch_order",
        "fetchBalance": "fetch_balance",
        "fetchPositions": "fetch_positions",
        "createOrder": "create_order",
        "createMarketOrder": "create_order",
        "createLimitOrder": "create_order",
        "cancelOrder": "cancel_order",
        "cancelOrderWithResult": "cancel_order_with_result",
        "watchTicker": "watch_ticker",
        "watchOHLCV": "watch_ohlcv",
        "watchTrades": "watch_trades",
    }

    _TF_MAP = {
        "1m": TimeFrame(1, TimeFrameUnit.Minute),
        "5m": TimeFrame(5, TimeFrameUnit.Minute),
        "15m": TimeFrame(15, TimeFrameUnit.Minute),
        "1h": TimeFrame(1, TimeFrameUnit.Hour),
        "1d": TimeFrame(1, TimeFrameUnit.Day),
    }

    _TIF_MAP = {
        "day": TimeInForce.DAY,
        "gtc": TimeInForce.GTC,
        "ioc": TimeInForce.IOC,
        "fok": TimeInForce.FOK,
        "opg": TimeInForce.OPG,
        "cls": TimeInForce.CLS,
    }

    _TERMINAL_STATUSES = {
        "filled",
        "canceled",
        "expired",
        "rejected",
        "stopped",
        "suspended",
        "done_for_day",
        "replaced",
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

        self.config = config
        exchange_conf = (
            exchange_config if exchange_config is not None else config.get("exchange", {})
        )
        self.exchange_config = exchange_conf
        self.id = "alpacastocks"
        self.dry_run = bool(config.get("dry_run", False))

        self.key = exchange_conf.get("key")
        self.secret = exchange_conf.get("secret")
        if not self.key or not self.secret:
            raise ConfigurationError("Alpaca API key and secret are required for alpacastocks.")

        self.data_feed = self._parse_data_feed(
            exchange_conf.get("data_feed", exchange_conf.get("feed", "iex"))
        )
        self.adjustment = self._parse_adjustment(exchange_conf.get("adjustment", "raw"))
        self.extended_hours = bool(exchange_conf.get("extended_hours", False))
        self.pairlist_file = Path(exchange_conf.get("pairlist_file", self.PAIRLIST_FILE))

        # The stock adapter does not use CCXT's caches, so maintain the few
        # pieces of state Freqtrade normally gets from Exchange.
        self._ft_has = dict(self._ft_has_default)
        self._markets: dict[str, dict[str, Any]] = {}
        self._assets: dict[str, Asset] = {}
        self._assets_lock = threading.RLock()
        self._klines: dict[tuple[str, str, CandleType], pd.DataFrame] = {}
        self._klines_lock = threading.RLock()

        # Rate cache: (pair, is_short) -> (entry, exit, monotonic_time).
        self._rate_cache: dict[tuple[str, bool], tuple[float, float, float]] = {}
        self._rate_cache_ttl = float(exchange_conf.get("rate_cache_ttl", 5.0))
        self._rate_cache_lock = threading.RLock()

        # Dry-run orders never reach Alpaca.  Keep a small local order book so
        # fetch_order/cancel_order continue to work with Freqtrade's state machine.
        self._dry_orders: dict[str, dict[str, Any]] = {}
        self._dry_order_lock = threading.RLock()

        # Trading websocket is for account/order updates only.
        self._trading_stream: TradingStream | None = None
        self._trading_stream_thread: threading.Thread | None = None
        self._stream_stop_events: list[threading.Event] = []
        self._stream_threads: list[threading.Thread] = []
        self._stream_clients: list[Any] = []
        self._stream_lock = threading.RLock()
        self._order_updates: dict[str, Any] = {}

        self._market_open_cache: tuple[bool, float, float] | None = None

        # Alpaca's trading client works against either paper or live endpoints.
        self.trading_client = TradingClient(self.key, self.secret, paper=self.dry_run)
        self.data_client = StockHistoricalDataClient(self.key, self.secret)
        self._trading_api_base_url = (
            "https://paper-api.alpaca.markets" if self.dry_run else "https://api.alpaca.markets"
        )
        self._http_session = requests.Session()

        if "candle_type_def" not in self.config:
            self.config["candle_type_def"] = "spot"

        logger.info(
            "Initialized alpacastocks (%s trading, feed=%s, adjustment=%s)",
            "dry-run" if self.dry_run else "live",
            self.data_feed.value,
            self.adjustment.value,
        )

        # Initial market load is intentionally done for both dry-run and live:
        # Alpaca remains the source of market metadata/data in dry-run; dry-run
        # only suppresses trading orders.
        self.reload_markets(force=True)

        if validate:
            self.validate_config(config)

        if self._ft_has_default["ws_enabled"]:
            self.setup_websocket()

    @property
    def name(self) -> str:
        return "alpacastocks"

    @property
    def markets(self) -> dict[str, dict[str, Any]]:
        return self._markets

    @property
    def timeframes(self) -> list[str]:
        return list(self._TF_MAP)

    @property
    def margin_mode(self):
        """Freqtrade margin-mode contract for this spot-only adapter.

        Alpaca US equities are handled as spot/long-only here, so there is no
        crypto-style cross/isolated margin mode to expose. Returning ``None``
        also makes the Freqtrade liquidation-price helper skip margin logic.
        """
        return None

    @property
    def precisionMode(self) -> int:
        # Alpaca's stock constraints are increments (tick sizes), not a fixed
        # number of decimal places. Freqtrade's TICK_SIZE mode preserves an
        # increment such as 0.000000001 instead of converting it to int(0).
        return self.TICK_SIZE

    @property
    def precision_mode_price(self) -> int:
        return self.TICK_SIZE

    def get_option(self, option: str, default: Any = None) -> Any:
        return self._ft_has.get(option, default)

    def exchange_has(self, method: str) -> bool:
        """Map Freqtrade/CCXT capability names to native adapter methods."""
        mapped = self._METHOD_MAP.get(method, method)
        if (
            method in {"watchTicker", "watchOHLCV", "watchTrades"}
            and not self._ft_has["ws_enabled"]
        ):
            return False
        return hasattr(self, mapped)

    # ------------------------------------------------------------------
    # Basic exchange/spot contract
    # ------------------------------------------------------------------

    def get_proxy_coin(self) -> str:
        return "USD"

    def get_pair_base_currency(self, pair: str) -> str:
        return self._split_pair(pair)[0]

    def get_pair_quote_currency(self, pair: str) -> str:
        return self._split_pair(pair)[1]

    def get_contract_size(self, pair: str) -> float:
        return 1.0

    def get_liquidation_price(
        self,
        pair: str,
        amount: float,
        current_price: float | None = None,
        order_side: str | None = None,
        order_type: str | None = None,
        open_rate: float | None = None,
        is_short: bool | None = None,
        stake_amount: float | None = None,
        leverage: float | None = None,
        wallet_balance: float | None = None,
        open_trades: list[Any] | None = None,
    ) -> None:
        """Return no liquidation price for spot equities.

        Alpaca stocks are represented by this adapter as Freqtrade spot
        positions with no futures-style liquidation price.  The fork's
        liquidation helper still calls this method for spot trades, so the
        method must exist and return ``None`` implicitly.
        """

    def _contracts_to_amount(self, pair: str, num_contracts: float) -> float:
        return float(num_contracts)

    def _amount_to_contracts(self, pair: str, amount: float) -> float:
        return float(amount)

    def amount_to_contract_precision(self, pair: str, amount: float) -> float:
        return self.amount_to_precision(pair, amount)

    def balance_includes_unrealized_pnl(self) -> bool:
        # get_balances() reports cash, not account equity.
        return False

    def market_is_tradable(self, market: dict[str, Any]) -> bool:
        return bool(market.get("spot") and market.get("tradable") and market.get("active"))

    def validate_trading_mode_and_margin_mode(
        self,
        trading_mode: Any,
        margin_mode: Any,
        allow_none_margin_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        if trading_mode and str(trading_mode).lower() != "spot":
            raise OperationalException("alpacastocks supports spot/long-only trading only.")
        if margin_mode not in (None, "", "none"):
            raise OperationalException("alpacastocks does not expose crypto margin modes.")

    def validate_required_startup_candles(self, startup_candle_count: int, timeframe: str) -> bool:
        return timeframe in self._TF_MAP and startup_candle_count >= 0

    # ------------------------------------------------------------------
    # Markets / asset metadata
    # ------------------------------------------------------------------

    def reload_markets(self, force: bool = True, *, load_leverage_tiers: bool = False) -> None:
        """Reload Alpaca asset metadata, honoring explicit force requests."""
        del load_leverage_tiers
        self.get_markets(reload=force)

    def get_markets(
        self,
        reload: bool = False,
        params: dict | None = None,
        tradable_only: bool = False,
        active_only: bool = False,
    ) -> dict[str, dict[str, Any]]:
        del params

        if not reload and self._markets:
            return self._filtered_markets(tradable_only, active_only)

        if not reload and self.pairlist_file.exists():
            age = time.time() - self.pairlist_file.stat().st_mtime
            if age < 86400:
                try:
                    payload = json.loads(self.pairlist_file.read_text())
                    if isinstance(payload, dict):
                        self._markets = payload
                        return self._filtered_markets(tradable_only, active_only)
                except (OSError, ValueError, TypeError):
                    logger.warning(
                        "Ignoring unreadable Alpaca pairlist file %s", self.pairlist_file
                    )

        request = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)
        assets = self._api_call(
            lambda: self.trading_client.get_all_assets(request), "get_all_assets"
        )

        with self._assets_lock:
            self._assets = {asset.symbol.upper(): asset for asset in assets}
            self._markets = self._build_markets(assets)

        try:
            self.pairlist_file.parent.mkdir(parents=True, exist_ok=True)
            self.pairlist_file.write_text(json.dumps(self._markets, indent=2, default=str))
        except OSError as exc:
            logger.warning("Unable to save Alpaca pairlist cache: %s", exc)

        return self._filtered_markets(tradable_only, active_only)

    def _filtered_markets(
        self, tradable_only: bool, active_only: bool
    ) -> dict[str, dict[str, Any]]:
        markets = self._markets
        if tradable_only:
            markets = {k: v for k, v in markets.items() if v.get("tradable")}
        if active_only:
            markets = {k: v for k, v in markets.items() if v.get("active")}
        return markets

    def _build_markets(self, assets: list[Asset]) -> dict[str, dict[str, Any]]:
        markets: dict[str, dict[str, Any]] = {}
        for asset in assets:
            symbol = asset.symbol.upper()
            pair = f"{symbol}/USD"
            price_increment = float(asset.price_increment or 0.01)
            if asset.fractionable:
                # Fractional stock orders support quantities down to 9 decimal
                # places. The current equity Asset API may leave
                # min_trade_increment/min_order_size unset, and those fields
                # historically described crypto-only constraints.
                raw_increment = self._as_float(getattr(asset, "min_trade_increment", None), 0.0)
                qty_increment = raw_increment if 0.0 < raw_increment < 1.0 else 1e-9
                min_qty = qty_increment
                min_cost = 1.0
            else:
                raw_increment = self._as_float(getattr(asset, "min_trade_increment", None), 1.0)
                qty_increment = max(1.0, raw_increment)
                min_qty = max(1.0, self._as_float(getattr(asset, "min_order_size", None), 1.0))
                min_cost = None
            markets[pair] = {
                "id": pair,
                "symbol": pair,
                "base": symbol,
                "quote": "USD",
                "spot": True,
                "tradable": bool(asset.tradable),
                "margin": bool(asset.marginable),
                "active": self._enum_value(asset.status) == AssetStatus.ACTIVE.value,
                "maker": 0.0,
                "taker": 0.0,
                "info": self._model_to_dict(asset),
                "precision": {
                    "amount": qty_increment,
                    "price": price_increment,
                },
                "limits": {
                    "amount": {"min": min_qty, "max": None},
                    "price": {"min": price_increment, "max": None},
                    "cost": {"min": min_cost, "max": None},
                },
                "future": False,
                "swap": False,
                "option": False,
                "linear": False,
                "inverse": False,
                "contractSize": 1.0,
                "market_type": "spot",
            }
        return markets

    def _get_asset(self, symbol: str) -> Asset:
        symbol = symbol.upper()
        asset = self._assets.get(symbol)
        if asset is not None:
            return asset
        try:
            asset = self.trading_client.get_asset(symbol)
        except APIError as exc:
            raise OperationalException(f"Unable to get Alpaca asset {symbol}: {exc}") from exc
        self._assets[symbol] = asset
        return asset

    # ------------------------------------------------------------------
    # Precision / stake limits
    # ------------------------------------------------------------------

    def get_precision_price(self, pair: str) -> float:
        return float(self.markets.get(pair, {}).get("precision", {}).get("price", 0.01))

    def get_precision_amount(self, pair: str) -> float:
        return float(self.markets.get(pair, {}).get("precision", {}).get("amount", 1.0))

    def amount_to_precision(self, pair: str, amount: float) -> float:
        increment = self.get_precision_amount(pair)
        if increment <= 0:
            return float(amount)
        return self._round_down_increment(float(amount), increment)

    def price_to_precision(
        self, pair: str, price: float, *, rounding_mode: int | None = None
    ) -> float:
        del rounding_mode
        increment = self.get_precision_price(pair)
        if increment <= 0:
            return float(price)
        return self._round_down_increment(float(price), increment)

    def get_min_pair_stake_amount(
        self,
        pair: str,
        price: float | None = None,
        stoploss: float = 0.0,
        leverage: float = 1.0,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        del stoploss, leverage, args, kwargs
        p = float(price or self.get_rate(pair, side="entry", is_short=False, refresh=True))
        limits = self.markets.get(pair, {}).get("limits", {})
        min_amount = float((limits.get("amount") or {}).get("min") or 1.0)
        min_cost = float((limits.get("cost") or {}).get("min") or 0.0)
        return max(min_amount * p, min_cost)

    def get_max_pair_stake_amount(
        self,
        pair: str,
        price: float | None = None,
        leverage: float = 1.0,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        del pair, price, leverage, args, kwargs
        return float("inf")

    def get_max_leverage(self, pair: str, stake_amount: float | None = None) -> float:
        del pair, stake_amount
        return 1.0

    # ------------------------------------------------------------------
    # Balances / positions
    # ------------------------------------------------------------------

    def get_balances(self) -> dict[str, dict[str, float]]:
        """Return normal spot balances: USD cash plus each stock's shares."""
        if self.dry_run and self.config.get("runmode") not in ("live",):
            return {}

        account = self._api_call(self.trading_client.get_account, "get_account")
        cash = self._as_float(account.cash, 0.0)
        balances: dict[str, dict[str, float]] = {
            "USD": {
                "free": cash,
                "used": 0.0,
                "total": cash,
            }
        }

        positions = self._api_call(self.trading_client.get_all_positions, "get_all_positions")
        for pos in positions:
            qty = abs(self._as_float(pos.qty, 0.0))
            qty_available = abs(self._as_float(pos.qty_available, qty))
            if qty <= 0:
                continue
            used = max(0.0, qty - qty_available)
            balances[pos.symbol.upper()] = {
                "free": qty_available,
                "used": used,
                "total": qty,
            }
        return balances

    def fetch_balance(self, params: dict | None = None) -> dict[str, dict[str, float]]:
        del params
        return self.get_balances()

    def fetch_positions(
        self, symbols: list[str] | None = None, params: dict | None = None
    ) -> list[dict[str, Any]]:
        del params
        positions = self._api_call(self.trading_client.get_all_positions, "get_all_positions")
        allowed = {s.split("/", 1)[0].upper() for s in symbols} if symbols else None
        result: list[dict[str, Any]] = []
        for position in positions:
            if allowed is not None and position.symbol.upper() not in allowed:
                continue
            qty = self._as_float(position.qty, 0.0)
            if qty == 0:
                continue
            side = "short" if qty < 0 else "long"
            quantity = abs(qty)
            market_value = abs(self._as_float(position.market_value, 0.0))
            result.append(
                {
                    "symbol": f"{position.symbol.upper()}/USD",
                    "amount": quantity,
                    "contracts": quantity,
                    "contractSize": 1.0,
                    "side": side,
                    # Kept populated because the fork's Wallets parser expects
                    # the futures-shaped key.  It is not used as collateral for
                    # spot trading because trading_mode remains spot.
                    "collateral": market_value,
                    "initialMargin": market_value,
                    "leverage": 1.0,
                    "unrealizedPnl": self._as_float(position.unrealized_pl, 0.0),
                    "currentPrice": self._as_float(position.current_price, 0.0),
                    "avgEntryPrice": self._as_float(position.avg_entry_price, 0.0),
                    "qtyAvailable": self._as_float(position.qty_available, quantity),
                    "info": self._model_to_dict(position),
                }
            )
        return result

    # ------------------------------------------------------------------
    # Pricing / ticker
    # ------------------------------------------------------------------

    def fetch_ticker(self, pair: str, params: dict | None = None) -> dict[str, Any]:
        del params
        symbol = self._pair_symbol(pair)

        quote = self._api_call(
            lambda: self.data_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=self.data_feed)
            ),
            f"latest quote {symbol}",
        )
        trade = self._api_call(
            lambda: self.data_client.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=symbol, feed=self.data_feed)
            ),
            f"latest trade {symbol}",
        )

        quote_obj = quote.get(symbol) if hasattr(quote, "get") else None
        trade_obj = trade.get(symbol) if hasattr(trade, "get") else None

        bid = self._as_float(getattr(quote_obj, "bid_price", None), 0.0)
        ask = self._as_float(getattr(quote_obj, "ask_price", None), 0.0)
        bid_size = self._as_float(getattr(quote_obj, "bid_size", None), 0.0)
        ask_size = self._as_float(getattr(quote_obj, "ask_size", None), 0.0)
        last = self._as_float(getattr(trade_obj, "price", None), 0.0)
        timestamp = getattr(trade_obj, "timestamp", None) or getattr(quote_obj, "timestamp", None)
        ts_ms = (
            self._datetime_to_ms(timestamp) if timestamp is not None else int(time.time() * 1000)
        )

        if last <= 0 and bid > 0 and ask > 0:
            last = (bid + ask) / 2.0
        if last <= 0:
            raise PricingError(f"Alpaca returned no usable price for {pair}.")

        return {
            "symbol": pair,
            "timestamp": ts_ms,
            "datetime": datetime.fromtimestamp(ts_ms / 1000, UTC).isoformat(),
            "last": last,
            "close": last,
            "bid": bid or None,
            "ask": ask or None,
            "bidVolume": bid_size or None,
            "askVolume": ask_size or None,
            "open": None,
            "high": None,
            "low": None,
            "baseVolume": None,
            "quoteVolume": None,
            "percentage": None,
            "info": {
                "quote": self._model_to_dict(quote_obj),
                "trade": self._model_to_dict(trade_obj),
            },
        }

    def fetch_tickers(
        self, symbols: list[str] | None = None, params: dict | None = None
    ) -> dict[str, dict[str, Any]]:
        del params
        pairs = symbols or list(self._markets)
        return {pair: self.fetch_ticker(pair) for pair in pairs}

    def get_rate(
        self,
        pair: str,
        side: str | None = None,
        is_short: bool = False,
        refresh: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        del args, kwargs
        key = (pair, bool(is_short))
        now = time.monotonic()
        if not refresh:
            with self._rate_cache_lock:
                cached = self._rate_cache.get(key)
            if cached and now - cached[2] <= self._rate_cache_ttl:
                return cached[0] if side == "entry" else cached[1]

        ticker = self.fetch_ticker(pair)
        bid = float(ticker.get("bid") or 0.0)
        ask = float(ticker.get("ask") or 0.0)
        last = float(ticker.get("last") or 0.0)

        if side == "entry":
            rate = (bid if is_short else ask) or last
        elif side == "exit":
            rate = (ask if is_short else bid) or last
        else:
            rate = last or (bid + ask) / 2.0

        if rate <= 0:
            raise PricingError(f"Could not determine a usable rate for {pair}.")

        with self._rate_cache_lock:
            previous = self._rate_cache.get(key)
            entry = rate if side == "entry" or previous is None else previous[0]
            exit_rate = rate if side == "exit" or previous is None else previous[1]
            if side is None:
                entry = rate
                exit_rate = rate
            self._rate_cache[key] = (entry, exit_rate, now)

        return rate

    def get_rates(self, pair: str, refresh: bool, is_short: bool) -> tuple[float, float]:
        key = (pair, bool(is_short))
        now = time.monotonic()
        with self._rate_cache_lock:
            cached = self._rate_cache.get(key)
        if not refresh and cached and now - cached[2] <= self._rate_cache_ttl:
            return cached[0], cached[1]

        ticker = self.fetch_ticker(pair)
        bid = float(ticker.get("bid") or 0.0)
        ask = float(ticker.get("ask") or 0.0)
        last = float(ticker.get("last") or 0.0)
        if bid <= 0 and ask <= 0 and last <= 0:
            raise PricingError(f"Could not determine rates for {pair}.")
        entry = (bid if is_short else ask) or last
        exit_rate = (ask if is_short else bid) or last
        with self._rate_cache_lock:
            self._rate_cache[key] = (entry, exit_rate, now)
        return entry, exit_rate

    def get_conversion_rate(self, base: str, quote: str) -> float:
        """Return the market value of one unit of ``base`` in ``quote``.

        Stocks in this adapter are USD-quoted pairs, so converting e.g.
        TSLA -> USD is simply the current TSLA/USD market price.  This is
        used by Freqtrade RPC/UI for fiat valuation of stock-denominated
        amounts.
        """
        base = base.upper()
        quote = quote.upper()
        if base == quote:
            return 1.0
        if quote == "USD":
            pair = f"{base}/USD"
            ticker = self.fetch_ticker(pair)
            bid = float(ticker.get("bid") or 0.0)
            ask = float(ticker.get("ask") or 0.0)
            last = float(ticker.get("last") or 0.0)
            if bid > 0 and ask > 0:
                return (bid + ask) / 2.0
            if last > 0:
                return last
            raise PricingError(f"No current USD conversion rate available for {base}.")
        raise PricingError(f"alpacastocks cannot convert {base} to {quote}.")

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def create_order(
        self,
        pair: str,
        ordertype: str,
        side: str,
        amount: float,
        rate: float | None = None,
        leverage: float = 1.0,
        reduceOnly: bool = False,
        time_in_force: str = "DAY",
        initial_order: bool = True,
        params: dict | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a native Alpaca stock order with Freqtrade semantics."""
        del leverage, reduceOnly, initial_order
        params = params or {}
        symbol = self._pair_symbol(pair)
        side_text = self._validate_order_side(side, amount)
        order_kind = ordertype.lower()
        tif = self._parse_tif(time_in_force)
        extended_hours = bool(params.get("extended_hours", self.extended_hours))
        client_order_id = params.get("client_order_id") or kwargs.get("client_order_id")
        asset = self._get_asset(symbol)

        if self.dry_run:
            return self._create_dry_order(
                pair=pair,
                ordertype=order_kind,
                side=side_text,
                amount=amount,
                rate=rate,
                tif=tif,
                extended_hours=extended_hours,
            )

        self._validate_asset_for_order(asset, symbol)
        tif, extended_hours = self._prepare_order_terms(
            asset, order_kind, amount, tif, extended_hours, symbol
        )
        qty = self._normalize_order_quantity(asset, amount, order_kind)
        self._validate_order_notional(asset, pair, qty, rate, side_text, order_kind)
        self._validate_sell_quantity(symbol, side_text, qty)
        request = self._build_order_request(
            asset=asset,
            symbol=symbol,
            order_kind=order_kind,
            side=side_text,
            qty=qty,
            rate=rate,
            tif=tif,
            extended_hours=extended_hours,
            client_order_id=client_order_id,
        )
        submitted = self._api_call(
            lambda: self.trading_client.submit_order(request),
            f"submit {order_kind} {side_text} {symbol}",
            order_exception=True,
        )
        return self._normalize_order(submitted, pair, requested_rate=rate)

    @staticmethod
    def _validate_order_side(side: str, amount: float) -> str:
        side_text = side.lower()
        if side_text not in {"buy", "sell"}:
            raise InvalidOrderException(f"Unsupported stock order side: {side}")
        if amount <= 0:
            raise InvalidOrderException(f"Order amount must be positive: {amount}")
        return side_text

    @staticmethod
    def _validate_asset_for_order(asset: Asset, symbol: str) -> None:
        if not asset.tradable or asset.status != AssetStatus.ACTIVE:
            raise InvalidOrderException(f"Alpaca asset {symbol} is not tradable/active.")

    def _prepare_order_terms(
        self,
        asset: Asset,
        order_kind: str,
        amount: float,
        tif: TimeInForce,
        extended_hours: bool,
        symbol: str,
    ) -> tuple[TimeInForce, bool]:
        if (
            order_kind == "limit"
            and asset.fractionable
            and not self._is_whole_quantity(amount)
            and tif != TimeInForce.DAY
        ):
            logger.warning(
                "Alpaca fractional limit order for %s requires DAY time-in-force; "
                "normalizing requested TIF %s to DAY (requested qty=%s).",
                symbol,
                self._enum_value(tif),
                amount,
            )
            tif = TimeInForce.DAY

        if (
            extended_hours
            and asset.fractionable
            and not self._is_whole_quantity(amount)
            and not bool(getattr(asset, "fractional_eh_enabled", False))
        ):
            raise InvalidOrderException(
                f"Fractional extended-hours trading is not enabled for {symbol}."
            )
        return tif, extended_hours

    def _validate_order_notional(
        self,
        asset: Asset,
        pair: str,
        qty: float,
        rate: float | None,
        side: str,
        order_kind: str,
    ) -> None:
        if not asset.fractionable or order_kind not in {"market", "limit"}:
            return
        reference_price = (
            rate
            if rate is not None
            else self.get_rate(pair, side="entry" if side == "buy" else "exit", refresh=True)
        )
        if qty * float(reference_price) < 1.0 - 1e-9:
            raise InvalidOrderException(
                f"Fractional Alpaca stock order for {asset.symbol} is below the $1 "
                f"minimum notional: qty={qty}, price={reference_price}."
            )

    def _validate_sell_quantity(self, symbol: str, side: str, qty: float) -> None:
        if side != "sell":
            return
        available = self._get_available_qty(symbol)
        if qty > available + 1e-12:
            raise InsufficientFundsError(
                f"Not enough {symbol} shares available to sell {qty}; available={available}."
            )

    def _build_order_request(
        self,
        asset: Asset,
        symbol: str,
        order_kind: str,
        side: str,
        qty: float,
        rate: float | None,
        tif: TimeInForce,
        extended_hours: bool,
        client_order_id: str | None,
    ) -> Any:
        side_enum = OrderSide.BUY if side == "buy" else OrderSide.SELL
        if order_kind == "market":
            return MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side_enum,
                time_in_force=tif,
                extended_hours=extended_hours,
                client_order_id=client_order_id,
            )
        if order_kind == "limit":
            if rate is None or rate <= 0:
                raise PricingError(f"A limit order for {symbol} requires a positive rate.")
            limit_price = self._normalize_price(asset, rate)
            if limit_price <= 0:
                raise InvalidOrderException(f"Invalid limit price {rate} for {symbol}")
            return LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side_enum,
                time_in_force=tif,
                limit_price=limit_price,
                extended_hours=extended_hours,
                client_order_id=client_order_id,
            )
        raise InvalidOrderException(
            f"Unsupported Alpaca stock order type: {order_kind}. Supported: market, limit."
        )

    def fetch_order(self, order_id: str, symbol: str, params: dict | None = None) -> dict[str, Any]:
        del params
        if not order_id:
            raise InvalidOrderException("Order id is required.")

        if self.dry_run:
            return self._fetch_dry_order(order_id, symbol)

        try:
            order = self.trading_client.get_order_by_id(order_id)
        except APIError as exc:
            message = str(exc).lower()
            if "not found" in message or "404" in message:
                raise InvalidOrderException(f"Alpaca order {order_id} was not found.") from exc
            raise self._translate_api_error(exc, f"fetch order {order_id}") from exc
        return self._normalize_order(order, symbol)

    def check_order_canceled_empty(self, order: dict[str, Any]) -> bool:
        """Return True when an order terminated without any fills.

        Freqtrade uses this distinction to delete/timeout an entry order immediately
        when it was canceled or otherwise reached a terminal state with zero fills.
        A partially filled terminal order must return False so its filled quantity
        is preserved and processed by Freqtrade.
        """
        if not isinstance(order, dict):
            return False

        status = str(order.get("status") or "").lower()
        filled = float(order.get("filled") or 0.0)

        return status in {
            "canceled",
            "cancelled",
            "expired",
            "rejected",
            "stopped",
            "done_for_day",
            "replaced",
        } and math.isclose(filled, 0.0, abs_tol=1e-12)

    def cancel_order(
        self, order_id: str, pair: str = "", params: dict | None = None
    ) -> dict[str, Any]:
        del params
        if self.dry_run:
            with self._dry_order_lock:
                order = self._dry_orders.get(order_id)
                if not order:
                    raise InvalidOrderException(f"Dry-run order {order_id} not found.")
                if order["status"] == "open":
                    order["status"] = "canceled"
                    order["remaining"] = max(0.0, order["amount"] - order["filled"])
                return dict(order)

        try:
            self.trading_client.cancel_order_by_id(order_id)
            # A cancel request is asynchronous. Return the actual order state;
            # pending_cancel remains "open" until Alpaca confirms the terminal state.
            return self.fetch_order(order_id, pair)
        except APIError as exc:
            message = str(exc).lower()
            if 'already in "filled" state' in message or "already filled" in message:
                return self.fetch_order(order_id, pair)
            raise InvalidOrderException(f"Could not cancel Alpaca order {order_id}: {exc}") from exc

    def cancel_order_with_result(self, order_id: str, pair: str, amount: float) -> dict[str, Any]:
        try:
            return self.cancel_order(order_id, pair)
        except InvalidOrderException:
            try:
                return self.fetch_order(order_id, pair)
            except InvalidOrderException:
                return {
                    "id": order_id,
                    "symbol": pair,
                    "type": "limit",
                    "side": "sell",
                    "amount": amount,
                    "filled": 0.0,
                    "remaining": amount,
                    "status": "canceled",
                    "price": None,
                    "average": None,
                    "cost": 0.0,
                    "fee": None,
                    "info": {},
                }

    def fetch_open_orders(
        self,
        symbol: str | None = None,
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ) -> list[dict[str, Any]]:
        del params
        if self.dry_run:
            with self._dry_order_lock:
                orders = [o for o in self._dry_orders.values() if o["status"] == "open"]
            if symbol:
                base = self._pair_symbol(symbol)
                orders = [o for o in orders if o["symbol"] == f"{base}/USD"]
            if since is not None:
                orders = [o for o in orders if o.get("timestamp", 0) >= since]
            return orders[:limit] if limit else orders

        after = self._ms_to_datetime(since) if since else None
        alpaca_filter = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            limit=min(int(limit), 500) if limit else None,
            after=after,
            symbols=[self._pair_symbol(symbol)] if symbol else None,
        )
        try:
            orders = self.trading_client.get_orders(alpaca_filter)
        except APIError as exc:
            raise self._translate_api_error(exc, "fetch open orders") from exc
        return [self._normalize_order(o, f"{o.symbol}/USD") for o in orders]

    def fetch_orders(
        self,
        pair: str,
        since: int | datetime | None = None,
        params: dict | None = None,
    ) -> list[dict[str, Any]]:
        del params
        if self.dry_run:
            with self._dry_order_lock:
                orders = list(self._dry_orders.values())
            orders = [o for o in orders if o["symbol"] == pair]
            if since:
                since_ms = (
                    int(since.timestamp() * 1000) if isinstance(since, datetime) else int(since)
                )
                orders = [o for o in orders if o.get("timestamp", 0) >= since_ms]
            return orders

        after = None
        if since:
            after = since if isinstance(since, datetime) else self._ms_to_datetime(int(since))
        symbol = self._pair_symbol(pair)
        request = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            limit=500,
            after=after,
            symbols=[symbol],
        )
        try:
            orders = self.trading_client.get_orders(request)
        except APIError as exc:
            raise self._translate_api_error(exc, f"fetch orders {pair}") from exc
        return [self._normalize_order(o, pair) for o in orders]

    # ------------------------------------------------------------------
    # Fees / fills
    # ------------------------------------------------------------------

    def get_fee(self, symbol: str, now: Any = None, taker_or_maker: str | None = None) -> float:
        del symbol, now, taker_or_maker
        # No maker/taker fee is assumed.  Actual broker/account fees and
        # regulatory charges are not synthesized into a crypto-style rate.
        return 0.0

    def order_has_fee(self, order: dict[str, Any]) -> bool:
        del order
        return False

    def get_funding_fees(
        self,
        pair: str,
        amount: float,
        is_short: bool = False,
        open_date: datetime | None = None,
        **kwargs: Any,
    ) -> float:
        """Return funding fees for a position.

        Equities in this adapter are spot positions, not perpetual/futures
        contracts, so there is no periodic funding payment.  Freqtrade still
        calls this method from the generic exit path, hence the explicit
        zero rather than leaving the method undefined.
        """
        del pair, amount, is_short, open_date, kwargs
        return 0.0

    def get_order_id_conditional(self, order: dict[str, Any]) -> str:
        """Return the exchange order id used for fill/fee lookup.

        Freqtrade calls this before querying execution trades.  Alpaca stock
        orders do not use a separate conditional/stop-loss id in this adapter,
        so the normal order ``id`` is always the correct identifier.
        """
        if not isinstance(order, dict):
            raise InvalidOrderException("Invalid order object: expected a dictionary.")
        order_id = order.get("id")
        if not order_id:
            raise InvalidOrderException("Order object does not contain an id.")
        return str(order_id)

    def get_trades_for_order(
        self,
        order_id: str,
        symbol: str,
        since: int | datetime | dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return Alpaca execution fills for a single order in Freqtrade format."""
        del kwargs
        since, params = self._normalise_fill_args(since, params)
        symbol = self._normalise_pair_string(symbol)
        if not order_id:
            raise InvalidOrderException("Order id is required for fill lookup.")

        if self.dry_run:
            return self._dry_order_fills(order_id, symbol)

        query = self._build_fill_query(order_id, since, params)
        return self._fetch_fill_activities(order_id, symbol, query)

    @staticmethod
    def _normalise_fill_args(
        since: int | datetime | dict[str, Any] | None,
        params: dict[str, Any] | None,
    ) -> tuple[int | datetime | None, dict[str, Any] | None]:
        if isinstance(since, dict):
            if params is not None:
                raise TypeError("Fill lookup accepts either since-as-params or params, not both.")
            return None, since
        return since, params

    def _dry_order_fills(self, order_id: str, symbol: str) -> list[dict[str, Any]]:
        with self._dry_order_lock:
            order = self._dry_orders.get(order_id)
        if not order:
            return []
        filled = float(order.get("filled") or 0.0)
        if filled <= 0:
            return []
        average = float(order.get("average") or order.get("price") or 0.0)
        timestamp = int(
            order.get("lastTradeTimestamp") or order.get("timestamp") or time.time() * 1000
        )
        return [
            {
                "id": f"{order_id}:fill",
                "timestamp": timestamp,
                "datetime": datetime.fromtimestamp(timestamp / 1000, UTC).isoformat(),
                "symbol": symbol,
                "side": str(order.get("side") or "buy").lower(),
                "price": average,
                "amount": filled,
                "cost": filled * average,
                "fee": None,
                "fees": [],
                "order": order_id,
                "info": {"dry_run": True},
            }
        ]

    def _build_fill_query(
        self,
        order_id: str,
        since: int | datetime | None,
        params: dict | None,
    ) -> dict[str, Any]:
        query: dict[str, Any] = {
            "activity_types": "FILL",
            "order_id": order_id,
            "page_size": 100,
            "direction": "asc",
        }
        if since is not None:
            since_dt = (
                self._ms_to_datetime(int(since)) if isinstance(since, (int, float)) else since
            )
            if since_dt.tzinfo is None:
                since_dt = since_dt.replace(tzinfo=UTC)
            query["after"] = since_dt.astimezone(UTC).isoformat().replace("+00:00", "Z")
        if params:
            for key in ("page_size", "direction"):
                if key in params:
                    query[key] = params[key]
        return query

    def _fetch_fill_activities(
        self,
        order_id: str,
        symbol: str,
        query: dict[str, Any],
    ) -> list[dict[str, Any]]:
        fills: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            payload = self._request_fill_page(order_id, query, page_token)
            for activity in payload:
                fill = self._activity_to_fill(activity, order_id, symbol, len(fills))
                if fill is not None:
                    fills.append(fill)
            page_size = int(query.get("page_size") or 100)
            if len(payload) < page_size:
                break
            last_id = payload[-1].get("id") if payload else None
            if not last_id or str(last_id) == str(page_token):
                break
            page_token = str(last_id)
        fills.sort(key=lambda item: (item["timestamp"], item["id"]))
        return fills

    def _request_fill_page(
        self,
        order_id: str,
        query: dict[str, Any],
        page_token: str | None,
    ) -> list[dict[str, Any]]:
        request_params = dict(query)
        if page_token:
            request_params["page_token"] = page_token
        try:
            self._throttle()
            response = self._http_session.get(
                f"{self._trading_api_base_url}/v2/account/activities",
                headers={
                    "APCA-API-KEY-ID": self.key,
                    "APCA-API-SECRET-KEY": self.secret,
                },
                params=request_params,
                timeout=30,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            status_code = getattr(response, "status_code", None)
            message = response.text[:500] if hasattr(response, "text") else str(exc)
            if status_code == 404:
                raise InvalidOrderException(
                    f"Alpaca fill lookup failed for order {order_id}: {message}"
                ) from exc
            if status_code == 429:
                raise TemporaryError(
                    f"Alpaca fill lookup rate-limited for order {order_id}: {message}"
                ) from exc
            raise OperationalException(
                f"Alpaca fill lookup failed for order {order_id}: {message}"
            ) from exc
        except requests.RequestException as exc:
            raise TemporaryError(
                f"Alpaca fill lookup request failed for order {order_id}: {exc}"
            ) from exc

        payload = response.json()
        if not isinstance(payload, list):
            raise OperationalException(
                f"Unexpected Alpaca account-activity response for order {order_id}."
            )
        return payload

    def _activity_to_fill(
        self,
        activity: Any,
        order_id: str,
        symbol: str,
        index: int,
    ) -> dict[str, Any] | None:
        if not isinstance(activity, dict):
            return None
        activity_order_id = str(activity.get("order_id") or "")
        if activity_order_id and activity_order_id != str(order_id):
            return None
        qty = self._as_float(activity.get("qty"), 0.0)
        price = self._as_float(activity.get("price"), 0.0)
        if qty <= 0 or price <= 0:
            return None
        timestamp_value = activity.get("transaction_time") or activity.get("date")
        timestamp = (
            self._datetime_to_ms(timestamp_value) if timestamp_value else int(time.time() * 1000)
        )
        fill_symbol = str(activity.get("symbol") or self._pair_symbol(symbol)).upper()
        return {
            "id": str(activity.get("id") or f"{order_id}:{index}"),
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp / 1000, UTC).isoformat(),
            "symbol": f"{fill_symbol}/USD",
            "side": str(activity.get("side") or "buy").lower(),
            "price": price,
            "amount": qty,
            "cost": qty * price,
            "fee": None,
            "fees": [],
            "order": str(order_id),
            "info": activity,
        }

    def extract_cost_curr_rate(self, *args: Any, **kwargs: Any) -> tuple[float, str, float]:
        del args, kwargs
        return 0.0, "USD", 0.0

    def handle_order_fee(self, *args: Any, **kwargs: Any) -> None:
        # Kept as a no-op for compatibility with older versions of the fork.
        # Actual fee handling is performed through Freqtrade's fill pipeline.
        pass

    # ------------------------------------------------------------------
    # Historical OHLCV
    # ------------------------------------------------------------------

    def validate_timeframes(self, timeframes: str | list[str]) -> None:
        values = [timeframes] if isinstance(timeframes, str) else list(timeframes)
        unsupported = [tf for tf in values if tf not in self._TF_MAP]
        if unsupported:
            raise ConfigurationError(
                f"Unsupported Alpaca stock timeframe(s): {', '.join(unsupported)}. "
                f"Supported: {', '.join(self._TF_MAP)}"
            )

    def convert_timeframe(self, timeframe: str) -> TimeFrame:
        try:
            return self._TF_MAP[timeframe]
        except KeyError as exc:
            raise ConfigurationError(f"Unsupported timeframe: {timeframe}") from exc

    def ohlcv_candle_limit(
        self, timeframe: str, candle_type: str = "trade", since_ms: int | None = None
    ) -> int:
        del candle_type, since_ms
        self.validate_timeframes(timeframe)
        return 10000

    def get_historic_ohlcv(
        self,
        pair: str,
        timeframe: str,
        since: int | None = None,
        limit: int = 10000,
        params: dict | None = None,
        since_ms: int | None = None,
        is_new_pair: bool = False,
        candle_type: str = "spot",
        until_ms: int | None = None,
    ) -> pd.DataFrame:
        del is_new_pair, candle_type
        return self._fetch_bars_df(
            pair=pair,
            timeframe=timeframe,
            since_ms=since if since is not None else since_ms,
            limit=limit,
            params=params or {},
            until_ms=until_ms,
        )

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ) -> list[list[Any]]:
        df = self._fetch_bars_df(
            pair=symbol,
            timeframe=timeframe,
            since_ms=since,
            limit=limit or 500,
            params=params or {},
        )
        if df.empty:
            return []
        pair_text = self._normalise_pair_string(symbol)
        with self._klines_lock:
            self._klines[(pair_text, timeframe, CandleType.SPOT)] = df.copy()
        return [
            [
                self._datetime_to_ms(row.date),
                float(row.open),
                float(row.high),
                float(row.low),
                float(row.close),
                float(row.volume),
            ]
            for row in df.itertuples(index=False)
        ]

    def klines(
        self,
        pair: Any,
        timeframe: str | None = None,
        *,
        candle_type: CandleType | str = CandleType.SPOT,
        copy: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Return the Freqtrade candle cache for an exact pair/timeframe/candle type.

        Freqtrade uses a 3-tuple candle identifier throughout the live data
        path: ``(pair, timeframe, CandleType)``.  Keep the cache keyed exactly
        that way so ``DataProvider.ohlcv()`` and ``available_pairs`` see the
        same objects the refresh layer populated.
        """
        del kwargs
        pair_text = self._normalise_pair_string(pair)
        if isinstance(pair, tuple) and len(pair) >= 2 and timeframe is None:
            timeframe = str(pair[1])
        if timeframe is None:
            raise ConfigurationError("timeframe is required for klines()")

        if isinstance(pair, tuple) and len(pair) >= 3:
            candle_type = pair[2]
        if not isinstance(candle_type, CandleType):
            candle_type = CandleType.from_string(str(candle_type))

        key = (pair_text, timeframe, candle_type)
        with self._klines_lock:
            cached = self._klines.get(key)
            if cached is not None:
                return cached.copy() if copy else cached

        # Populate the exact cache key on first access.
        df = self.get_historic_ohlcv(
            pair_text, timeframe, limit=self._initial_candle_limit(timeframe)
        )
        if not df.empty:
            with self._klines_lock:
                self._klines[key] = df.copy()
        return df.copy() if copy else df

    def refresh_latest_ohlcv(
        self,
        pair_list: list[Any],
        *,
        since_ms: int | None = None,
        cache: bool = True,
        drop_incomplete: bool | None = None,
    ) -> dict[tuple[str, str, CandleType], pd.DataFrame]:
        """Refresh live OHLCV using Freqtrade's exact candle identifiers."""
        is_open, time_until_open = self.is_market_open()
        if not is_open:
            self._wait_for_market_open(time_until_open)

        result: dict[tuple[str, str, CandleType], pd.DataFrame] = {}
        use_drop_incomplete = (
            self._ft_has.get("ohlcv_partial_candle", True)
            if drop_incomplete is None
            else bool(drop_incomplete)
        )
        for item in pair_list or []:
            key = self._normalise_candle_key(item)
            try:
                df = self._refresh_candle_key(key, since_ms, use_drop_incomplete)
                if df.empty:
                    logger.debug("No new OHLCV for %s %s", key[0], key[1])
                    continue
                if cache:
                    with self._klines_lock:
                        self._klines[key] = df.copy()
                result[key] = df.copy()
            except Exception as exc:
                logger.warning("Unable to refresh %s candles: %s", key, exc)
        return result

    def _wait_for_market_open(self, time_until_open: float) -> None:
        if time_until_open > 300:
            sleep_time = time_until_open - 300
            hours, rem = divmod(sleep_time, 3600)
            minutes, seconds = divmod(rem, 60)
            logger.info(
                "US stock market is closed, sleeping for %d hours, %d minutes, and %.1f "
                "seconds until 5 minutes before the next market opens.",
                hours,
                minutes,
                seconds,
            )
            time.sleep(sleep_time)
            return
        logger.info(
            "US stock market is closed, but opens in %.1f seconds; continuing with the "
            "final pre-open refresh.",
            max(0.0, time_until_open),
        )

    def _normalise_candle_key(self, item: Any) -> tuple[str, str, CandleType]:
        if isinstance(item, tuple) and len(item) >= 3:
            pair_text = self._normalise_pair_string(item[0])
            timeframe = str(item[1])
            raw_candle_type = item[2]
        elif isinstance(item, tuple) and len(item) == 2:
            pair_text = self._normalise_pair_string(item[0])
            timeframe = str(item[1])
            raw_candle_type = CandleType.SPOT
        else:
            pair_text = self._normalise_pair_string(item)
            timeframe = str(self.config.get("timeframe", "1m"))
            raw_candle_type = CandleType.SPOT
        candle_type = (
            raw_candle_type
            if isinstance(raw_candle_type, CandleType)
            else CandleType.from_string(str(raw_candle_type))
        )
        return pair_text, timeframe, candle_type

    def _refresh_candle_key(
        self,
        key: tuple[str, str, CandleType],
        since_ms: int | None,
        drop_incomplete: bool,
    ) -> pd.DataFrame:
        pair_text, timeframe, candle_type = key
        with self._klines_lock:
            existing = self._klines.get(key)
            existing_last_ms = (
                self._datetime_to_ms(existing["date"].iloc[-1])
                if existing is not None and not existing.empty
                else None
            )
        fetch_since = since_ms
        if fetch_since is None and existing_last_ms is not None:
            fetch_since = max(0, existing_last_ms - self._timeframe_seconds(timeframe) * 1000)
        limit = self._initial_candle_limit(timeframe) if existing is None or existing.empty else 10
        df = self.get_historic_ohlcv(
            pair_text,
            timeframe,
            since=fetch_since,
            limit=limit,
            candle_type=candle_type.value,
        )
        if df.empty:
            return existing.copy() if existing is not None and not existing.empty else df
        if existing is not None and not existing.empty:
            df = pd.concat([existing, df], ignore_index=True)
            df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        if drop_incomplete:
            df = self._drop_incomplete_candle(df, timeframe)
        return df

    def _drop_incomplete_candle(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if df.empty:
            return df
        now = pd.Timestamp.now(tz="UTC")
        tf_ms = self._timeframe_seconds(timeframe) * 1000
        last_ms = self._datetime_to_ms(df["date"].iloc[-1])
        current_bucket = (int(now.timestamp() * 1000) // tf_ms) * tf_ms
        if last_ms >= current_bucket and len(df) > 1:
            return df.iloc[:-1].copy()
        return df

    def _fetch_bars_df(
        self,
        pair: str,
        timeframe: str,
        since_ms: int | None,
        limit: int,
        params: dict[str, Any],
        until_ms: int | None = None,
    ) -> pd.DataFrame:
        self.validate_timeframes(timeframe)
        symbol = self._pair_symbol(pair)
        requested_limit = max(1, min(int(limit), 10000))

        end_dt = self._ms_to_datetime(until_ms) if until_ms is not None else datetime.now(UTC)
        if since_ms is None:
            seconds = self._timeframe_seconds(timeframe) * requested_limit
            start_dt = end_dt - timedelta(seconds=max(seconds, 86400))
        else:
            start_dt = self._ms_to_datetime(int(since_ms))

        # Give Alpaca a bounded request.  The SDK/API may paginate internally,
        # but we still enforce Freqtrade's requested result limit below.
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            start=start_dt,
            end=end_dt,
            limit=min(requested_limit, 10000),
            timeframe=self.convert_timeframe(timeframe),
            adjustment=self._parse_adjustment(params.get("adjustment", self.adjustment.value)),
            feed=self._parse_data_feed(params.get("feed", self.data_feed.value)),
            sort=Sort.ASC,
            currency="USD",
        )

        try:
            barset = self.data_client.get_stock_bars(request)
        except APIError as exc:
            raise self._translate_api_error(exc, f"fetch bars {symbol} {timeframe}") from exc

        df = getattr(barset, "df", None)
        if df is None or df.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        df = self._normalise_bar_dataframe(df, symbol)
        if df.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        if since_ms is not None:
            df = df[df["date"] >= self._ms_to_datetime(int(since_ms))]
        if until_ms is not None:
            df = df[df["date"] <= self._ms_to_datetime(int(until_ms))]
        df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

        # CCXT/Freqtrade limit means number of returned candles, not page size.
        if len(df) > requested_limit:
            df = df.iloc[:requested_limit].copy()
        return df[["date", "open", "high", "low", "close", "volume"]]

    def _normalise_bar_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        work = df.copy()
        if isinstance(work.index, pd.MultiIndex):
            names = list(work.index.names)
            if "symbol" in names:
                work = work.xs(symbol, level="symbol", drop_level=True)
            else:
                # Common BarSet layout is (symbol, timestamp).
                first_level = work.index.get_level_values(0)
                if symbol in first_level:
                    work = work.xs(symbol, level=0, drop_level=True)
            work = work.reset_index()
        else:
            work = work.reset_index()

        if "timestamp" not in work.columns:
            if "date" in work.columns:
                work = work.rename(columns={"date": "timestamp"})
            elif "index" in work.columns:
                work = work.rename(columns={"index": "timestamp"})

        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required.issubset(work.columns):
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        work["date"] = pd.to_datetime(work["timestamp"], utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        work = work.dropna(subset=["date", "open", "high", "low", "close", "volume"])
        return work[["date", "open", "high", "low", "close", "volume"]]

    # ------------------------------------------------------------------
    # Historical / live trades
    # ------------------------------------------------------------------

    def fetch_trades(
        self,
        pair: str,
        since: int | None = None,
        since_ms: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ) -> list[dict[str, Any]]:
        symbol = self._pair_symbol(pair)
        requested_limit = max(1, min(int(limit or self._ft_has["trades_limit"]), 10000))
        start_ms = since_ms if since_ms is not None else since
        request = StockTradesRequest(
            symbol_or_symbols=symbol,
            start=self._ms_to_datetime(start_ms) if start_ms is not None else None,
            limit=requested_limit,
            feed=self._parse_data_feed((params or {}).get("feed", self.data_feed.value)),
            sort=Sort.ASC,
        )
        try:
            trade_set = self.data_client.get_stock_trades(request)
        except APIError as exc:
            raise self._translate_api_error(exc, f"fetch trades {symbol}") from exc

        trades = trade_set.get(symbol, []) if hasattr(trade_set, "get") else []
        result = []
        for trade in trades[:requested_limit]:
            timestamp = self._datetime_to_ms(getattr(trade, "timestamp", None))
            result.append(
                {
                    "id": str(getattr(trade, "id", "")),
                    "timestamp": timestamp,
                    "datetime": datetime.fromtimestamp(timestamp / 1000, UTC).isoformat(),
                    "symbol": pair,
                    # Alpaca stock prints do not provide a CCXT-style taker side.
                    "side": None,
                    "price": self._as_float(getattr(trade, "price", None), 0.0),
                    "amount": self._as_float(getattr(trade, "size", None), 0.0),
                    "info": self._model_to_dict(trade),
                }
            )
        return result

    # ------------------------------------------------------------------
    # WebSockets
    # ------------------------------------------------------------------

    def setup_websocket(self) -> None:
        """Start Alpaca TradingStream for account/order updates only."""
        with self._stream_lock:
            if self._trading_stream_thread and self._trading_stream_thread.is_alive():
                return
            self._trading_stream = TradingStream(self.key, self.secret, paper=self.dry_run)
            self._trading_stream.subscribe_trade_updates(self._handle_trade_update)
            thread = threading.Thread(
                target=self._run_stream,
                args=(self._trading_stream, "trading"),
                daemon=True,
                name="alpaca-trading-stream",
            )
            self._trading_stream_thread = thread
            self._stream_threads.append(thread)
            self._stream_clients.append(self._trading_stream)
            thread.start()

    async def _handle_trade_update(self, trade_update: Any) -> None:
        order = getattr(trade_update, "order", None)
        order_id = getattr(order, "id", None)
        if order_id is not None:
            self._order_updates[str(order_id)] = trade_update
        logger.debug("Alpaca trade update: %s", trade_update)

    def _run_stream(self, stream: Any, name: str) -> None:
        try:
            stream.run()  # Alpaca's run() starts its own event loop; it is not a coroutine.
        except Exception:
            logger.exception("Alpaca %s websocket stopped unexpectedly.", name)

    async def watch_ticker(self, pairs: list[str], params: dict | None = None):
        del params
        symbols = [self._pair_symbol(pair) for pair in pairs]
        target_loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        stream = StockDataStream(self.key, self.secret, feed=self.data_feed)
        self._register_live_stream(stream)

        async def on_quote(quote: Any) -> None:
            payload = {
                "symbol": f"{quote.symbol}/USD",
                "timestamp": self._datetime_to_ms(quote.timestamp),
                "bid": self._as_float(getattr(quote, "bid_price", None), 0.0),
                "ask": self._as_float(getattr(quote, "ask_price", None), 0.0),
            }
            asyncio.run_coroutine_threadsafe(queue.put(payload), target_loop)

        stream.subscribe_quotes(on_quote, *symbols)
        thread = threading.Thread(target=self._run_stream, args=(stream, "quote"), daemon=True)
        self._stream_threads.append(thread)
        thread.start()

        try:
            while True:
                yield await queue.get()
        finally:
            self._stop_stream(stream)

    async def watch_trades(
        self,
        pair: str,
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ):
        del since, limit, params
        symbol = self._pair_symbol(pair)
        target_loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        stream = StockDataStream(self.key, self.secret, feed=self.data_feed)
        self._register_live_stream(stream)

        async def on_trade(trade: Any) -> None:
            timestamp = self._datetime_to_ms(trade.timestamp)
            payload = {
                "id": str(getattr(trade, "id", "")),
                "timestamp": timestamp,
                "datetime": datetime.fromtimestamp(timestamp / 1000, UTC).isoformat(),
                "symbol": pair,
                "side": None,
                "price": self._as_float(getattr(trade, "price", None), 0.0),
                "amount": self._as_float(getattr(trade, "size", None), 0.0),
            }
            asyncio.run_coroutine_threadsafe(queue.put(payload), target_loop)

        stream.subscribe_trades(on_trade, symbol)
        thread = threading.Thread(target=self._run_stream, args=(stream, "trade"), daemon=True)
        self._stream_threads.append(thread)
        thread.start()

        try:
            while True:
                yield await queue.get()
        finally:
            self._stop_stream(stream)

    async def watch_ohlcv(
        self,
        pair: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ):
        del since, limit, params
        self.validate_timeframes(timeframe)
        symbol = self._pair_symbol(pair)
        target_loop = asyncio.get_running_loop()
        queue: asyncio.Queue[list[Any]] = asyncio.Queue()
        stream = StockDataStream(self.key, self.secret, feed=self.data_feed)
        self._register_live_stream(stream)

        # Alpaca's stock websocket supplies native minute bars and daily bars.
        # Higher Freqtrade timeframes are assembled locally.  Buckets are
        # aligned to fixed wall-clock boundaries rather than the moment the
        # stream was started, and no candles are invented across market gaps.
        current_bucket: pd.Timestamp | None = None
        bucket_bars: list[list[Any]] = []

        async def on_minute_bar(bar: Any) -> None:
            nonlocal current_bucket, bucket_bars
            data = self._bar_to_ohlcv(bar)
            if timeframe == "1m":
                asyncio.run_coroutine_threadsafe(queue.put(data), target_loop)
                return

            bar_time = pd.Timestamp(data[0], unit="ms", tz="UTC")
            bucket = self._stream_bucket_start(bar_time, timeframe)
            if current_bucket is None:
                current_bucket = bucket
            elif bucket != current_bucket:
                if bucket_bars:
                    aggregated = self._aggregate_stream_buffer(bucket_bars, timeframe)
                    if aggregated is not None:
                        asyncio.run_coroutine_threadsafe(queue.put(aggregated), target_loop)
                current_bucket = bucket
                bucket_bars = []

            bucket_bars.append(data)

        async def on_daily_bar(bar: Any) -> None:
            asyncio.run_coroutine_threadsafe(queue.put(self._bar_to_ohlcv(bar)), target_loop)

        if timeframe == "1d":
            stream.subscribe_daily_bars(on_daily_bar, symbol)
        else:
            stream.subscribe_bars(on_minute_bar, symbol)

        thread = threading.Thread(target=self._run_stream, args=(stream, "bar"), daemon=True)
        self._stream_threads.append(thread)
        thread.start()

        try:
            while True:
                yield await queue.get()
        finally:
            self._stop_stream(stream)

    def _stream_bucket_start(self, timestamp: pd.Timestamp, timeframe: str) -> pd.Timestamp:
        """Return a fixed UTC bucket start for a streamed US-equity bar."""
        if timeframe in ("1m", "5m", "15m"):
            minutes = int(timeframe[:-1])
            local = timestamp.tz_convert("America/New_York")
            bucket = local.floor(f"{minutes}min")
            return bucket.tz_convert("UTC")
        if timeframe == "1h":
            # US regular-session hourly bars conventionally start at :30
            # (09:30, 10:30, ...).  This also keeps the adapter aligned with
            # the common equity-session anchor when aggregating Alpaca minute data.
            local = timestamp.tz_convert("America/New_York")
            bucket = (local - pd.Timedelta(minutes=30)).floor("1h") + pd.Timedelta(minutes=30)
            return bucket.tz_convert("UTC")
        raise ConfigurationError(f"Cannot stream-aggregate timeframe {timeframe}")

    def _aggregate_stream_buffer(self, bars: list[list[Any]], timeframe: str) -> list[Any] | None:
        """Aggregate the bars currently belonging to one fixed time bucket."""
        if not bars:
            return None
        if timeframe == "1m":
            return bars[-1]
        expected_minutes = self._timeframe_minutes(timeframe)
        if expected_minutes <= 0:
            raise ConfigurationError(f"Invalid aggregation timeframe: {timeframe}")

        # Never bridge a gap in actual market data.  A partial bucket is valid
        # for live streaming, but a missing-minute gap means the bucket should
        # be discarded rather than padded with synthetic zero-volume candles.
        timestamps = [pd.Timestamp(b[0], unit="ms", tz="UTC") for b in bars]
        for previous, current in pairwise(timestamps):
            if current - previous != pd.Timedelta(minutes=1):
                logger.debug("Skipping incomplete streamed %s bucket due to data gap", timeframe)
                return None

        return [
            bars[0][0],
            float(bars[0][1]),
            max(float(b[2]) for b in bars),
            min(float(b[3]) for b in bars),
            float(bars[-1][4]),
            sum(float(b[5]) for b in bars),
        ]

    def _bar_to_ohlcv(self, bar: Any) -> list[Any]:
        if isinstance(bar, dict):
            timestamp = bar.get("timestamp") or bar.get("t")
            return [
                self._datetime_to_ms(timestamp),
                self._as_float(bar.get("open", bar.get("o", 0.0))),
                self._as_float(bar.get("high", bar.get("h", 0.0))),
                self._as_float(bar.get("low", bar.get("l", 0.0))),
                self._as_float(bar.get("close", bar.get("c", 0.0))),
                self._as_float(bar.get("volume", bar.get("v", 0.0))),
            ]
        return [
            self._datetime_to_ms(bar.timestamp),
            float(bar.open),
            float(bar.high),
            float(bar.low),
            float(bar.close),
            float(bar.volume),
        ]

    def ws_connection_reset(self) -> None:
        with self._stream_lock:
            clients = list(self._stream_clients)
        for client in clients:
            self._stop_stream(client)

    # ------------------------------------------------------------------
    # Market clock
    # ------------------------------------------------------------------

    def is_market_open(self) -> tuple[bool, float]:
        try:
            clock = self._api_call(self.trading_client.get_clock, "get_clock")
        except ExchangeError:
            if self._market_open_cache is not None:
                return self._market_open_cache[0], max(
                    0.0, self._market_open_cache[1] - time.time()
                )
            raise

        now = time.time()
        if clock.is_open:
            self._market_open_cache = (True, now, now)
            return True, 0.0
        next_open = clock.next_open
        wait = max(0.0, (next_open - datetime.now(UTC)).total_seconds())
        self._market_open_cache = (False, now + wait, now)
        return False, wait

    # ------------------------------------------------------------------
    # Lifecycle / validation
    # ------------------------------------------------------------------

    def validate_config(self, config: dict) -> None:
        self.validate_trading_mode_and_margin_mode(
            config.get("trading_mode", "spot"),
            config.get("margin_mode"),
            allow_none_margin_mode=True,
        )
        self.validate_timeframes([config.get("timeframe", "1h")])

        # Authentication is checked only when not in a backtest and when
        # credentials are expected to be live-usable.  Market data is fetched
        # through the same credentials on dry-run, so data access is still real.
        if config.get("runmode") in ("dry_run", "live", "worker", None):
            self._api_call(self.trading_client.get_account, "validate Alpaca account")

    def close(self) -> None:
        self.ws_connection_reset()

    # ------------------------------------------------------------------
    # Dry-run implementation
    # ------------------------------------------------------------------

    def _create_dry_order(
        self,
        pair: str,
        ordertype: str,
        side: str,
        amount: float,
        rate: float | None,
        tif: TimeInForce,
        extended_hours: bool,
    ) -> dict[str, Any]:
        del extended_hours
        symbol = self._pair_symbol(pair)
        asset = self._get_asset(symbol)
        if not asset.tradable or self._enum_value(asset.status) != AssetStatus.ACTIVE.value:
            raise InvalidOrderException(f"Alpaca asset {symbol} is not tradable/active.")
        amount = self._normalize_order_quantity(asset, amount, ordertype)
        if amount <= 0:
            raise InvalidOrderException(f"Order quantity becomes zero for {symbol}.")
        if asset.fractionable and ordertype.lower() in {"market", "limit"}:
            reference_price = (
                rate
                if rate is not None
                else self.get_rate(pair, side="entry" if side == "buy" else "exit", refresh=True)
            )
            if amount * float(reference_price) < 1.0 - 1e-9:
                raise InvalidOrderException(
                    f"Fractional Alpaca stock order for {symbol} is below the $1 minimum notional: "
                    f"qty={amount}, price={reference_price}."
                )
        if ordertype.lower() == "limit" and (rate is None or rate <= 0):
            raise PricingError(f"A dry-run limit order for {pair} requires a positive rate.")
        if ordertype.lower() not in {"market", "limit"}:
            raise InvalidOrderException(f"Unsupported order type for dry-run: {ordertype}")
        order_id = f"dry_run_{side}_{uuid4()}"
        now_ms = int(time.time() * 1000)
        requested_rate = float(
            rate
            or self.get_rate(
                pair, side="entry" if side == "buy" else "exit", is_short=False, refresh=True
            )
        )
        if ordertype.lower() == "limit":
            requested_rate = self._normalize_price(asset, requested_rate)
        status = "closed" if ordertype.lower() == "market" else "open"
        filled = amount if status == "closed" else 0.0
        average = requested_rate if filled else None
        order = {
            "id": order_id,
            "symbol": pair,
            "type": ordertype.lower(),
            "side": side,
            "price": requested_rate,
            "average": average,
            "amount": amount,
            "filled": filled,
            "remaining": amount - filled,
            "status": status,
            "timestamp": now_ms,
            "lastTradeTimestamp": now_ms if filled else None,
            "datetime": datetime.fromtimestamp(now_ms / 1000, UTC).isoformat(),
            "cost": filled * (average or requested_rate),
            "fee": None,
            "timeInForce": tif.value,
            "info": {"dry_run": True, "tif": tif.value},
        }
        with self._dry_order_lock:
            self._dry_orders[order_id] = order
        return dict(order)

    def _fetch_dry_order(self, order_id: str, pair: str) -> dict[str, Any]:
        with self._dry_order_lock:
            order = self._dry_orders.get(order_id)

        # Freqtrade persists its Order rows even in dry-run mode.  The native
        # Exchange implementation normally keeps dry-run orders in memory, so
        # a bot restart can leave an old DB order that this adapter can no longer
        # see.  Recover that DB representation when possible instead of treating
        # the missing in-memory object as a fatal exchange error.
        if not order and order_id.startswith("dry_run_"):
            try:
                from freqtrade.persistence import Order as PersistedOrder

                db_order = PersistedOrder.order_by_id(order_id, self._normalise_pair_string(pair))
                if db_order is not None:
                    order = db_order.to_ccxt_object()
                    # Normalize persisted DB values to the fields used by the
                    # dry-run fill simulator.
                    order["type"] = (order.get("type") or "limit").lower()
                    order["side"] = (order.get("side") or db_order.ft_order_side or "buy").lower()
                    order["amount"] = float(order.get("amount") or db_order.ft_amount or 0.0)
                    order["filled"] = float(order.get("filled") or 0.0)
                    order["remaining"] = self._as_float(
                        order.get("remaining"),
                        order["amount"] - order["filled"],
                    )
                    order["price"] = (
                        float(order["price"])
                        if order.get("price") is not None
                        else float(db_order.ft_price)
                    )
                    order["average"] = (
                        float(order["average"]) if order.get("average") is not None else None
                    )
                    order.setdefault("symbol", self._normalise_pair_string(pair))
                    order.setdefault("cost", 0.0)
                    order.setdefault("fee", None)
                    with self._dry_order_lock:
                        self._dry_orders[order_id] = order
                    logger.info("Recovered dry-run order %s from Freqtrade persistence.", order_id)
            except Exception:
                logger.debug(
                    "Could not recover persisted dry-run order %s.",
                    order_id,
                    exc_info=True,
                )

        if not order:
            raise InvalidOrderException(f"Dry-run order {order_id} not found.")

        if order["status"] == "open" and order["type"] == "limit":
            ticker = self.fetch_ticker(pair)
            bid = float(ticker.get("bid") or 0.0)
            ask = float(ticker.get("ask") or 0.0)
            fillable = (order["side"] == "buy" and ask > 0 and ask <= order["price"]) or (
                order["side"] == "sell" and bid > 0 and bid >= order["price"]
            )
            if fillable:
                fill_price = ask if order["side"] == "buy" else bid
                with self._dry_order_lock:
                    order["status"] = "closed"
                    order["filled"] = order["amount"]
                    order["remaining"] = 0.0
                    order["average"] = fill_price
                    order["cost"] = order["amount"] * fill_price
                    order["lastTradeTimestamp"] = int(time.time() * 1000)
        return dict(order)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _enum_value(value: Any) -> Any:
        return getattr(value, "value", value)

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _datetime_to_ms(value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, datetime):
            dt = value
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return int(dt.timestamp() * 1000)
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.timestamp() * 1000)

    @staticmethod
    def _ms_to_datetime(value: int) -> datetime:
        return datetime.fromtimestamp(value / 1000, UTC)

    @staticmethod
    def _model_to_dict(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump(mode="json")
            except TypeError:
                return value.model_dump()
        if hasattr(value, "dict"):
            try:
                return value.dict()
            except Exception:
                logger.debug("Unable to serialize Alpaca model with dict()", exc_info=True)
        try:
            return dict(value)
        except Exception:
            return str(value)

    @staticmethod
    def _round_down_increment(value: float, increment: float) -> float:
        """Round *value* down to the supplied increment without destroying fractions.

        Very small increments such as 1e-9 are commonly rendered in scientific
        notation (``str(1e-9) == "1e-09"``).  The previous implementation treated
        that as a zero-decimal increment and rounded the result to an integer,
        which turned valid fractional stock quantities such as 5.266900165 into
        5.0.  Keep sufficient precision for Alpaca's 9-decimal fractional share
        quantities while still removing floating-point noise.
        """
        if increment <= 0:
            return float(value)

        units = math.floor((float(value) / float(increment)) + 1e-12)
        result = units * float(increment)

        # Alpaca fractionable equity quantities are supported to 9 decimal
        # places. Twelve decimals also gives enough room for price increments
        # while avoiding binary floating-point artifacts.
        return round(result, 12)

    @staticmethod
    def _normalise_pair_string(pair: Any) -> str:
        """Normalize Freqtrade pair inputs to a plain pair string.

        Freqtrade may pass candle identifiers as:
            ("TSLA/USD", "1m", CandleType.SPOT)
        while most exchange methods receive just "TSLA/USD".  The Alpaca
        adapter only needs the first element for symbol parsing.
        """
        if isinstance(pair, tuple):
            if not pair:
                raise ConfigurationError("Empty pair tuple received")
            pair = pair[0]
        if not isinstance(pair, str):
            raise ConfigurationError(
                f"Stock pair must be a string or Freqtrade pair tuple, got {pair!r}"
            )
        return pair

    @classmethod
    def _split_pair(cls, pair: Any) -> tuple[str, str]:
        text = cls._normalise_pair_string(pair).replace(":USD", "")
        if "/" not in text:
            raise ConfigurationError(f"Stock pair must be BASE/USD, got {pair}")
        base, quote = text.split("/", 1)
        if quote.upper() != "USD":
            raise ConfigurationError(f"Alpaca stock adapter only supports USD-quoted pairs: {pair}")
        return base.upper(), quote.upper()

    def _pair_symbol(self, pair: str) -> str:
        return self._split_pair(pair)[0]

    def _parse_tif(self, value: str | TimeInForce | None) -> TimeInForce:
        if isinstance(value, TimeInForce):
            return value
        key = str(value or "DAY").lower()
        try:
            return self._TIF_MAP[key]
        except KeyError as exc:
            raise ConfigurationError(
                f"Unsupported Alpaca time-in-force {value}. Supported: {', '.join(self._TIF_MAP)}"
            ) from exc

    @staticmethod
    def _parse_data_feed(value: str | DataFeed) -> DataFeed:
        if isinstance(value, DataFeed):
            return value
        text = str(value).lower()
        for member in DataFeed:
            if member.value.lower() == text or member.name.lower() == text:
                return member
        raise ConfigurationError(f"Unsupported Alpaca stock data feed: {value}")

    @staticmethod
    def _parse_adjustment(value: str | Adjustment) -> Adjustment:
        if isinstance(value, Adjustment):
            return value
        text = str(value).lower()
        for member in Adjustment:
            if member.value.lower() == text or member.name.lower() == text:
                return member
        raise ConfigurationError(f"Unsupported Alpaca stock adjustment: {value}")

    def _initial_candle_limit(self, timeframe: str) -> int:
        """Return enough candles for startup plus a small safety margin."""
        startup = int(self.config.get("startup_candle_count", 0) or 0)
        return max(500, startup + 50, 1000 if timeframe == "1m" else 500)

    @staticmethod
    def _timeframe_minutes(timeframe: str) -> int:
        if timeframe.endswith("m"):
            return int(timeframe[:-1])
        if timeframe.endswith("h"):
            return int(timeframe[:-1]) * 60
        if timeframe.endswith("d"):
            return int(timeframe[:-1]) * 1440
        raise ConfigurationError(f"Unsupported timeframe {timeframe}")

    def _timeframe_seconds(self, timeframe: str) -> int:
        return self._timeframe_minutes(timeframe) * 60

    @staticmethod
    def _is_whole_quantity(value: float) -> bool:
        return math.isclose(float(value), round(float(value)), rel_tol=0.0, abs_tol=1e-10)

    def _normalize_order_quantity(self, asset: Asset, amount: float, ordertype: str) -> float:
        value = float(amount)
        if value <= 0:
            return 0.0

        ordertype = ordertype.lower()
        fractionable = bool(getattr(asset, "fractionable", False))

        # For fractionable equities, preserve fractional quantities for market
        # orders and DAY limit orders. Alpaca supports up to 9 decimal places.
        if fractionable and ordertype in {"market", "limit"}:
            raw_increment = self._as_float(getattr(asset, "min_trade_increment", None), 0.0)
            increment = raw_increment if 0.0 < raw_increment < 1.0 else 1e-9
            # For fractionable equities, Alpaca's equity minimum is a notional
            # rule rather than a whole-share minimum. Quantity itself can be
            # represented to 9 decimal places.
            minimum = increment
            value = math.floor(value * 1_000_000_000 + 1e-9) / 1_000_000_000
        else:
            # Non-fractionable equity quantities must be whole shares.
            increment = max(1.0, float(getattr(asset, "min_trade_increment", None) or 1.0))
            minimum = max(1.0, float(getattr(asset, "min_order_size", None) or 1.0))

        qty = self._round_down_increment(value, increment)
        if not fractionable or ordertype not in {"market", "limit"}:
            qty = math.floor(qty + 1e-12)

        if qty < minimum - 1e-12:
            raise InvalidOrderException(
                f"Order quantity {qty} for {asset.symbol} is below Alpaca minimum {minimum}."
            )
        return qty

    def _normalize_price(self, asset: Asset, price: float) -> float:
        increment = float(asset.price_increment or 0.01)
        return self._round_down_increment(price, increment)

    def _get_available_qty(self, symbol: str) -> float:
        positions = self._api_call(self.trading_client.get_all_positions, "get_all_positions")
        for position in positions:
            if position.symbol.upper() == symbol.upper():
                return max(
                    0.0, self._as_float(position.qty_available, self._as_float(position.qty, 0.0))
                )
        return 0.0

    def _normalize_order(
        self,
        order: Any,
        pair: str,
        requested_rate: float | None = None,
    ) -> dict[str, Any]:
        status = str(self._enum_value(getattr(order, "status", "new"))).lower()
        normalized_status = self._ft_has["order_statuses"].get(status, "open")
        filled = self._as_float(getattr(order, "filled_qty", None), 0.0)
        amount = self._as_float(getattr(order, "qty", None), filled)
        remaining = max(0.0, amount - filled)
        avg = self._as_float(getattr(order, "filled_avg_price", None), 0.0)
        limit_price = self._as_float(getattr(order, "limit_price", None), 0.0)
        requested_price = requested_rate or limit_price or avg or None
        if filled >= amount > 0 and status not in {"rejected", "canceled", "expired"}:
            normalized_status = "closed"
            remaining = 0.0

        submitted_at = getattr(order, "submitted_at", None) or getattr(order, "created_at", None)
        filled_at = getattr(order, "filled_at", None)
        timestamp = self._datetime_to_ms(submitted_at) if submitted_at else None
        last_trade_timestamp = self._datetime_to_ms(filled_at) if filled_at else None
        side = str(self._enum_value(getattr(order, "side", "buy"))).lower()
        order_type = str(
            self._enum_value(getattr(order, "order_type", getattr(order, "type", "limit")))
        ).lower()
        symbol = f"{getattr(order, 'symbol', self._pair_symbol(pair))}/USD"
        cost = filled * avg if avg > 0 else 0.0

        return {
            "id": str(getattr(order, "id", "")),
            "clientOrderId": getattr(order, "client_order_id", None),
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "price": requested_price,
            "average": avg or None,
            "amount": amount,
            "filled": filled,
            "remaining": remaining,
            "status": normalized_status,
            "timestamp": timestamp,
            "lastTradeTimestamp": last_trade_timestamp,
            "datetime": datetime.fromtimestamp(timestamp / 1000, UTC).isoformat()
            if timestamp
            else None,
            "cost": cost,
            "fee": None,
            "timeInForce": self._enum_value(getattr(order, "time_in_force", None)),
            "info": self._model_to_dict(order),
        }

    def _translate_api_error(self, error: APIError, context: str) -> Exception:
        message = str(error)
        lower = message.lower()
        if any(token in lower for token in ("insufficient", "buying power", "not enough")):
            return InsufficientFundsError(f"{context}: {message}")
        if any(token in lower for token in ("not found", "order not found")):
            return InvalidOrderException(f"{context}: {message}")
        if any(
            token in lower
            for token in ("timeout", "temporarily", "rate limit", "too many requests", "connection")
        ):
            return TemporaryError(f"{context}: {message}")
        return OperationalException(f"{context}: {message}")

    def _api_call(self, func: Any, context: str, order_exception: bool = False) -> Any:
        try:
            return func()
        except APIError as exc:
            translated = self._translate_api_error(exc, context)
            if order_exception and not isinstance(
                translated, (InsufficientFundsError, InvalidOrderException)
            ):
                translated = OperationalException(str(translated))
            raise translated from exc
        except (TimeoutError, OSError) as exc:
            raise TemporaryError(f"{context}: {exc}") from exc

    def _throttle(self) -> None:
        """Throttle direct HTTP activity requests made by this adapter."""
        # Alpaca SDK clients have their own request handling; this throttle only
        # applies to our direct account-activity HTTP calls.
        now = time.monotonic()
        last = getattr(self, "_last_http_request", 0.0)
        elapsed = now - last
        if elapsed < 3.0:
            time.sleep(3.0 - elapsed)
        self._last_http_request = time.monotonic()

    def _register_live_stream(self, stream: Any) -> None:
        with self._stream_lock:
            self._stream_clients.append(stream)

    def _stop_stream(self, stream: Any) -> None:
        try:
            stream.stop()
        except Exception:
            logger.debug("Error stopping Alpaca websocket", exc_info=True)
        with self._stream_lock:
            if stream in self._stream_clients:
                self._stream_clients.remove(stream)


# Keep the historical class naming convention used by the fork.
__all__ = ["Alpacastocks"]
