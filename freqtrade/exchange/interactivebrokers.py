"""Native Interactive Brokers forex exchange adapter for the Freqtrade stock/forex fork.

This module intentionally bypasses CCXT.  It translates between the
Freqtrade/CCXT-shaped interface used by the fork and IBKR's TWS API via
ib_insync (IDEALPRO cash forex).

Supported trading model:
    * Forex (CASH / IDEALPRO) only
    * spot / long-only (no crypto-style margin modes)
    * market and limit orders
    * DAY / GTC / IOC / FOK time-in-force

Important design rules:
    * Freqtrade ``dry_run`` maps to IBKR **paper** trading (IB Gateway paper
      port, typically 4002). Orders are submitted through the TWS API and
      filled by IB's paper system — they are not simulated locally.
    * Live mode uses the live port (typically 7496 for TWS).
    * Currency balances come from the connected paper or live account.
    * Missing forex candles are NOT fabricated across the weekend gap.
    * Order lifecycle always returns Freqtrade-normalized dicts; it does not
      raise for ordinary cancels/rejects.
    * After (re)connect, open orders and executions are resynced so
      fetch_order does not false-negative on pre-connection orders.
"""

from __future__ import annotations

import atexit
import logging
import math
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pandas as pd
from ib_insync import IB, Forex, LimitOrder, MarketOrder, Order, Trade, util

from freqtrade.enums import CandleType, MarginMode
from freqtrade.exceptions import (
    ConfigurationError,
    ExchangeError,
    InsufficientFundsError,
    InvalidOrderException,
    OperationalException,
    PricingError,
    TemporaryError,
)
from freqtrade.exchange.foreignexchange import Foreignexchange


util.patchAsyncio()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global request throttle (IBKR pacing)
# ---------------------------------------------------------------------------

_min_interval = 0.35
_last_request_ts = 0.0
_request_lock = threading.Lock()


def throttle() -> None:
    """Serialize IB API calls to reduce pacing violations."""
    global _last_request_ts
    with _request_lock:
        now = time.time()
        elapsed = now - _last_request_ts
        if elapsed < _min_interval:
            time.sleep(_min_interval - elapsed)
        _last_request_ts = time.time()


# IDEALPRO minimum order sizes (base-currency units).  Orders below these
# sizes are odd lots and receive inferior spreads.  Values follow IBKR's
# published forex min/max chart (approximate; account-specific limits may
# differ).
_IDEALPRO_MIN_BASE: dict[str, float] = {
    "USD": 25_000.0,
    "EUR": 20_000.0,
    "GBP": 20_000.0,
    "AUD": 25_000.0,
    "NZD": 35_000.0,
    "CAD": 25_000.0,
    "CHF": 25_000.0,
    "JPY": 2_500_000.0,
    "CNH": 150_000.0,
    "HKD": 200_000.0,
    "MXN": 300_000.0,
    "SGD": 35_000.0,
    "NOK": 150_000.0,
    "SEK": 175_000.0,
    "DKK": 150_000.0,
    "ZAR": 350_000.0,
    "TRY": 100_000.0,
}

# Default price increments when contract details are unavailable.
_DEFAULT_PRICE_TICK: dict[str, float] = {
    "JPY": 0.001,
}


class Interactivebrokers(Foreignexchange):
    """Freqtrade adapter for Interactive Brokers IDEALPRO forex."""

    RECONNECT_MAX_BACKOFF = 32.0
    RECONNECT_BASE_BACKOFF = 1.0

    # Freqtrade precision-mode constants.  Tick size is preferred because
    # IBKR expresses forex constraints as minTick increments.
    DECIMAL_PLACES = 2
    SIGNIFICANT_DIGITS = 3
    TICK_SIZE = 4

    _ft_has_default: dict[str, Any] = {
        "stoploss_on_exchange": False,
        "order_time_in_force": ["GTC", "DAY", "IOC", "FOK"],
        "ohlcv_candle_limit": 1000,
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
        "trades_has_history": False,
        "market_has_ticker": True,
        "market_has_ohlcv": True,
        "order_has_status": True,
        "order_has_type": True,
        "order_has_side": True,
        "order_has_time_in_force": True,
        "order_has_price": True,
        "order_has_amount": True,
        "order_has_cost": True,
        "order_has_fee": False,
        "order_has_slippage": False,
        "order_has_filled": True,
        "order_has_remaining": True,
        "order_statuses": {
            "apipending": "open",
            "pendingsubmit": "open",
            "presubmitted": "open",
            "submitted": "open",
            "pendingcancel": "open",
            "filled": "closed",
            "cancelled": "canceled",
            "canceled": "canceled",
            "apicancelled": "canceled",
            "inactive": "canceled",
            "rejected": "rejected",
        },
        "ws_enabled": True,
        "ws_auto_reconnect": True,
        "ws_reconnect_interval": 30,
        "balance_includes_unrealized_pnl": False,
        "contract_size": 1.0,
        "needs_trading_fees": False,
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
    }

    _TF_MAP = {
        "1m": ("1 min", 1),
        "5m": ("5 mins", 5),
        "15m": ("15 mins", 15),
        "30m": ("30 mins", 30),
        "1h": ("1 hour", 60),
        "4h": ("4 hours", 240),
        "1d": ("1 day", 1440),
    }

    _TIF_MAP = {
        "day": "DAY",
        "gtc": "GTC",
        "ioc": "IOC",
        "fok": "FOK",
    }

    # Default major pairs when no custom list is supplied.
    _DEFAULT_PAIRS: list[tuple[str, str]] = [
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
        self.id = "interactivebrokers"
        self.dry_run = bool(config.get("dry_run", False))

        self._ft_has = dict(self._ft_has_default)
        self._markets: dict[str, dict[str, Any]] = {}
        self._klines: dict[tuple[str, str, CandleType], pd.DataFrame] = {}
        self._klines_lock = threading.RLock()

        # Rate cache: (pair, is_short) -> (entry, exit, monotonic_time).
        self._rate_cache: dict[tuple[str, bool], tuple[float, float, float]] = {}
        self._rate_cache_ttl = float(exchange_conf.get("rate_cache_ttl", 2.0))
        self._rate_cache_lock = threading.RLock()

        # IDEALPRO full interbank size starts around 20k-25k base units.
        # Odd lots are accepted by IB but get wider spreads. Default False so
        # Freqtrade's min-stake logic matches real IDEALPRO requirements.
        self.allow_odd_lots = bool(exchange_conf.get("allow_odd_lots", False))
        self.odd_lot_min_base = float(exchange_conf.get("odd_lot_min_base", 1_000.0))

        # Dry-run orders never reach IBKR.
        self._dry_orders: dict[str, dict[str, Any]] = {}
        self._dry_order_lock = threading.RLock()

        # Live price cache: pair -> (monotonic_ts, price).
        self._live_price_cache: dict[str, tuple[float, float]] = {}
        self._live_price_lock = threading.RLock()

        # Connection state.
        self.host = str(exchange_conf.get("ib_host") or config.get("ib_host") or "127.0.0.1")
        self.client_id = int(exchange_conf.get("ib_client_id") or config.get("ib_client_id") or 1)
        if self.dry_run:
            self.port = int(
                exchange_conf.get("ib_paper_port") or config.get("ib_paper_port") or 4002
            )
        else:
            self.port = int(exchange_conf.get("ib_live_port") or config.get("ib_live_port") or 7496)

        self._running = True
        self._ws_connected = False
        self.connected = False
        self.is_shutting_down = False
        self._connection_thread: threading.Thread | None = None
        self._reconnect_lock = threading.Lock()
        self._market_open_cache: tuple[bool, float] | None = None

        # Contract detail cache: pair -> (min_tick, min_size).
        self._contract_meta: dict[str, dict[str, float]] = {}
        self._contract_meta_lock = threading.RLock()

        self.ib = IB()
        try:
            self.ib.startLoop()
        except Exception as exc:
            logger.debug("ib.startLoop() unavailable or already running: %s", exc)

        atexit.register(self.close)

        if "candle_type_def" not in self.config:
            self.config["candle_type_def"] = "spot"

        logger.info(
            "Initialized interactivebrokers (%s, host=%s, port=%s, clientId=%s)",
            "dry-run" if self.dry_run else "live",
            self.host,
            self.port,
            self.client_id,
        )

        # Markets are static metadata and do not require a live connection.
        self.reload_markets(force=True)

        # Connect for market data even in dry-run (orders stay local).
        # Keep this path free of extra blocking IB requests (reqAllOpenOrders /
        # reqCurrentTime / reqExecutions): those can hang indefinitely on some
        # TWS/Gateway builds and stall Freqtrade during startup validation.
        try:
            self._connect_to_ib(do_resync=False)
        except Exception as exc:
            if validate and not self.dry_run:
                raise OperationalException(
                    f"Could not connect to IBKR at {self.host}:{self.port}: {exc}"
                ) from exc
            logger.warning(
                "IBKR connection deferred (dry-run or validate=False): %s",
                exc,
            )

        if validate:
            self.validate_config(config)

        # Supervisor + lazy order-cache resync only after config validation
        # so a stuck IB RPC cannot block bot startup.
        if self._ft_has["ws_enabled"] and self.ib.isConnected():
            self._setup_connection_supervisor()
            threading.Thread(
                target=self._safe_resync_order_cache,
                daemon=True,
                name="ibkr-order-resync",
            ).start()

    # ------------------------------------------------------------------
    # Identity / capability
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "interactivebrokers"

    @property
    def markets(self) -> dict[str, dict[str, Any]]:
        return self._markets

    @markets.setter
    def markets(self, value: dict[str, dict[str, Any]]) -> None:
        self._markets = value

    @property
    def timeframes(self) -> list[str]:
        return list(self._TF_MAP)

    @property
    def margin_mode(self):
        return MarginMode.NONE

    @property
    def precisionMode(self) -> int:
        return self.TICK_SIZE

    @property
    def precision_mode_price(self) -> int:
        return self.TICK_SIZE

    def get_option(self, option: str, default: Any = None) -> Any:
        return self._ft_has.get(option, default)

    def exchange_has(self, method: str) -> bool:
        mapped = self._METHOD_MAP.get(method, method)
        return hasattr(self, mapped)

    # ------------------------------------------------------------------
    # Basic exchange / spot contract
    # ------------------------------------------------------------------

    def get_proxy_coin(self) -> str:
        return str(self.config.get("stake_currency", "USD")).upper()

    def get_pair_base_currency(self, pair: str) -> str:
        return self._split_pair(pair)[0]

    def get_pair_quote_currency(self, pair: str) -> str:
        return self._split_pair(pair)[1]

    def get_contract_size(self, pair: str) -> float:
        del pair
        return 1.0

    def get_liquidation_price(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def _contracts_to_amount(self, pair: str, num_contracts: float) -> float:
        del pair
        return float(num_contracts)

    def _amount_to_contracts(self, pair: str, amount: float) -> float:
        del pair
        return float(amount)

    def amount_to_contract_precision(self, pair: str, amount: float) -> float:
        return self.amount_to_precision(pair, amount)

    def balance_includes_unrealized_pnl(self) -> bool:
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
        del allow_none_margin_mode, kwargs
        if trading_mode and str(trading_mode).lower() != "spot":
            raise OperationalException(
                "interactivebrokers forex adapter supports spot trading only."
            )
        if margin_mode not in (
            None,
            "",
            "none",
            MarginMode.NONE,
        ) and str(margin_mode).lower() not in ("none", ""):
            raise OperationalException(
                "interactivebrokers forex adapter does not expose crypto margin modes."
            )

    def validate_required_startup_candles(self, startup_candle_count: int, timeframe: str) -> bool:
        return timeframe in self._TF_MAP and startup_candle_count >= 0

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def _connect_to_ib(self, *, do_resync: bool = True) -> None:
        with self._reconnect_lock:
            if self.ib.isConnected():
                self._ws_connected = True
                self.connected = True
                return

            logger.info(
                "Connecting to IBKR host=%s port=%s clientId=%s (%s)",
                self.host,
                self.port,
                self.client_id,
                "paper/dry-run" if self.dry_run else "live",
            )
            try:
                throttle()
                # timeout applies to the TCP/handshake phase inside ib_insync.
                self.ib.connect(
                    self.host,
                    self.port,
                    clientId=self.client_id,
                    timeout=8,
                )
            except ConnectionRefusedError as exc:
                self._ws_connected = False
                self.connected = False
                raise OperationalException(
                    f"Connection refused: is TWS/IB Gateway running on {self.host}:{self.port}?"
                ) from exc
            except Exception as exc:
                self._ws_connected = False
                self.connected = False
                raise OperationalException(f"IBKR connect failed: {exc}") from exc

            if not self.ib.isConnected():
                self._ws_connected = False
                self.connected = False
                raise OperationalException("IBKR connect returned without an active session.")

            self._ws_connected = True
            self.connected = True
            logger.info("IBKR connection established.")
            if do_resync:
                self._safe_resync_order_cache()

    def _safe_resync_order_cache(self) -> None:
        """Best-effort order cache refresh; never block startup forever."""
        try:
            self._resync_order_cache(timeout=5.0)
        except Exception as exc:
            logger.warning("Deferred IBKR order-cache resync failed: %s", exc)

    def _resync_order_cache(self, timeout: float = 5.0) -> None:
        """Repopulate ib_insync client-side order/trade cache after (re)connect.

        ``reqAllOpenOrders`` / ``reqExecutions`` are blocking.  On some TWS and
        Gateway builds they never complete, which previously froze Freqtrade
        during exchange __init__.  Run each call with a hard timeout.
        """
        if not self.ib.isConnected():
            return

        def _call(label: str, func: Any) -> None:
            done = threading.Event()
            error: list[BaseException] = []

            def runner() -> None:
                try:
                    throttle()
                    func()
                except BaseException as exc:
                    error.append(exc)
                finally:
                    done.set()

            thread = threading.Thread(target=runner, daemon=True, name=f"ibkr-resync-{label}")
            thread.start()
            if not done.wait(timeout):
                logger.warning(
                    "IBKR %s timed out after %.1fs — continuing without it.",
                    label,
                    timeout,
                )
                return
            if error:
                raise error[0]

        try:
            _call("reqAllOpenOrders", self.ib.reqAllOpenOrders)
            _call("reqExecutions", self.ib.reqExecutions)
            logger.debug("IBKR order/execution cache resync finished.")
        except Exception as exc:
            logger.warning("Failed to resync IBKR order/execution cache: %s", exc)

    def ensure_connected(self) -> None:
        if self.is_shutting_down:
            raise OperationalException("Shutdown in progress.")
        if self.ib.isConnected():
            return

        backoff = self.RECONNECT_BASE_BACKOFF
        while backoff <= self.RECONNECT_MAX_BACKOFF and not self.is_shutting_down:
            logger.warning("IBKR disconnected — reconnecting in %.1fs…", backoff)
            time.sleep(backoff)
            try:
                self._connect_to_ib()
                return
            except Exception as exc:
                logger.error("Reconnect attempt failed: %s", exc)
                backoff = min(backoff * 2, self.RECONNECT_MAX_BACKOFF + 1)

        raise TemporaryError("Unable to reconnect to IBKR after multiple attempts.")

    def _setup_connection_supervisor(self) -> None:
        if self._connection_thread and self._connection_thread.is_alive():
            return

        def _loop() -> None:
            while self._running and not self.is_shutting_down:
                try:
                    if not self.ib.isConnected():
                        try:
                            self.ensure_connected()
                        except Exception:
                            time.sleep(5)
                    else:
                        self.ib.sleep(1)
                except Exception:
                    logger.exception("IBKR connection supervisor error")
                    time.sleep(5)

        self._connection_thread = threading.Thread(
            target=_loop, daemon=True, name="ibkr-connection-supervisor"
        )
        self._connection_thread.start()

    def ws_start(self) -> None:
        try:
            self.ensure_connected()
            self._setup_connection_supervisor()
        except Exception as exc:
            logger.error("ws_start failed: %s", exc)
            self._ws_connected = False

    def ws_stop(self) -> None:
        try:
            if self.ib.isConnected():
                self.ib.disconnect()
        except Exception as exc:
            logger.debug("ws_stop disconnect: %s", exc)
        self._ws_connected = False

    def ws_connection_reset(self) -> None:
        self.ws_stop()
        try:
            self._connect_to_ib()
            self._setup_connection_supervisor()
        except Exception as exc:
            logger.error("ws_connection_reset failed: %s", exc)

    def ws_health_check(self) -> bool:
        """Cheap liveness probe — avoid reqCurrentTime (can hang on IB)."""
        return bool(self.ib.isConnected())

    def close(self) -> None:
        self.is_shutting_down = True
        self._running = False
        try:
            if self.ib.isConnected():
                self.ib.disconnect()
        except Exception as exc:
            logger.debug("Error during IBKR disconnect: %s", exc)
        self._ws_connected = False
        self.connected = False

    # ------------------------------------------------------------------
    # Markets / asset metadata
    # ------------------------------------------------------------------

    def reload_markets(self, force: bool = True, *, load_leverage_tiers: bool = False) -> None:
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

        custom = self.exchange_config.get("pair_whitelist") or self.config.get("exchange", {}).get(
            "pair_whitelist"
        )
        pairs: list[tuple[str, str]] = []
        if custom:
            for item in custom:
                try:
                    pairs.append(self._split_pair(str(item)))
                except ConfigurationError:
                    continue
        if not pairs:
            pairs = list(self._DEFAULT_PAIRS)

        markets: dict[str, dict[str, Any]] = {}
        for base, quote in pairs:
            pair = f"{base}/{quote}"
            min_qty = self._idealpro_min_base(base)
            price_tick = self._default_price_tick(pair)
            markets[pair] = {
                "id": pair,
                "symbol": pair,
                "base": base,
                "quote": quote,
                "spot": True,
                "tradable": True,
                "margin": False,
                "active": True,
                "maker": 0.0,
                "taker": 0.0,
                "info": {"base": base, "quote": quote, "exchange": "IDEALPRO"},
                "precision": {
                    "amount": 1.0,
                    "price": price_tick,
                },
                "limits": {
                    "amount": {"min": min_qty, "max": None},
                    "price": {"min": price_tick, "max": None},
                    "cost": {"min": None, "max": None},
                },
                "future": False,
                "swap": False,
                "option": False,
                "linear": False,
                "inverse": False,
                "contractSize": 1.0,
                "market_type": "spot",
            }
        self._markets = markets
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

    # ------------------------------------------------------------------
    # Precision / stake limits
    # ------------------------------------------------------------------

    def get_precision_price(self, pair: str) -> float:
        meta = self._get_contract_meta(pair)
        if meta and meta.get("min_tick", 0) > 0:
            return float(meta["min_tick"])
        return float(self.markets.get(pair, {}).get("precision", {}).get("price", 0.00001))

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
        base, _quote = self._split_pair(pair)
        if self.allow_odd_lots:
            min_base = max(1.0, float(self.odd_lot_min_base))
        else:
            min_base = self._idealpro_min_base(base)
        p = float(price or self.get_rate(pair, side="entry", is_short=False, refresh=True))
        # Stake is in quote (typically USD). Convert min base units to quote notional.
        return float(min_base * p)

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
        """Return spot balances: cash per currency plus FX position sizes.

        Paper (dry_run) and live both read the connected IB account so
        Freqtrade wallets match what TWS/Gateway shows.
        """
        self.ensure_connected()
        balances: dict[str, dict[str, float]] = {}

        try:
            throttle()
            summary = self.ib.accountSummary()
        except Exception as exc:
            raise TemporaryError(f"accountSummary failed: {exc}") from exc

        for item in summary:
            tag = getattr(item, "tag", "")
            currency = str(getattr(item, "currency", "") or "").upper()
            if not currency:
                continue
            if tag == "TotalCashValue":
                cash = self._as_float(item.value, 0.0)
                entry = balances.setdefault(currency, {"free": 0.0, "used": 0.0, "total": 0.0})
                entry["free"] = cash
                entry["total"] = cash + entry.get("used", 0.0)

        try:
            throttle()
            positions = self.ib.positions()
        except Exception as exc:
            logger.warning("positions() failed while building balances: %s", exc)
            positions = []

        for pos in positions:
            symbol = str(getattr(pos.contract, "symbol", "") or "").upper()
            qty = self._as_float(getattr(pos, "position", 0.0), 0.0)
            if not symbol or abs(qty) < 1e-12:
                continue
            # Long base currency is a positive balance of that currency.
            entry = balances.setdefault(symbol, {"free": 0.0, "used": 0.0, "total": 0.0})
            # Net FX position into free; IB does not split reserved FX the way
            # stock equity qty_available does.
            entry["free"] = entry.get("free", 0.0) + qty
            entry["total"] = entry.get("free", 0.0) + entry.get("used", 0.0)

        return balances

    def fetch_balance(self, params: dict | None = None) -> dict[str, dict[str, float]]:
        del params
        return self.get_balances()

    def fetch_positions(
        self, symbols: list[str] | None = None, params: dict | None = None
    ) -> list[dict[str, Any]]:
        del params
        self.ensure_connected()
        try:
            throttle()
            positions = self.ib.positions()
        except Exception as exc:
            raise TemporaryError(f"positions() failed: {exc}") from exc

        allowed = None
        if symbols:
            allowed = {self._normalise_pair_string(s) for s in symbols}

        result: list[dict[str, Any]] = []
        for pos in positions:
            base = str(getattr(pos.contract, "symbol", "") or "").upper()
            quote = str(getattr(pos.contract, "currency", "") or "").upper()
            if not base or not quote:
                continue
            pair = f"{base}/{quote}"
            if allowed is not None and pair not in allowed:
                continue
            qty = self._as_float(getattr(pos, "position", 0.0), 0.0)
            if abs(qty) < 1e-12:
                continue
            side = "short" if qty < 0 else "long"
            quantity = abs(qty)
            avg_cost = self._as_float(getattr(pos, "avgCost", 0.0), 0.0)
            result.append(
                {
                    "symbol": pair,
                    "amount": quantity,
                    "contracts": quantity,
                    "contractSize": 1.0,
                    "side": side,
                    "collateral": quantity * avg_cost if avg_cost else 0.0,
                    "initialMargin": 0.0,
                    "leverage": 1.0,
                    "unrealizedPnl": 0.0,
                    "currentPrice": 0.0,
                    "avgEntryPrice": avg_cost,
                    "info": {
                        "account": getattr(pos, "account", None),
                        "position": qty,
                        "avgCost": avg_cost,
                    },
                }
            )
        return result

    # ------------------------------------------------------------------
    # Pricing / ticker
    # ------------------------------------------------------------------

    def fetch_ticker(self, pair: str, params: dict | None = None) -> dict[str, Any]:
        del params
        pair = self._normalise_pair_string(pair)
        self.ensure_connected()
        contract = self._make_contract(pair)
        throttle()
        try:
            # snapshot=True: IB delivers one quote and ends the subscription.
            # Do NOT cancelMktData afterward — that races the auto-cleanup and
            # produces Error 300 ("Can't find EId with tickerId").
            ticker = self.ib.reqMktData(contract, genericTickList="", snapshot=True)
            deadline = time.time() + 5.0
            while time.time() < deadline:
                bid = getattr(ticker, "bid", None)
                ask = getattr(ticker, "ask", None)
                if (
                    bid is not None
                    and ask is not None
                    and not math.isnan(float(bid))
                    and not math.isnan(float(ask))
                    and float(bid) > 0
                    and float(ask) > 0
                ):
                    break
                self.ib.sleep(0.1)
            else:
                raise PricingError(f"Timeout waiting for live quote on {pair}.")

            bid_f = float(ticker.bid)
            ask_f = float(ticker.ask)
            last = (bid_f + ask_f) / 2.0
            ts_ms = int(time.time() * 1000)
            return {
                "symbol": pair,
                "timestamp": ts_ms,
                "datetime": datetime.fromtimestamp(ts_ms / 1000, UTC).isoformat(),
                "last": last,
                "close": last,
                "bid": bid_f,
                "ask": ask_f,
                "bidVolume": None,
                "askVolume": None,
                "open": None,
                "high": None,
                "low": None,
                "baseVolume": None,
                "quoteVolume": None,
                "percentage": None,
                "info": {},
            }
        except PricingError:
            raise
        except Exception as exc:
            raise TemporaryError(f"fetch_ticker {pair}: {exc}") from exc

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
        pair = self._normalise_pair_string(pair)
        key = (pair, bool(is_short))
        now = time.monotonic()
        if not refresh:
            with self._rate_cache_lock:
                cached = self._rate_cache.get(key)
            if cached and now - cached[2] <= self._rate_cache_ttl:
                return cached[0] if side == "entry" else cached[1]

        try:
            ticker = self.fetch_ticker(pair)
            bid = float(ticker.get("bid") or 0.0)
            ask = float(ticker.get("ask") or 0.0)
            last = float(ticker.get("last") or 0.0)
        except Exception as exc:
            logger.warning("Live rate failed for %s (%s); trying historical close.", pair, exc)
            last = self._fallback_historical_rate(pair)
            bid = ask = last

        if side == "entry":
            rate = (bid if is_short else ask) or last
        elif side == "exit":
            rate = (ask if is_short else bid) or last
        else:
            rate = last or ((bid + ask) / 2.0 if bid and ask else 0.0)

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
        pair = self._normalise_pair_string(pair)
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
        base = base.upper()
        quote = quote.upper()
        if base == quote:
            return 1.0
        pair = f"{base}/{quote}"
        try:
            ticker = self.fetch_ticker(pair)
            bid = float(ticker.get("bid") or 0.0)
            ask = float(ticker.get("ask") or 0.0)
            last = float(ticker.get("last") or 0.0)
            if bid > 0 and ask > 0:
                return (bid + ask) / 2.0
            if last > 0:
                return last
        except Exception as exc:
            logger.debug("Direct conversion fetch failed for %s: %s", pair, exc)

        inv = f"{quote}/{base}"
        try:
            ticker = self.fetch_ticker(inv)
            bid = float(ticker.get("bid") or 0.0)
            ask = float(ticker.get("ask") or 0.0)
            mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else float(ticker.get("last") or 0.0)
            if mid > 0:
                return 1.0 / mid
        except Exception as exc:
            raise PricingError(f"Cannot convert {base} to {quote}: {exc}") from exc
        raise PricingError(f"Cannot convert {base} to {quote}.")

    def _fallback_historical_rate(self, pair: str) -> float:
        timeframe = str(self.config.get("timeframe", "5m"))
        df = self.get_historic_ohlcv(pair, timeframe=timeframe, limit=1)
        if df.empty:
            raise PricingError(f"No historical data available for {pair}.")
        close_price = float(df.iloc[-1]["close"])
        if close_price <= 0 or math.isnan(close_price):
            raise PricingError(f"Invalid historical close for {pair}.")
        return close_price

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def create_order(  # noqa: C901
        self,
        pair: str,
        ordertype: str,
        side: str,
        amount: float,
        rate: float | None = None,
        leverage: float = 1.0,
        reduceOnly: bool = False,
        time_in_force: str = "GTC",
        params: dict | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del leverage, reduceOnly
        params = params or {}
        pair = self._normalise_pair_string(pair)
        side_text = side.lower()
        if side_text not in {"buy", "sell"}:
            raise InvalidOrderException(f"Unsupported forex order side: {side}")
        if amount <= 0:
            raise InvalidOrderException(f"Order amount must be positive: {amount}")

        # Accept Freqtrade's rate= keyword as limit price.
        if rate is None:
            rate = kwargs.get("price") or params.get("price")

        tif = self._parse_tif(
            kwargs.get("time_in_force")
            or params.get("timeInForce")
            or params.get("time_in_force")
            or time_in_force
        )
        normalized_type = ordertype.lower()
        if normalized_type not in {"market", "limit"}:
            raise InvalidOrderException(
                f"Unsupported IBKR forex order type: {ordertype}. Supported: market, limit."
            )

        qty = self.amount_to_precision(pair, amount)
        base, _quote = self._split_pair(pair)
        min_qty = self._idealpro_min_base(base)
        if qty + 1e-9 < min_qty:
            logger.warning(
                "Order qty %s for %s is below IDEALPRO minimum %s (odd lot).",
                qty,
                pair,
                min_qty,
            )

        # dry_run -> IB paper account (port 4002 / paper TWS). Always submit
        # through the API so fills come from IB's paper trading system.
        if not self.is_market_open():
            raise OperationalException("Forex market is currently closed. Order rejected.")

        self.ensure_connected()

        # Soft duplicate guard: refuse a second open order on same side/pair.
        try:
            open_orders = self.fetch_open_orders(pair)
            dups = [
                o
                for o in open_orders
                if str(o.get("side", "")).lower() == side_text and o.get("status") == "open"
            ]
            if dups:
                raise InvalidOrderException(
                    f"Duplicate in-flight {side_text} order for {pair} ({len(dups)} open)."
                )
        except InvalidOrderException:
            raise
        except Exception:
            logger.exception("Error checking existing orders for %s", pair)

        contract = self._make_contract(pair)
        try:
            throttle()
            if not self.ib.qualifyContracts(contract):
                raise InvalidOrderException(f"Contract qualification failed for {pair}.")
        except InvalidOrderException:
            raise
        except Exception as exc:
            raise OperationalException(f"Contract qualification failed for {pair}: {exc}") from exc

        action = side_text.upper()
        if normalized_type == "market":
            order: Order = MarketOrder(action, qty)
            order.tif = tif
        else:
            if rate is None or float(rate) <= 0:
                raise PricingError(f"A limit order for {pair} requires a positive rate.")
            limit_price = self.price_to_precision(pair, float(rate))
            if limit_price <= 0:
                raise InvalidOrderException(f"Invalid limit price {rate} for {pair}")
            order = LimitOrder(action, qty, limit_price)
            order.tif = tif

        client_order_id = params.get("client_order_id") or kwargs.get("client_order_id")
        if client_order_id:
            order.orderRef = str(client_order_id)

        try:
            throttle()
            trade = self.ib.placeOrder(contract, order)
        except Exception as exc:
            raise self._translate_ib_error(exc, f"create {side_text} {pair}") from exc

        # Wait briefly for acknowledgement / fill (market) / open (limit).
        deadline = time.time() + 12.0
        while time.time() < deadline and not self.is_shutting_down:
            status = str(getattr(trade.orderStatus, "status", "") or "")
            if status in {
                "Filled",
                "Cancelled",
                "Canceled",
                "ApiCancelled",
                "Inactive",
            }:
                break
            if status in {"Submitted", "PreSubmitted"} and normalized_type == "limit":
                # Limit resting is enough.
                break
            self.ib.waitOnUpdate(timeout=0.25)

        # Transient 10349: TWS may flip Cancelled -> PreSubmitted.
        self._wait_for_10349_recovery(trade, pair, qty)

        return self._normalize_trade(trade, pair, requested_rate=rate)

    def fetch_order(
        self, order_id: str, symbol: str | None = None, params: dict | None = None
    ) -> dict[str, Any]:
        del params
        if not order_id:
            raise InvalidOrderException("Order id is required.")

        pair = self._normalise_pair_string(symbol) if symbol else ""

        # Legacy local dry-run ids (from older adapter builds).
        if str(order_id).startswith("dry_run_"):
            return self._fetch_dry_order(str(order_id), pair)

        self.ensure_connected()
        try:
            oid = int(order_id)
        except (TypeError, ValueError) as exc:
            raise InvalidOrderException(f"Invalid IBKR order id: {order_id}") from exc

        for trade in self.ib.trades():
            if int(getattr(trade.order, "orderId", -1)) == oid:
                return self._normalize_trade(trade, pair or self._trade_pair(trade))

        # Not in cache — best-effort resync once and retry.
        self._safe_resync_order_cache()
        for trade in self.ib.trades():
            if int(getattr(trade.order, "orderId", -1)) == oid:
                return self._normalize_trade(trade, pair or self._trade_pair(trade))

        raise InvalidOrderException(f"IBKR order {order_id} was not found.")

    def check_order_canceled_empty(self, order: dict[str, Any]) -> bool:
        if not isinstance(order, dict):
            return False
        status = str(order.get("status") or "").lower()
        filled = float(order.get("filled") or 0.0)
        return status in {
            "canceled",
            "cancelled",
            "expired",
            "rejected",
            "inactive",
        } and math.isclose(filled, 0.0, abs_tol=1e-12)

    def cancel_order(  # noqa: C901
        self, order_id: str, pair: str = "", params: dict | None = None
    ) -> dict[str, Any]:
        del params
        if str(order_id).startswith("dry_run_"):
            with self._dry_order_lock:
                order = self._dry_orders.get(str(order_id))
                if not order:
                    raise InvalidOrderException(f"Dry-run order {order_id} not found.")
                if order["status"] == "open":
                    order["status"] = "canceled"
                    order["remaining"] = max(0.0, order["amount"] - order["filled"])
                return dict(order)

        self.ensure_connected()
        try:
            oid = int(order_id)
        except (TypeError, ValueError) as exc:
            raise InvalidOrderException(f"Invalid IBKR order id: {order_id}") from exc

        target: Trade | None = None
        for trade in self.ib.openTrades():
            if int(getattr(trade.order, "orderId", -1)) == oid:
                target = trade
                break
        if target is None:
            for trade in self.ib.trades():
                if int(getattr(trade.order, "orderId", -1)) == oid:
                    target = trade
                    break

        if target is None:
            # Already gone — return a canceled shell if possible.
            try:
                return self.fetch_order(order_id, pair)
            except InvalidOrderException:
                return {
                    "id": str(order_id),
                    "symbol": pair,
                    "type": "limit",
                    "side": "sell",
                    "amount": 0.0,
                    "filled": 0.0,
                    "remaining": 0.0,
                    "status": "canceled",
                    "price": None,
                    "average": None,
                    "cost": 0.0,
                    "fee": None,
                    "info": {},
                }

        try:
            throttle()
            self.ib.cancelOrder(target.order)
        except Exception as exc:
            message = str(exc).lower()
            if "filled" in message:
                return self._normalize_trade(target, pair or self._trade_pair(target))
            raise InvalidOrderException(f"Could not cancel IBKR order {order_id}: {exc}") from exc

        # Cancel is async — wait briefly for terminal state.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            status = str(getattr(target.orderStatus, "status", "") or "")
            if status in {"Cancelled", "Canceled", "ApiCancelled", "Filled", "Inactive"}:
                break
            self.ib.waitOnUpdate(timeout=0.25)

        return self._normalize_trade(target, pair or self._trade_pair(target))

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
        self.ensure_connected()
        result: list[dict[str, Any]] = []
        for trade in self.ib.openTrades():
            if not hasattr(trade, "contract") or not hasattr(trade, "order"):
                continue
            pair = self._trade_pair(trade)
            if symbol and pair != self._normalise_pair_string(symbol):
                continue
            normalized = self._normalize_trade(trade, pair)
            if since is not None and (normalized.get("timestamp") or 0) < since:
                continue
            result.append(normalized)
        if limit:
            result = result[: int(limit)]
        return result

    def fetch_orders(
        self,
        pair: str,
        since: int | datetime | None = None,
        params: dict | None = None,
    ) -> list[dict[str, Any]]:
        del params
        pair = self._normalise_pair_string(pair)
        self.ensure_connected()
        since_ms = None
        if since is not None:
            since_ms = int(since.timestamp() * 1000) if isinstance(since, datetime) else int(since)
        result: list[dict[str, Any]] = []
        for trade in self.ib.trades():
            tpair = self._trade_pair(trade)
            if tpair != pair:
                continue
            normalized = self._normalize_trade(trade, pair)
            if since_ms is not None and (normalized.get("timestamp") or 0) < since_ms:
                continue
            result.append(normalized)
        return result

    # ------------------------------------------------------------------
    # Fees / fills
    # ------------------------------------------------------------------

    def get_fee(self, symbol: str, now: Any = None, taker_or_maker: str | None = None) -> float:
        del symbol, now, taker_or_maker
        # IBKR commissions are account/tier specific; do not invent a crypto rate.
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
        del pair, amount, is_short, open_date, kwargs
        return 0.0

    def get_order_id_conditional(self, order: dict[str, Any]) -> str:
        if not isinstance(order, dict):
            raise InvalidOrderException("Invalid order object: expected a dictionary.")
        order_id = order.get("id")
        if not order_id:
            raise InvalidOrderException("Order object does not contain an id.")
        return str(order_id)

    def get_trades_for_order(  # noqa: C901
        self,
        order_id: str,
        symbol: str,
        since: int | datetime | dict | None = None,
        params: dict | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        del kwargs
        if isinstance(since, dict):
            params = params or since
            since = None
        del params
        symbol = self._normalise_pair_string(symbol)
        if not order_id:
            raise InvalidOrderException("Order id is required for fill lookup.")

        if str(order_id).startswith("dry_run_"):
            with self._dry_order_lock:
                order = self._dry_orders.get(str(order_id))
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

        self.ensure_connected()
        try:
            oid = int(order_id)
        except (TypeError, ValueError):
            return []

        fills: list[dict[str, Any]] = []
        for trade in self.ib.trades():
            if int(getattr(trade.order, "orderId", -1)) != oid:
                continue
            for fill in getattr(trade, "fills", []) or []:
                execution = getattr(fill, "execution", None)
                if execution is None:
                    continue
                qty = self._as_float(getattr(execution, "shares", 0.0), 0.0)
                price = self._as_float(getattr(execution, "price", 0.0), 0.0)
                if qty <= 0 or price <= 0:
                    continue
                ts = getattr(execution, "time", None)
                if isinstance(ts, datetime):
                    timestamp = int(ts.replace(tzinfo=ts.tzinfo or UTC).timestamp() * 1000)
                else:
                    timestamp = int(time.time() * 1000)
                if since is not None:
                    since_ms = (
                        int(since.timestamp() * 1000) if isinstance(since, datetime) else int(since)
                    )
                    if timestamp < since_ms:
                        continue
                fills.append(
                    {
                        "id": str(getattr(execution, "execId", f"{order_id}:{len(fills)}")),
                        "timestamp": timestamp,
                        "datetime": datetime.fromtimestamp(timestamp / 1000, UTC).isoformat(),
                        "symbol": symbol,
                        "side": str(getattr(trade.order, "action", "BUY")).lower(),
                        "price": price,
                        "amount": qty,
                        "cost": qty * price,
                        "fee": None,
                        "fees": [],
                        "order": str(order_id),
                        "info": {},
                    }
                )
        fills.sort(key=lambda item: (item["timestamp"], item["id"]))
        return fills

    def extract_cost_curr_rate(self, *args: Any, **kwargs: Any) -> tuple[float, str, float]:
        del args, kwargs
        return 0.0, "USD", 0.0

    def handle_order_fee(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    # ------------------------------------------------------------------
    # Historical OHLCV
    # ------------------------------------------------------------------

    def validate_timeframes(self, timeframes: str | list[str]) -> None:
        values = [timeframes] if isinstance(timeframes, str) else list(timeframes)
        unsupported = [tf for tf in values if tf not in self._TF_MAP]
        if unsupported:
            raise ConfigurationError(
                f"Unsupported IBKR forex timeframe(s): {', '.join(unsupported)}. "
                f"Supported: {', '.join(self._TF_MAP)}"
            )

    def ohlcv_candle_limit(
        self, timeframe: str, candle_type: str = "spot", since_ms: int | None = None
    ) -> int:
        del candle_type, since_ms
        self.validate_timeframes(timeframe)
        return int(self._ft_has.get("ohlcv_candle_limit", 1000))

    def get_historic_ohlcv(
        self,
        pair: str,
        timeframe: str | None = None,
        since: int | None = None,
        limit: int = 1000,
        params: dict | None = None,
        since_ms: int | None = None,
        is_new_pair: bool = False,
        candle_type: str = "spot",
        until_ms: int | None = None,
    ) -> pd.DataFrame:
        del is_new_pair, candle_type, params
        # Accept both (pair, timeframe, ...) and legacy (pair, since, timeframe).
        if isinstance(timeframe, (int, float)) and since is None:
            # Called as (pair, since, timeframe=...)
            since = int(timeframe)
            timeframe = None
        tf = timeframe or str(self.config.get("timeframe", "1h"))
        return self._fetch_bars_df(
            pair=pair,
            timeframe=tf,
            since_ms=since if since is not None else since_ms,
            limit=limit,
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
        del params
        df = self._fetch_bars_df(
            pair=symbol,
            timeframe=timeframe,
            since_ms=since,
            limit=limit or 500,
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

        df = self.get_historic_ohlcv(
            pair_text, timeframe, limit=self._initial_candle_limit(timeframe)
        )
        if not df.empty:
            with self._klines_lock:
                self._klines[key] = df.copy()
        return df.copy() if copy else df

    def refresh_latest_ohlcv(  # noqa: C901
        self,
        pair_list: list[Any],
        *,
        since_ms: int | None = None,
        cache: bool = True,
        drop_incomplete: bool | None = None,
    ) -> dict[tuple[str, str, CandleType], pd.DataFrame]:
        is_open, time_until_open = self.is_market_open()
        if not is_open and time_until_open > 300:
            sleep_time = time_until_open - 300
            hours, rem = divmod(int(sleep_time), 3600)
            minutes, seconds = divmod(rem, 60)
            logger.info(
                "Forex market closed; sleeping %dh %dm %ds until ~5 minutes before open.",
                hours,
                minutes,
                seconds,
            )
            # Interruptible sleep
            end = time.time() + sleep_time
            while time.time() < end and not self.is_shutting_down:
                time.sleep(min(1.0, end - time.time()))

        result: dict[tuple[str, str, CandleType], pd.DataFrame] = {}
        use_drop_incomplete = (
            self._ft_has.get("ohlcv_partial_candle", True)
            if drop_incomplete is None
            else bool(drop_incomplete)
        )

        for item in pair_list or []:
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
            key = (pair_text, timeframe, candle_type)

            try:
                fetch_since = since_ms
                with self._klines_lock:
                    existing = self._klines.get(key)
                    existing_last_ms = (
                        self._datetime_to_ms(existing["date"].iloc[-1])
                        if existing is not None and not existing.empty
                        else None
                    )
                if fetch_since is None and existing_last_ms is not None:
                    fetch_since = max(
                        0, existing_last_ms - self._timeframe_seconds(timeframe) * 1000
                    )

                limit = (
                    self._initial_candle_limit(timeframe)
                    if existing is None or existing.empty
                    else 10
                )
                df = self.get_historic_ohlcv(
                    pair_text,
                    timeframe,
                    since=fetch_since,
                    limit=limit,
                )
                if df.empty:
                    if existing is not None and not existing.empty:
                        result[key] = existing.copy()
                    continue

                if existing is not None and not existing.empty:
                    df = pd.concat([existing, df], ignore_index=True)
                    df = (
                        df.sort_values("date")
                        .drop_duplicates("date", keep="last")
                        .reset_index(drop=True)
                    )

                if use_drop_incomplete and not df.empty:
                    now = pd.Timestamp.now(tz="UTC")
                    tf_ms = self._timeframe_seconds(timeframe) * 1000
                    last_ms = self._datetime_to_ms(df["date"].iloc[-1])
                    current_bucket = (int(now.timestamp() * 1000) // tf_ms) * tf_ms
                    if last_ms >= current_bucket and len(df) > 1:
                        df = df.iloc[:-1].copy()

                if cache:
                    with self._klines_lock:
                        self._klines[key] = df.copy()
                result[key] = df.copy()
            except Exception as exc:
                logger.warning(
                    "Unable to refresh %s %s candles: %s",
                    key,
                    timeframe,
                    exc,
                )
        return result

    def _fetch_bars_df(
        self,
        pair: str,
        timeframe: str,
        since_ms: int | None,
        limit: int,
        until_ms: int | None = None,
    ) -> pd.DataFrame:
        self.validate_timeframes(timeframe)
        pair = self._normalise_pair_string(pair)
        requested_limit = max(1, min(int(limit), 3000))
        bar_size, _minutes = self._TF_MAP[timeframe]

        if since_ms is not None:
            duration_seconds = max(
                60,
                int(time.time() - (int(since_ms) / 1000.0)) + 60,
            )
        else:
            duration_seconds = self._timeframe_seconds(timeframe) * requested_limit
            duration_seconds = max(duration_seconds, 3600)

        duration_str = self._format_ib_duration(duration_seconds)
        contract = self._make_contract(pair)

        try:
            self.ensure_connected()
            throttle()
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime=""
                if until_ms is None
                else datetime.fromtimestamp(until_ms / 1000, UTC),
                durationStr=duration_str,
                barSizeSetting=bar_size,
                whatToShow="MIDPOINT",
                useRTH=False,
                formatDate=1,
            )
        except Exception as exc:
            raise self._translate_ib_error(exc, f"fetch bars {pair} {timeframe}") from exc

        if not bars:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        df = util.df(bars)
        if df is None or df.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        df = df.rename(columns={"date": "date"})
        keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep].copy()
        df["date"] = pd.to_datetime(df["date"], utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "volume" in df.columns:
            df.loc[df["volume"] < 0, "volume"] = 0.0
        df = df.dropna(subset=["date", "open", "high", "low", "close"])

        if since_ms is not None:
            df = df[df["date"] >= pd.Timestamp(self._ms_to_datetime(int(since_ms)))]
        if until_ms is not None:
            df = df[df["date"] <= pd.Timestamp(self._ms_to_datetime(int(until_ms)))]

        df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        if len(df) > requested_limit:
            df = df.iloc[-requested_limit:].copy()
        return df[["date", "open", "high", "low", "close", "volume"]]

    # ------------------------------------------------------------------
    # Trades (public prints — limited on IBKR FX)
    # ------------------------------------------------------------------

    def fetch_trades(
        self,
        pair: str,
        since: int | None = None,
        since_ms: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ) -> list[dict[str, Any]]:
        del pair, since, since_ms, limit, params
        # IDEALPRO does not expose a reliable public trade tape via TWS API
        # comparable to crypto exchanges.
        return []

    # ------------------------------------------------------------------
    # Market clock (forex 24/5 approximation)
    # ------------------------------------------------------------------

    def is_market_open(self) -> tuple[bool, float]:
        """Return (is_open, seconds_until_open).

        Approximation: open Sunday 22:00 UTC -> Friday 22:00 UTC.
        Holiday closures are not modeled.
        """
        now = datetime.now(UTC)
        day = now.weekday()  # Mon=0 … Sun=6
        t = now.time()

        open_tod = datetime.strptime("22:00", "%H:%M").time()
        close_tod = datetime.strptime("22:00", "%H:%M").time()

        if day == 5:  # Saturday
            # Next open: Sunday 22:00
            days_ahead = 1
            next_open = datetime.combine(
                (now + timedelta(days=days_ahead)).date(), open_tod, tzinfo=UTC
            )
            wait = max(0.0, (next_open - now).total_seconds())
            self._market_open_cache = (False, wait)
            return False, wait
        if day == 6:  # Sunday
            if t >= open_tod:
                self._market_open_cache = (True, 0.0)
                return True, 0.0
            next_open = datetime.combine(now.date(), open_tod, tzinfo=UTC)
            wait = max(0.0, (next_open - now).total_seconds())
            self._market_open_cache = (False, wait)
            return False, wait
        if day == 4:  # Friday
            if t < close_tod:
                self._market_open_cache = (True, 0.0)
                return True, 0.0
            # Next open Sunday 22:00
            next_open = datetime.combine((now + timedelta(days=2)).date(), open_tod, tzinfo=UTC)
            wait = max(0.0, (next_open - now).total_seconds())
            self._market_open_cache = (False, wait)
            return False, wait

        # Mon-Thu
        self._market_open_cache = (True, 0.0)
        return True, 0.0

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_config(self, config: dict) -> None:
        """Validate trading mode / timeframe only.

        Do **not** call blocking IB RPCs here (especially ``reqCurrentTime``).
        IB Gateway/TWS can drop or never answer that request, which hangs the
        Freqtrade process right after "Validating configuration ...".
        Connectivity was already established in ``__init__`` when required.
        """
        self.validate_trading_mode_and_margin_mode(
            config.get("trading_mode", "spot"),
            config.get("margin_mode"),
            allow_none_margin_mode=True,
        )
        tf = config.get("timeframe", "1h")
        if tf:
            self.validate_timeframes([tf])

        # Paper (dry_run) and live both need a working TWS/Gateway session.
        if (
            config.get("runmode") in ("dry_run", "live", "worker", None)
            and not self.ib.isConnected()
        ):
            mode = "paper" if self.dry_run else "live"
            raise OperationalException(
                f"IBKR is not connected; cannot start {mode} trading. "
                f"Is TWS/IB Gateway running on {self.host}:{self.port}?"
            )

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
        tif: str,
    ) -> dict[str, Any]:
        amount = self.amount_to_precision(pair, amount)
        if amount <= 0:
            raise InvalidOrderException(f"Order quantity becomes zero for {pair}.")
        if ordertype == "limit" and (rate is None or float(rate) <= 0):
            raise PricingError(f"A dry-run limit order for {pair} requires a positive rate.")

        order_id = f"dry_run_{side}_{uuid4()}"
        now_ms = int(time.time() * 1000)
        try:
            requested_rate = float(
                rate
                if rate is not None
                else self.get_rate(
                    pair,
                    side="entry" if side == "buy" else "exit",
                    is_short=False,
                    refresh=True,
                )
            )
        except Exception:
            requested_rate = float(rate or 0.0)

        if ordertype == "limit":
            requested_rate = self.price_to_precision(pair, requested_rate)

        status = "closed" if ordertype == "market" else "open"
        filled = amount if status == "closed" else 0.0
        average = requested_rate if filled else None
        order = {
            "id": order_id,
            "symbol": pair,
            "type": ordertype,
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
            "cost": filled * (average or requested_rate or 0.0),
            "fee": None,
            "timeInForce": tif,
            "info": {"dry_run": True, "tif": tif},
        }
        with self._dry_order_lock:
            self._dry_orders[order_id] = order
        return dict(order)

    def _fetch_dry_order(self, order_id: str, pair: str) -> dict[str, Any]:
        with self._dry_order_lock:
            order = self._dry_orders.get(order_id)

        if not order and order_id.startswith("dry_run_"):
            try:
                from freqtrade.persistence import Order as PersistedOrder

                db_order = PersistedOrder.order_by_id(
                    order_id, self._normalise_pair_string(pair) if pair else None
                )
                if db_order is not None:
                    order = db_order.to_ccxt_object()
                    order["type"] = (order.get("type") or "limit").lower()
                    order["side"] = (
                        order.get("side") or getattr(db_order, "ft_order_side", None) or "buy"
                    ).lower()
                    order["amount"] = self._as_float(
                        order.get("amount") or getattr(db_order, "ft_amount", 0.0)
                    )
                    order["filled"] = self._as_float(order.get("filled"))
                    order["remaining"] = self._as_float(
                        order.get("remaining")
                        if order.get("remaining") is not None
                        else order["amount"] - order["filled"]
                    )
                    ft_price = getattr(db_order, "ft_price", 0.0)
                    order["price"] = (
                        self._as_float(order.get("price"))
                        if order.get("price") is not None
                        else self._as_float(ft_price if ft_price is not None else 0.0)
                    )
                    order["average"] = (
                        self._as_float(order["average"])
                        if order.get("average") is not None
                        else None
                    )
                    order.setdefault(
                        "symbol",
                        self._normalise_pair_string(pair) if pair else order.get("symbol"),
                    )
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

        if order["status"] == "open" and order["type"] == "limit" and pair:
            try:
                ticker = self.fetch_ticker(pair)
            except Exception:
                return dict(order)
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
    # IB order helpers
    # ------------------------------------------------------------------

    def _wait_for_10349_recovery(self, trade: Trade, pair: str, amount: float) -> None:
        if not any(
            getattr(entry, "errorCode", 0) == 10349 for entry in getattr(trade, "log", []) or []
        ):
            return
        order_id = getattr(trade.order, "orderId", 0)
        logger.warning(
            "IBKR order %s for %s reported Cancelled with error 10349; waiting for final state.",
            order_id,
            pair,
        )
        deadline = time.time() + 10.0
        while time.time() < deadline and not self.is_shutting_down:
            self.ib.waitOnUpdate(timeout=0.25)
            status = str(getattr(trade.orderStatus, "status", "") or "")
            filled = self._as_float(getattr(trade.orderStatus, "filled", 0.0), 0.0)
            if filled > 0 or status in {"PreSubmitted", "Submitted", "Filled"}:
                logger.info(
                    "IBKR order %s for %s recovered from 10349 (status=%s filled=%s/%s).",
                    order_id,
                    pair,
                    status,
                    filled,
                    amount,
                )
                return

    def _normalize_trade(
        self,
        trade: Trade,
        pair: str,
        requested_rate: float | None = None,
    ) -> dict[str, Any]:
        status_raw = str(getattr(trade.orderStatus, "status", "unknown") or "unknown")
        status_key = status_raw.lower().replace(" ", "")
        normalized_status = self._ft_has["order_statuses"].get(status_key, "open")

        filled = self._as_float(getattr(trade.orderStatus, "filled", 0.0), 0.0)
        amount = self._as_float(getattr(trade.order, "totalQuantity", 0.0), filled)
        remaining = max(0.0, amount - filled)
        avg = self._as_float(getattr(trade.orderStatus, "avgFillPrice", 0.0), 0.0)
        limit_price = self._as_float(getattr(trade.order, "lmtPrice", 0.0), 0.0)
        requested_price = requested_rate or limit_price or avg or None

        if filled >= amount > 0 and normalized_status not in {"canceled", "rejected"}:
            normalized_status = "closed"
            remaining = 0.0

        side = str(getattr(trade.order, "action", "BUY") or "BUY").lower()
        order_type = str(getattr(trade.order, "orderType", "LMT") or "LMT").lower()
        if order_type in {"mkt", "market"}:
            order_type = "market"
        elif order_type in {"lmt", "limit"}:
            order_type = "limit"

        symbol = pair or self._trade_pair(trade)
        cost = filled * avg if avg > 0 else 0.0
        tif = getattr(trade.order, "tif", None)
        oid = str(getattr(trade.order, "orderId", "") or "")

        return {
            "id": oid,
            "clientOrderId": getattr(trade.order, "orderRef", None),
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "price": float(requested_price) if requested_price else None,
            "average": avg or None,
            "amount": amount,
            "filled": filled,
            "remaining": remaining,
            "status": normalized_status,
            "timestamp": None,
            "lastTradeTimestamp": None,
            "datetime": None,
            "cost": cost,
            "fee": None,
            "timeInForce": tif,
            "info": {"ib_status": status_raw},
        }

    def _trade_pair(self, trade: Trade) -> str:
        base = str(getattr(trade.contract, "symbol", "") or "").upper()
        quote = str(getattr(trade.contract, "currency", "") or "").upper()
        if base and quote:
            return f"{base}/{quote}"
        return ""

    def _make_contract(self, pair: str) -> Forex:
        base, quote = self._split_pair(pair)
        return Forex(symbol=base, currency=quote, exchange="IDEALPRO")

    def _get_contract_meta(self, pair: str) -> dict[str, float] | None:
        pair = self._normalise_pair_string(pair)
        with self._contract_meta_lock:
            cached = self._contract_meta.get(pair)
            if cached:
                return cached
        if not self.ib.isConnected():
            return None
        try:
            contract = self._make_contract(pair)
            throttle()
            details = self.ib.reqContractDetails(contract)
            if not details:
                return None
            detail = details[0]
            min_tick = self._as_float(getattr(detail, "minTick", 0.0), 0.0)
            meta = {
                "min_tick": min_tick if min_tick > 0 else self._default_price_tick(pair),
            }
            with self._contract_meta_lock:
                self._contract_meta[pair] = meta
                # Also refresh markets precision when possible.
                if pair in self._markets and meta["min_tick"] > 0:
                    self._markets[pair]["precision"]["price"] = meta["min_tick"]
                    self._markets[pair]["limits"]["price"]["min"] = meta["min_tick"]
            return meta
        except Exception as exc:
            logger.debug("reqContractDetails failed for %s: %s", pair, exc)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
    def _round_down_increment(value: float, increment: float) -> float:
        if increment <= 0:
            return float(value)
        units = math.floor((float(value) / float(increment)) + 1e-12)
        result = units * float(increment)
        return round(result, 12)

    @staticmethod
    def _normalise_pair_string(pair: Any) -> str:
        if isinstance(pair, tuple):
            if not pair:
                raise ConfigurationError("Empty pair tuple received")
            pair = pair[0]
        if not isinstance(pair, str):
            raise ConfigurationError(
                f"Forex pair must be a string or Freqtrade pair tuple, got {pair!r}"
            )
        return pair.replace(":USD", "").upper() if "/" in str(pair) else str(pair).upper()

    @classmethod
    def _split_pair(cls, pair: Any) -> tuple[str, str]:
        text = cls._normalise_pair_string(pair)
        if "/" not in text:
            raise ConfigurationError(f"Forex pair must be BASE/QUOTE, got {pair}")
        base, quote = text.split("/", 1)
        base = base.strip().upper()
        quote = quote.strip().upper()
        if len(base) < 3 or len(quote) < 3:
            raise ConfigurationError(f"Invalid forex pair currencies: {pair}")
        return base, quote

    def _parse_tif(self, value: str | None) -> str:
        key = str(value or "GTC").lower()
        try:
            return self._TIF_MAP[key]
        except KeyError as exc:
            raise ConfigurationError(
                f"Unsupported IBKR time-in-force {value}. Supported: {', '.join(self._TIF_MAP)}"
            ) from exc

    @classmethod
    def _idealpro_min_base(cls, base: str) -> float:
        return float(_IDEALPRO_MIN_BASE.get(base.upper(), 25_000.0))

    @classmethod
    def _default_price_tick(cls, pair: str) -> float:
        base, quote = cls._split_pair(pair)
        if base == "JPY" or quote == "JPY":
            return _DEFAULT_PRICE_TICK.get("JPY", 0.001)
        return 0.00001

    def _initial_candle_limit(self, timeframe: str) -> int:
        startup = int(self.config.get("startup_candle_count", 0) or 0)
        return max(200, startup + 50, 500 if timeframe == "1m" else 200)

    def _timeframe_seconds(self, timeframe: str) -> int:
        _bar, minutes = self._TF_MAP[timeframe]
        return int(minutes) * 60

    @staticmethod
    def _format_ib_duration(seconds: int) -> str:
        if seconds <= 0:
            return "3600 S"
        if seconds > 31_536_000:
            years = math.ceil(seconds / 31_536_000)
            return f"{years} Y"
        if seconds > 2_592_000:
            months = math.ceil(seconds / 2_592_000)
            return f"{months} M"
        if seconds > 604_800:
            weeks = math.ceil(seconds / 604_800)
            return f"{weeks} W"
        if seconds > 86_400:
            days = math.ceil(seconds / 86_400)
            return f"{days} D"
        return f"{max(60, int(seconds))} S"

    def _translate_ib_error(
        self, error: Exception, context: str
    ) -> ExchangeError | OperationalException:
        message = str(error)
        lower = message.lower()
        if any(token in lower for token in ("insufficient", "buying power", "not enough")):
            return InsufficientFundsError(f"{context}: {message}")
        if any(
            token in lower
            for token in (
                "not found",
                "order not found",
                "no security definition",
                "invalid contract",
            )
        ):
            return InvalidOrderException(f"{context}: {message}")
        if any(
            token in lower
            for token in (
                "timeout",
                "temporarily",
                "pacing",
                "rate limit",
                "too many requests",
                "connection",
                "not connected",
                "162",
            )
        ):
            return TemporaryError(f"{context}: {message}")
        return OperationalException(f"{context}: {message}")


__all__ = ["Interactivebrokers"]
