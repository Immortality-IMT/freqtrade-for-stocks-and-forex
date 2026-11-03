import logging
from datetime import datetime, timedelta

from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy


class TestIMT(IStrategy):
    """
    Test strategy for allow_multiple_positions + allow_multiple_exchanges.

    - Entry: signal on every new candle; confirm_trade_entry() caps the stack per pair.
      The bot (not the strategy) picks the exchange, round-robin.
    - Exit: no pair-wide exit signal. custom_exit() is evaluated per trade and only the
      OLDEST filled trade of the pair is closed once it has been held long enough.
      The sell order goes to that trade's own exchange automatically.
    """

    minimal_roi = {}
    stoploss = -1
    trailing_stop = False
    exit_profit_only = False
    timeframe = "5m"
    can_short = False

    # A pair-wide exit signal would close every leg at once -> decide per trade instead.
    use_exit_signal = False

    # Test knobs
    max_stack = 6  # max open trades per pair (use > number of exchanges to see wrap-around)
    hold_minutes = 60  # oldest trade is sold after this long

    logger = logging.getLogger(__name__)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Constant signal on the newest candle. Do NOT look at Trade here: this runs once per
        # candle and the result is cached, so the trade count would be stale.
        # The bot's 15 min cooldown (allow_multiple_positions) spaces the entries out.
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = None
        last = dataframe.index[-1]
        dataframe.loc[last, "enter_long"] = 1
        dataframe.loc[last, "enter_tag"] = "test-entry"
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = None
        return dataframe

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        # Evaluated at order time with fresh DB state, so the cap is accurate.
        open_trades = Trade.get_trades_proxy(pair=pair, is_open=True)
        if len(open_trades) >= self.max_stack:
            self.logger.info(f"{pair}: stack full ({len(open_trades)}/{self.max_stack})")
            return False
        held = sorted(t.exchange for t in open_trades)
        self.logger.info(f"{pair}: entering, open on exchanges so far: {held}")
        return True

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        # Called for every open trade on every loop. Only the oldest filled trade may exit.
        candidates = [
            t for t in Trade.get_trades_proxy(pair=pair, is_open=True) if t.has_open_position
        ]
        if not candidates:
            return None
        oldest = min(candidates, key=lambda t: t.open_date_utc)
        if trade.id != oldest.id:
            return None
        if current_time - trade.open_date_utc < timedelta(minutes=self.hold_minutes):
            return None
        self.logger.info(f"Exiting oldest trade {trade.id} of {pair} on {trade.exchange}")
        return "fifo-exit"
