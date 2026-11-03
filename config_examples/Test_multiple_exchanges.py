import logging
from datetime import datetime

from pandas import DataFrame

from freqtrade.persistence import LocalTrade, Trade
from freqtrade.strategy import IStrategy


class Test_multiple_exchanges(IStrategy):
    """
    Test strategy for allow_multiple_positions + allow_multiple_exchanges.

    Buy: a signal on every candle. position_cooldown_minutes spaces the trades out and the bot
    spreads them over the exchanges (A, B, C, A, B, C, ...).

    Sell: as soon as a pair holds one trade per exchange, it is time to sell.
      sell_all_at_once = True : the normal exit signal (exit_long) sells every open trade of
                                the pair now, each one on its own exchange.
      sell_all_at_once = False: round robin. Only the oldest trade is sold (first in, first
                                out) through custom_exit(), then the next one on the next round.
    """

    minimal_roi = {}
    stoploss = -1
    timeframe = "5m"
    can_short = False

    # Freqtrade only uses exit_long and calls custom_exit() while this is True.
    use_exit_signal = True

    # The one switch of this test strategy: sell all open trades of a pair now, or one by one.
    sell_all_at_once = False

    logger = logging.getLogger(__name__)

    def _exchange_count(self) -> int:
        """Exchanges a pair can be spread over (1 unless allow_multiple_exchanges is on)."""
        if not self.config.get("allow_multiple_exchanges", False):
            return 1
        return 1 + len(self.config.get("additional_exchanges", []))

    def _open_trades(self, pair: str) -> list[LocalTrade]:
        return [t for t in Trade.get_trades_proxy(pair=pair, is_open=True) if t.has_open_position]

    def _time_to_sell(self, pair: str) -> bool:
        """The pair holds one trade per exchange."""
        return len(self._open_trades(pair)) >= self._exchange_count()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # The exchange's OHLCV cache can hand back float tag columns left over from earlier runs,
        # and Freqtrade then fails writing "" into them. Make sure they can hold strings.
        for col in ("enter_tag", "exit_tag"):
            if col in dataframe.columns:
                dataframe[col] = dataframe[col].astype(object)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = None
        # Freqtrade ignores an exit signal on a candle that also has an entry signal
        # (exit_ and not enter), so there is no entry while it is time to sell.
        if not self._time_to_sell(metadata["pair"]):
            last = dataframe.index[-1]
            dataframe.loc[last, "enter_long"] = 1
            dataframe.loc[last, "enter_tag"] = "test-entry"
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = None
        # Sell all now: the pair-wide exit signal closes every open trade of the pair.
        if self.sell_all_at_once and self._time_to_sell(metadata["pair"]):
            last = dataframe.index[-1]
            dataframe.loc[last, "exit_long"] = 1
            dataframe.loc[last, "exit_tag"] = "sell-all"
        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        # Round robin: called for every open trade, only the oldest one is sold.
        if self.sell_all_at_once or not self._time_to_sell(pair):
            return None
        oldest = min(self._open_trades(pair), key=lambda t: t.open_date_utc)
        if trade.id == oldest.id:
            self.logger.info(f"Selling oldest trade {trade.id} of {pair} on {trade.exchange}")
            return "oldest-out"
        return None
