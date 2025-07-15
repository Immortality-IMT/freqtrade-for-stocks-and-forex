import logging

from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy


class TestIMT(IStrategy):
    minimal_roi = {}
    stoploss = -1
    trailing_stop = False
    exit_profit_only = False
    timeframe = "5m"
    use_entry_signal = True
    use_exit_signal = True
    entry_tag = "enter"
    exit_tag = "exit"

    logger = logging.getLogger(__name__)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Initialize tags to object dtype to allow string labels
        dataframe["enter_tag"] = dataframe.get("enter_tag", "").astype(object)
        dataframe["exit_tag"] = dataframe.get("exit_tag", "").astype(object)
        self.logger.debug(
            f"DataFrame in populate_indicators:\n"
            f"{dataframe[['date', 'open', 'high', 'low', 'close']].tail(5)}"
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = dataframe["enter_tag"].astype(object)
        pair = metadata.get("pair", "IMT/BNB")

        # Enter only if there are no open trades for this pair
        open_trades = Trade.get_trades_proxy(pair=pair, is_open=True)
        if not open_trades:
            # No open trades, mark a forced entry on the latest candle
            last_idx = dataframe.index[-1]
            dataframe.at[last_idx, "enter_long"] = 1
            dataframe.at[last_idx, "enter_tag"] = "force-entry"
            self.logger.debug(f"No open trades for {pair}, forcing entry at idx {last_idx}")
        else:
            self.logger.debug(f"Existing open trade(s) for {pair}, skipping entry")

        self.logger.debug(
            f"populate_entry_trend:"
            f"enter_long[-1]={dataframe['enter_long'].iloc[-1]},"
            f"enter_tag[-1]={dataframe['enter_tag'].iloc[-1]}"
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = dataframe["exit_tag"].astype(object)
        pair = metadata.get("pair", "IMT/BNB")

        # Exit only if there is at least one open trade for this pair
        open_trades = Trade.get_trades_proxy(pair=pair, is_open=True)
        if open_trades:
            # Mark exit on the latest candle
            last_idx = dataframe.index[-1]
            dataframe.at[last_idx, "exit_long"] = 1
            dataframe.at[last_idx, "exit_tag"] = "force-sell"
            self.logger.debug(f"Open trade detected for {pair}, forcing exit at idx {last_idx}")
        else:
            self.logger.debug(f"No open trades for {pair}, skipping exit")

        self.logger.debug(
            f"populate_exit_trend:"
            f"exit_long[-1]={dataframe['exit_long'].iloc[-1]},"
            f"exit_tag[-1]={dataframe['exit_tag'].iloc[-1]}"
        )
        return dataframe
