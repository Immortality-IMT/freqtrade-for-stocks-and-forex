import numpy as np
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class TestIMT(IStrategy):
    # Strategy configuration
    minimal_roi = {}
    stoploss = -1
    trailing_stop = False
    timeframe = "5m"
    use_entry_signal = True
    use_exit_signal = True
    entry_tag = "enter"
    exit_tag = "exit"

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_tag"] = ""
        dataframe["exit_tag"] = ""
        dataframe["enter_tag"] = dataframe["enter_tag"].astype(object)
        dataframe["exit_tag"] = dataframe["exit_tag"].astype(object)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe.loc[dataframe.index[-1], "enter_long"] = 1
        dataframe["enter_tag"] = ""
        dataframe.loc[dataframe.index[-1], "enter_tag"] = "force-entry"

        """
        # Randomize only for last row
        dataframe["exit_long"] = 0
        if np.random.rand() < 0.5:
            dataframe.iloc[-1, dataframe.columns.get_loc("enter_long")] = 1
            dataframe.iloc[-1, dataframe.columns.get_loc("enter_tag")]  = "random-exit"
        else:
            dataframe.iloc[-1, dataframe.columns.get_loc("enter_tag")]  = ""
        """
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        if np.random.rand() < 0.5:
            dataframe.iloc[-1, dataframe.columns.get_loc("exit_long")] = 1
            dataframe.iloc[-1, dataframe.columns.get_loc("exit_tag")] = "random-exit"
        else:
            dataframe.iloc[-1, dataframe.columns.get_loc("exit_tag")] = ""
        return dataframe
