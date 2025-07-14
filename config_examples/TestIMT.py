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
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Use numpy for random generation
        dataframe["enter_long"] = np.random.rand(len(dataframe)) < 0.5
        dataframe["enter"] = np.random.rand(len(dataframe)) < 0.5
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Use numpy for random generation
        dataframe["exit_long"] = np.random.rand(len(dataframe)) < 0.5
        dataframe["exit"] = np.random.rand(len(dataframe)) < 0.5
        return dataframe
