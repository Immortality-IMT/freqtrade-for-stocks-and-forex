import numpy as np
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class TestIB(IStrategy):
    # Strategy configuration
    minimal_roi = {}
    stoploss = -1
    trailing_stop = False
    timeframe = "1m"
    use_exit_signal = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Use numpy for random generation
        dataframe["enter_long"] = np.where(np.random.rand(len(dataframe)) < 0.5, 1, 0)
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Use numpy for random generation
        dataframe["exit_long"] = np.where(np.random.rand(len(dataframe)) < 0.5, 1, 0)
        return dataframe
