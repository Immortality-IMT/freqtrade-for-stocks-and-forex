# oanda.py
# This file contains the implementation for the Oanda Foreign exchange integration in Freqtrade.

import logging

from freqtrade.exchange.exchange_types import FtHas
from freqtrade.exchange.foreignexchange import Foreignexchange


logger = logging.getLogger(__name__)


class Oanda(Foreignexchange):
    """
    Oanda exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.
    """

    _ft_has: FtHas = {
        "ohlcv_candle_limit": 1000,
    }