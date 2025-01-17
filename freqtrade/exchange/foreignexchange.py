# foreignexchange.py
import logging


logger = logging.getLogger(__name__)


class Foreignexchange:
    def __init__(self, config, validate=True, exchange_config=None, load_leverage_tiers=False):
        self.config = config
        self.validate = validate
        self.exchange_config = exchange_config
        self.load_leverage_tiers = load_leverage_tiers
        logger.info("Foreignexchange retrieved successfully.")

    @property
    def name(self):
        return "foreignexchange"

    def close(self):
        pass

    def exchange_has(self, method):
        """
        Check if the exchange supports a specific method.
        """
        if method == "fetchTickers" and not hasattr(self, method):
            raise ValueError(
                "Exchange does not support dynamic whitelist in this configuration."
                "Please edit your config and either remove VolumePairList, or switch"
                " to using candles, and restart the bot."
            )
        return hasattr(self, method)
