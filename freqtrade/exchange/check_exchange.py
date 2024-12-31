import logging

from freqtrade.constants import Config
from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import available_exchanges, is_exchange_known_ccxt, validate_exchange
from freqtrade.exchange.common import MAP_EXCHANGE_CHILDCLASS, SUPPORTED_EXCHANGES


logger = logging.getLogger(__name__)


def check_exchange(config: Config, check_for_bad: bool = True) -> bool:
    """
    Check if the exchange name in the config file is supported by Freqtrade
    :param check_for_bad: if True, check the exchange against the list of known 'bad'
                          exchanges
    :return: False if exchange is 'bad', i.e. is known to work with the bot with
             critical issues or does not work at all, crashes, etc. True otherwise.
             raises an exception if the exchange if not supported by ccxt
             and thus is not known for the Freqtrade at all.
    """

    if config["runmode"] in [
        RunMode.PLOT,
        RunMode.UTIL_NO_EXCHANGE,
        RunMode.OTHER,
    ] and not config.get("exchange", {}).get("name"):
        # Skip checking exchange in plot mode, since it requires no exchange
        return True
    logger.info("Checking exchange...")

    exchange = config.get("exchange", {}).get("name", "").lower()
    if not exchange:
        raise OperationalException(
            f"This command requires a configured exchange. You should either use "
            f"`--exchange <exchange_name>` or specify a configuration file via `--config`.\n"
            f"The following exchanges are available for Freqtrade: "
            f'{", ".join(available_exchanges())}'
        )

    if MAP_EXCHANGE_CHILDCLASS.get(exchange, exchange) in SUPPORTED_EXCHANGES:
        if is_exchange_known_ccxt(exchange):
            logger.info(
                f"The {exchange.capitalize()} exchange has been recognized "
                f"and is compatible with ccxt."
            )
            logger.info(
                f"Exchange {exchange} is officially supported by the Freqtrade development team."
            )
        else:
            logger.warning(
                f"The {exchange.capitalize()} exchange is recognized by Freqtrade "
                f"but not compatible with ccxt. Experimental!!!"
            )
    else:
        if is_exchange_known_ccxt(exchange):
            logger.warning(
                f"The {exchange.capitalize()} exchange is not recognized by Freqtrade "
                f"but is compatible with ccxt. "
                f"Not officially supported by the Freqtrade development team. "
                f"It may work flawlessly (please report back) or have serious issues. "
                f"Use it at your own discretion."
            )
        else:
            raise OperationalException(
                f"Exchange '{exchange}' is not recognized by Freqtrade and not "
                f"compatible with ccxt, and therefore not available for the bot.\n"
                f"The following exchanges are available for Freqtrade: "
                f'{", ".join(available_exchanges())}'
            )

    if is_exchange_known_ccxt(exchange):
        valid, reason, _ = validate_exchange(exchange)
        if not valid:
            if check_for_bad:
                raise OperationalException(
                    f'Exchange "{exchange}"  will not work with Freqtrade. ' f"Reason: {reason}"
                )
            else:
                logger.warning(
                    f'Exchange "{exchange}"  will not work with Freqtrade. Reason: {reason}'
                )

    return True
