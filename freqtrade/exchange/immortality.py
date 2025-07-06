from typing import Dict, List, Optional, Any
import requests
import time
import logging
from freqtrade.data.converter import ohlcv_to_dataframe
from web3 import Web3
from web3.exceptions import InvalidAddress, TimeExhausted
from freqtrade.exchange.stockexchange import Stockexchange
#from freqtrade.exchange import Exchange # Calls ccxt and we have to bypass it to use a custom token
from freqtrade.exceptions import ExchangeError, InsufficientFundsError, OperationalException
from freqtrade.exchange.common import retrier


# Constants
BSC_RPC_URL = "https://bsc-dataseed.bnbchain.org"
DEFAULT_WALLET = "<YOUR_WALLET_ADDRESS>"
DEFAULT_PRIVATE_KEY = "<YOUR_PRIVATE_KEY>"
IMMORTALITY_ADDR = "0x2bf2141ed175f3236903cf07de33d7324871802d"
WBNB_ADDR = "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c"
PANCAKESWAP_ROUTER_ADDR = "0x10ED43C718714eb63d5aA57B78B54704E256024E"
PAIR_ADDRESS = "0xfA56E9AbcaA45207bE5E43cF475Ee061768CA915"

# Trading parameters
IMT_DECIMALS = 8
BUY_BNB_AMOUNT = 0.005
SELL_IMT_QUANTITY = 10_000_000
RPC_SYNC_DELAY_SECONDS = 7

# ABIs
PAIR_ABI = [{"constant": True, "inputs": [], "name": "getReserves", "outputs": [
    {"internalType": "uint112", "name": "_reserve0", "type": "uint112"},
    {"internalType": "uint112", "name": "_reserve1", "type": "uint112"},
    {"internalType": "uint32", "name": "_blockTimestampLast", "type": "uint32"}
], "stateMutability": "view", "type": "function"}]
ROUTER_ABI = [
    {"name": "getAmountsOut", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "amountIn", "type": "uint256"}, {"name": "path", "type": "address[]"}],
     "outputs": [{"name": "", "type": "uint256[]"}]},
    {"name": "swapExactETHForTokensSupportingFeeOnTransferTokens", "type": "function",
     "stateMutability": "payable", "inputs": [{"name": "amountOutMin", "type": "uint256"},
     {"name": "path", "type": "address[]"}, {"name": "to", "type": "address"},
     {"name": "deadline", "type": "uint256"}], "outputs": []},
    {"name": "swapExactTokensForETHSupportingFeeOnTransferTokens", "type": "function",
     "stateMutability": "nonpayable", "inputs": [{"name": "amountIn", "type": "uint256"},
     {"name": "amountOutMin", "type": "uint256"}, {"name": "path", "type": "address[]"},
     {"name": "to", "type": "address"}, {"name": "deadline", "type": "uint256"}], "outputs": []}
]
TOKEN_ABI = [
    {"name": "balanceOf", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "owner", "type": "address"}], "outputs": [{"name": "", "type": "uint256"}]},
    {"name": "approve", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}],
     "outputs": [{"name": "", "type": "bool"}]},
    {"name": "allowance", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}],
     "outputs": [{"name": "", "type": "uint256"}]},
    {"name": "decimals", "type": "function", "stateMutability": "view",
     "inputs": [], "outputs": [{"name": "", "type": "uint8"}]}
]

# NodeReal GraphQL
# NODEREAL_GRAPHQL_URL = "https://open-platform.nodereal.io/apikey/pancakeswap-free/graphql"

def get_nodereal_graphql_url(api_key: str) -> str:
    # free-tier v2 endpoint (no trailing slash)
    return f"https://open-platform.nodereal.io/{api_key}/pancakeswap-free/graphql"

def fetch_ohlcv_nodereal(api_key: str, pair_address: str, limit: int) -> List[List]:
    """
    Query only the fields supported in free-tier v2:
      hourStartUnix, reserve0, reserve1, hourlyVolumeToken0
    And build simple OHLCV candles: [ts, price, price, price, price, volume].
    """
    query = f"""
    {{
      pairHourDatas(
        first: {limit},
        orderBy: hourStartUnix,
        orderDirection: desc,
        where: {{ pair: "{pair_address.lower()}" }}
      ) {{
        hourStartUnix
        reserve0
        reserve1
        hourlyVolumeToken0
      }}
    }}
    """
    url = get_nodereal_graphql_url(api_key)
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, json={"query": query}, headers=headers)
    resp.raise_for_status()
    data = resp.json().get("data", {}).get("pairHourDatas", [])

    candles: List[List] = []
    for entry in data:
        ts = int(entry["hourStartUnix"]) * 1000
        r0 = float(entry["reserve0"])
        r1 = float(entry["reserve1"])
        price = (r1 / r0) if r0 else 0.0
        volume = float(entry["hourlyVolumeToken0"])
        # open, high, low, close all = price
        candles.append([ts, price, price, price, price, volume])

    # reverse to oldest-first
    return list(reversed(candles))

def send_tx(w3, fn, wallet_address: str, private_key: str, value: int = 0) -> str:
    tx_params = {
        'from': wallet_address,
        'gas': 300000,
        'gasPrice': max(int(w3.eth.gas_price * 1.1), w3.to_wei('5', 'gwei')),
        'nonce': w3.eth.get_transaction_count(wallet_address),
    }
    if value:
        tx_params['value'] = value
    transaction = fn.build_transaction(tx_params)
    signed = w3.eth.account.sign_transaction(transaction, private_key=private_key)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=600)
    if receipt.status == 0:
        raise ExchangeError(f"Transaction {tx_hash.hex()} failed")
    return tx_hash.hex()

class Immortality(Stockexchange):
    _use_ccxt = False
    _ft_has = {
        "ohlcv_candle_limit": 500,
        "order_time_in_force": ["gtc"],
        "stoploss_on_exchange": False,
    }

    def __init__(self, config: Dict, **kwargs):
        super().__init__(config, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Immortality exchange")
        self.w3 = Web3(Web3.HTTPProvider(BSC_RPC_URL))
        if not self.w3.is_connected():
            raise OperationalException("Cannot connect to BSC RPC")
        self._exchange_ws = None

    @property
    def api_key(self) -> str:
        # Freqtrade puts your exchange settings under config["exchange"]
        exchange_conf = self.config.get("exchange", {})
        key = exchange_conf.get("api_key", "").strip()
        if not key:
            self.logger.error("Missing nodereal_api_key under exchange in config.json.")
        else:
            self.logger.debug(f"Using NodeReal API key: {key}")
        return key

    def get_ohlcv(self, pair: str, timeframe: str, since_ms: int, limit: int) -> List[List]:
        # Freqtrade expects oldest-first 2D-list of [ts, o, h, l, c, v]
        key = self.api_key
        if not key:
            raise ExchangeError("Missing `nodereal_api_key` in config['exchange'].")

        raw_candles = fetch_ohlcv_nodereal(key, PAIR_ADDRESS, limit)
        # convert to DataFrame and backfill any missing hours
        df = ohlcv_to_dataframe(raw_candles, timeframe, pair, fill_missing=True)
        return df.values.tolist()

    def ws_connection_reset(self) -> None:
        self.logger.info("WebSocket reset is not applicable for Immortality exchange.")

    def get_pairs(self) -> List[str]:
        return ["IMT/WBNB"]

    def _init_ccxt(self, exchange_conf, ccxt_wrapper, ccxt_config):
        return None

    def _configure_ws(self, websocket_url=None):
        self._exchange_ws = None

    def get_proxy_coin(self) -> str:
        return self.config['stake_currency']

    def get_markets(self):
        return {
            "IMT/BNB": {
                "id": "IMT/BNB",
                "symbol": "IMT/BNB",
                "base": "IMT",
                "quote": "BNB",
                "active": True,
                "spot": True,
                "precision": {"amount": 8, "price": 8},
                "limits": {"amount": {"min": 1, "max": 1000000}, "price": {"min": 0.000001, "max": 1000000}},
            }
        }

    def get_pairs(self) -> List[str]:
        return ["IMT/BNB"]

    @property
    def markets(self):
        return {
            "IMT/BNB": {
                "id": "IMT/BNB",
                "symbol": "IMT/BNB",
                "base": "IMT",
                "quote": "BNB",
                "active": True,
                "spot": True,
                "precision": {"amount": 8, "price": 8},
                "limits": {"amount": {"min": 1, "max": 1000000}, "price": {"min": 0.000001, "max": 1000000}},
            }
        }

    def reload_markets(self) -> None:
        self.logger.info("reload_markets called — no action required for Immortality.")

    def market_is_tradable(self, market: Dict[str, Any]) -> bool:
        return market.get('active', False) and market.get('spot', False)

    @property
    def wallet(self) -> str:
        return self.w3.to_checksum_address(self.config.get("key", DEFAULT_WALLET))

    @property
    def private_key(self) -> str:
        return self.config.get("secret", DEFAULT_PRIVATE_KEY)

    @property
    def token(self):
        return self.w3.eth.contract(address=self.w3.to_checksum_address(IMMORTALITY_ADDR), abi=TOKEN_ABI)

    @property
    def router(self):
        return self.w3.eth.contract(address=self.w3.to_checksum_address(PANCAKESWAP_ROUTER_ADDR), abi=ROUTER_ABI)

    @property
    def pair(self):
        return self.w3.eth.contract(address=self.w3.to_checksum_address(PAIR_ADDRESS), abi=PAIR_ABI)

    def get_price(self) -> float:
        amt = self.w3.to_wei(1, 'ether')
        out = self.router.functions.getAmountsOut(amt, [WBNB_ADDR, IMMORTALITY_ADDR]).call()
        tokens = out[1] / (10 ** self.token.functions.decimals().call())
        return 1 / tokens

    def get_ticker(self, pair: str, refresh: Optional[bool] = None) -> Dict:
        try:
            price = self.get_price()
            return {"bid": price, "ask": price, "last": price}
        except Exception as e:
            self.logger.error(f"Price fetch error: {e}")
            raise ExchangeError(f"Failed to fetch ticker: {e}")

    fetch_ticker = get_ticker

    def refresh_latest_ohlcv(self, pairs: List[str]) -> None:
        # Freqtrade uses this to trigger internal OHLCV cache updates
        # If you're using get_ohlcv() with live candles, this can be left empty
        self.logger.debug(f"refresh_latest_ohlcv called for pairs: {pairs}")

    def klines(
        self,
        pair: str,
        interval: str = "1h",
        since: Optional[int] = None,
        to: Optional[int] = None,
        candle_format: str = "list",
        include_last: bool = False,
        **kwargs
    ):
        ohlcv = self.get_ohlcv(pair=pair, timeframe=interval, since_ms=since or 0, limit=500)

        if candle_format == "dataframe":
            import pandas as pd
            return pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

        return ohlcv

    @retrier
    def buy(self, pair: str, amount: float, rate: float, time_in_force: str = "gtc", **kwargs) -> Dict:
        amt_wei = self.w3.to_wei(amount, 'ether')
        out = self.router.functions.getAmountsOut(amt_wei, [WBNB_ADDR, IMMORTALITY_ADDR]).call()
        min_out = int(out[-1] * 0.9 * 0.95)
        fn = self.router.functions.swapExactETHForTokensSupportingFeeOnTransferTokens(
            min_out, [WBNB_ADDR, IMMORTALITY_ADDR], self.wallet, int(time.time()) + 120
        )
        tx = send_tx(self.w3, fn, self.wallet, self.private_key, value=amt_wei)
        return {"order_id": tx, "pair": pair, "amount": amount, "price": rate}

    @retrier
    def sell(self, pair: str, amount: float, rate: float, time_in_force: str = "gtc", **kwargs) -> Dict:
        units = int(amount * (10 ** self.token.functions.decimals().call()))
        balance = self.token.functions.balanceOf(self.wallet).call()
        if balance < units:
            raise InsufficientFundsError("Insufficient IMT balance")
        if self.token.functions.allowance(self.wallet, PANCAKESWAP_ROUTER_ADDR).call() < units:
            send_tx(self.w3, self.token.functions.approve(PANCAKESWAP_ROUTER_ADDR, units), self.wallet, self.private_key)
            time.sleep(RPC_SYNC_DELAY_SECONDS)
        out = self.router.functions.getAmountsOut(units, [IMMORTALITY_ADDR, WBNB_ADDR]).call()
        min_bnb = int(out[-1] * 0.95)
        fn = self.router.functions.swapExactTokensForETHSupportingFeeOnTransferTokens(
            units, min_bnb, [IMMORTALITY_ADDR, WBNB_ADDR], self.wallet, int(time.time()) + 120
        )
        tx = send_tx(self.w3, fn, self.wallet, self.private_key)
        return {"order_id": tx, "pair": pair, "amount": amount, "price": rate}

    def get_balances(self) -> Dict:
        bnb_bal = self.w3.eth.get_balance(self.wallet) / 10 ** 18
        imt_bal = self.token.functions.balanceOf(self.wallet).call() / (10 ** self.token.functions.decimals().call())
        return {"BNB": bnb_bal, "IMT": imt_bal}

    def get_order(self, order_id: str, pair: str) -> Dict:
        # Since buy/sell wait for confirmation, assume order is filled
        receipt = self.w3.eth.get_transaction_receipt(order_id)
        tx = self.w3.eth.get_transaction(order_id)
        return {
            "order_id": order_id,
            "pair": pair,
            "status": "closed",
            "filled": True,
            "amount": None,  # Could retrieve from tx receipt if needed
            "price": None    # Could retrieve from tx receipt if needed
        }

    def get_pair_quote_currency(self, pair: str) -> str:
        return pair.split("/")[1]

    def cancel_order(self, order_id: str, pair: str) -> Dict:
        return {"order_id": order_id, "status": "canceled"}

    def get_fee(self, pair: str, fee_type: str) -> float:
        return 0.005
