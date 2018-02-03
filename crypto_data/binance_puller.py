import time
import math
import os
from datetime import datetime
from binance.client import Client
from binance.enums import KLINE_INTERVAL_15MINUTE, KLINE_INTERVAL_5MINUTE, KLINE_INTERVAL_1HOUR, KLINE_INTERVAL_1DAY
from tqdm import tqdm
import pandas as pd

tqdm.monitor_interval = 0

INTERVALS = {KLINE_INTERVAL_15MINUTE: 'mid', KLINE_INTERVAL_5MINUTE: 'short', KLINE_INTERVAL_1HOUR: 'long'}
INTERVAL_TIME = {
    KLINE_INTERVAL_15MINUTE: 15 * 60,
    KLINE_INTERVAL_5MINUTE: 5 * 60,
    KLINE_INTERVAL_1HOUR: 4 * 60 * 60,
    KLINE_INTERVAL_1DAY: 24 * 60 * 60
}
CANDLE_COLUMNS = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asses_volume', 'n_trades',
                  'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']


def get_binance(): return Client(os.environ.get("BINANCE_API_KEY"), os.environ.get("BINANCE_SECRET_KEY"))


def pull_currency_candles(symbol: str, period: int, interval: str, name: str, path: str):
    client = get_binance()
    end_time = time.time() * 1000
    start_time = (time.time() - period) * 1000
    pbar = tqdm(total=math.ceil(period / (INTERVAL_TIME[interval] * 500)), desc=f'Pulling interval {interval} for {symbol}')
    candles = []
    while start_time < end_time:
        candles += client.get_klines(symbol=symbol, interval=interval, startTime=int(start_time), endTime=int(end_time))
        start_time = candles[-1][6]
        pbar.update(1)
    if len(candles) == 0: return

    path = f'{path}/{symbol}'
    os.makedirs(path, exist_ok=True)

    pretty_start = datetime.fromtimestamp((end_time / 1000 - period)).strftime('%Y-%m-%d')
    pretty_end = datetime.fromtimestamp(end_time / 1000).strftime('%Y-%m-%d')

    df = pd.DataFrame(candles, columns=CANDLE_COLUMNS)
    df.set_index('open_time', inplace=True)
    df.to_csv(f'{path}/{symbol}_{name}_{interval}_{pretty_start}-{pretty_end}.csv')
