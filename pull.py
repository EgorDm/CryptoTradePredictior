import argparse

from constants import DATA_PATH
from crypto_data.binance_puller import pull_currency_candles


def main(symbols, intervals, period, name):
    for symbol in symbols:
        for interval in intervals:
            pull_currency_candles(symbol, period, interval, name, DATA_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get training data.')
    parser.add_argument('--symbols', type=str, help='Symbols supported by data providers. Delimitered by commas.')
    parser.add_argument('--intervals', type=str, help='Intervals supported by data providers. Delimitered by commas.')
    parser.add_argument('--name', type=str, default='default', help='Dataset name.')
    parser.add_argument('--period', type=int, default=365 * 3, help='Period which to scrap in days')
    args = parser.parse_args()
    main(args.symbols.split(','), args.intervals.split(','), args.period * 24 * 60 * 60, args.name)
