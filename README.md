# Crypto Trade Predictior
Basically we are trying to predict if the price will rise or fall in pariod of time depending on past data and market indicators.

## Usage
### Pull data
Pull data to train on from exchanges
```
pull.py [-h] [--symbols SYMBOLS] [--intervals INTERVALS] [--name NAME] [--period PERIOD]

optional arguments:
  --symbols SYMBOLS     Symbols supported by data providers. Delimitered by
                        commas.
  --intervals INTERVALS
                        Intervals supported by data providers. Delimitered by
                        commas.
  --name NAME           Dataset name.
  --period PERIOD       Period which to scrap in days
```
#### Example
```
pull.py --symbols=BTCUSDT,ETHUSDT --intervals=15m,1h --name=example_dataset --period=365
```

### Train
Train on pulled data
```
train.py [-h] [--data DATA] [--model MODEL]

optional arguments:
  --data DATA    File or file wildcard for all data files
  --model MODEL  Name of model configuration yu would like to use
```
#### Example
```
train.py --data='data\\*\\*15m*.csv'
```
