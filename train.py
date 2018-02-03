import argparse

import constants
import utils
import glob
import numpy as np
from crypto_data import create_batches, load
from crypto_predictor import POCModel


# noinspection PyShadowingNames
def main(model_params, files):
    if len(files) == 0: return print('No data files found!')

    batch_data = None
    for file in files:
        df = load(file, **model_params['loading'])
        file_batches = create_batches(df, **model_params['batching'])
        if batch_data is None: batch_data = file_batches
        else: batch_data = (np.append(batch_data[i], file_batches[i], axis=0) for i in range(len(batch_data)))
    X_train, Y_train, X_test, Y_test = batch_data
    print(X_train.shape)

    data_params = {'window_size': X_train.shape[1], 'feature_count': X_train.shape[2], 'outputs': 3}
    model = POCModel({**model_params['model'], **data_params})
    model.model.summary()
    model.fit(X_train, Y_train, **model_params['training'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data', type=str, default='data\\*\\*[15m]*.csv', help='File or file wildcard for all data files')
    parser.add_argument('--model', type=str, default=None, help='Name of model configuration yu would like to use')
    args = parser.parse_args()

    model_params = utils.load_model_params(constants.DEFAULT_MODEL)
    if args.model is not None: model_params = {**model_params, **utils.load_model_params(args.model)}
    main(model_params, glob.glob(args.data))
