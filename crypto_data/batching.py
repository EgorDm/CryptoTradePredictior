import numpy as np
import pandas as pd
from crypto_data.normalization import normalize_windows


def create_batches(data_frame: pd.DataFrame, data_columns: list, label_columns: list, window_size: int, train_size: float = 0.9) -> tuple:
    data = np.array([data_frame[column].replace(to_replace=0, method='ffill').values for column in data_columns + label_columns]).transpose((1, 0))
    batches = np.array([data[i:i + (window_size + 1)] for i in range(len(data) - (window_size + 1))])

    # Save bases and normalize data
    bases = batches[:, 0, :]
    batches = normalize_windows(batches)

    # Splitting data
    split_index = int(round(train_size * batches.shape[0]))

    # Extract the inputs and labels
    X = batches[:, :-1, :]  # Everything but the last
    Y = batches[:, -1, len(data_columns):]  # Labels of the lest frame in window

    # Training data
    X_train, Y_train = X[:split_index], Y[:split_index]

    # Testing data
    X_test, Y_test = X[split_index:], Y[split_index:]

    return X_train, Y_train, X_test, Y_test, bases
