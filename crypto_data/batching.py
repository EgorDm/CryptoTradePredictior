import numpy as np
import pandas as pd
import crypto_data.utils as utils
from crypto_data.normalization import normalize_windows

LABEL_NONE = 0
LABEL_SHORT = 1
LABEL_LONG = 2
LABEL_COUNT = 3


def generate_label(recent_label, future_label):  # TODO: use fees/treshhold for NONE
    if recent_label < future_label: return utils.one_hot(LABEL_LONG, LABEL_COUNT)
    return utils.one_hot(LABEL_SHORT, LABEL_COUNT)


def create_batches(data_frame: pd.DataFrame, ignore_columns: list, label_column: str, window_size: int, future_offset_factor: float = 0.1,
                   train_size: float = 0.9) -> tuple:
    data_columns = sorted(list(set(data_frame.columns) - set(ignore_columns + [label_column])))
    data = np.array([data_frame[column].replace(to_replace=0, method='ffill').values for column in data_columns + [label_column]]).transpose((1, 0))
    future_offset = int(round(window_size * future_offset_factor))

    # Make batch with body data and as last element future data for labelling
    batches = np.array([np.append(data[i:i + window_size], [data[i + window_size + future_offset]], axis=0)
                        for i in range(len(data) - (window_size + future_offset))])

    np.random.shuffle(batches)  # Shuffle data
    batches = normalize_windows(batches)  # Normalize data
    split_index = int(round(train_size * batches.shape[0]))  # Splitting data position

    # Extract the inputs and labels
    X = batches[:, :-1, :]  # Everything but the last
    Y = np.array([generate_label(batch[-2, -1], batch[-1, -1]) for batch in batches])

    # Training data
    X_train, Y_train = X[:split_index], Y[:split_index]

    # Testing data
    X_test, Y_test = X[split_index:], Y[split_index:]

    return X_train, Y_train, X_test, Y_test
