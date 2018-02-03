import numpy as np
from keras import Sequential, Model
from keras.layers import LSTM, Dropout, Dense, Activation
from crypto_predictor.base_model import BaseModel


class POCModel(BaseModel):
    def _initialize(self, window_size: int, feature_count: int, optimizer: str, lossf: str, cell_size: int, outputs: int, activation: str, dropout: float = 0.2,
                    **options) -> Model:
        model = Sequential()

        model.add(LSTM(cell_size, return_sequences=True, input_shape=(window_size, feature_count)))
        model.add(Dropout(dropout))
        model.add(LSTM((cell_size * 2), return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(cell_size, return_sequences=False))

        model.add(Dense(units=outputs))
        model.add(Activation(activation))

        model.compile(loss=lossf, optimizer=optimizer)
        return model

    def _process_labels(self, labels) -> np.ndarray: return np.argmax(labels, axis=1)

