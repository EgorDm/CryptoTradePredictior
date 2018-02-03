import time
import math
import numpy as np
from abc import ABC, abstractmethod
from keras import Model
from sklearn.metrics import confusion_matrix

from crypto_predictor import statistics


class BaseModel(ABC):
    __slots__ = 'model'

    def __init__(self, options: dict) -> None:
        super().__init__()
        self.model = self._initialize(**options)

    @abstractmethod
    def _initialize(self, **options) -> Model: pass

    def fit(self, X, Y, batch_size: int = 1024, epochs: int = 10, validation_split: float = 0.9):
        start = time.time()
        self.model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
        training_time = int(math.floor(time.time() - start))
        return training_time

    def test(self, X, Y) -> np.ndarray:
        Y_predict = self.model.predict(X)
        return statistics.confusion_matrix_stats(self._process_labels(Y), self._process_labels(Y_predict))

    @abstractmethod
    def _process_labels(self, labels) -> np.ndarray: pass
