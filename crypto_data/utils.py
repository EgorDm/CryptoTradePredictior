import numpy as np


def one_hot(label, vocab_size):
    ret = np.zeros(vocab_size)
    ret[label] = 1
    return ret
