import pandas as pd
import tulipy as ti
import numpy as np


def load(path, indicators=None):
    ret = pd.read_csv(path)
    for indicator in indicators: add_indicator(ret, indicator['name'], indicator['options'])
    return ret


def translate_fields(inputs):
    ret = ['close']
    if len(inputs) == 1: return ret
    # TODO: add more
    return ret


def add_indicator(frame, indicator, options):
    indicator = getattr(ti, indicator) if isinstance(indicator, str) else indicator
    fields = translate_fields(indicator.inputs)
    outputs = indicator.outputs
    inputs = frame.as_matrix(fields).T
    ret = indicator(*inputs, **options)

    for i in range(len(outputs)):
        identifier = f'{outputs[i]}_{"_".join([str(v) for v in options.values()])}'
        series = ret[i] if len(outputs) > 1 else ret
        frame[identifier] = np.append(np.zeros([inputs.shape[1] - len(series)]), series)
