import pandas as pd
import tulipy as ti
import numpy as np


def load(path, indicators=None):
    ret = pd.read_csv(path)
    if indicators is not None: indicator_cols = [add_indicator(ret, indicator['name'], indicator['options']) for indicator in indicators]
    # Make sure thet we start data at non zeros which moving averages tend to produce
    ret = ret.loc[ret[(ret != 0).all(axis=1)].first_valid_index():]  # (ret != 0).all(axis=1)

    return ret


def translate_fields(inputs):
    if len(inputs) == 1: return ['close']
    return inputs


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

    return identifier
