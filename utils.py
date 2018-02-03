import json
import os
from constants import MODEL_PATH


def load_model_params(model_name):
    path = os.path.join(MODEL_PATH, f'{model_name}.json')
    return json.load(open(path))
