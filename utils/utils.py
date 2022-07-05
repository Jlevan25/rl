import time
from typing import Union

import torch
import numpy as np
import math


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} in {end - start:0.8f} seconds")
        return result

    return wrapper


def append_dict2dict(input_dict, output_dict):
    for key in input_dict:
        if key in output_dict:
            output_dict[key].append(input_dict[key].detach().numpy())
        else:
            output_dict[key] = [input_dict[key].detach().numpy()]


def to_one_hot(x: Union[np.ndarray, torch.Tensor], num_columns: int = -1):
    if num_columns < 0:
        num_columns = x.max()

    columns = np.arange(num_columns) if isinstance(x, np.ndarray) else torch.arange(num_columns)

    return (columns == x[..., None]) * 1.


def actions_to_one_hot(indexes, width, is_numpy: bool = True):
    if hasattr(indexes, '__len__'):
        batch_size, actions, *_ = indexes.shape
        one_hot = np.zeros((batch_size, actions, width), dtype=np.int32)
        indexes = (np.repeat(np.arange(batch_size), actions),
                   np.tile(np.arange(actions), batch_size),
                   np.array(indexes).reshape(-1))
    else:
        one_hot = np.zeros(width, dtype=np.int32)

    one_hot[indexes] = 1
    return one_hot if is_numpy else torch.tensor(one_hot)


def indices_except(index, num_elems):
    return np.arange(len(num_elems)) != index


def get_rows_cols(num_elems):
    x = int(math.sqrt(num_elems))
    return x, int(math.ceil(num_elems / x))
