import random
from typing import Set
from typing import List
from typing import Tuple


def split_train_test(items: Set[str], ratio: float = 0.8,ratio_test:float = -1) -> Tuple[Set[str], Set[str]]:
    _items = list(items)
    length = len(_items)
    items_to_select = int(length * ratio)
    train_samples = set(random.sample(_items, items_to_select))
    test_samples = set([item for item in items if item not in train_samples])
    if ratio_test != -1:
        _test_items = list(test_samples)
        len_test = len(_test_items)
        _ratio_test = ratio_test/(1-ratio)
        test_select = int(len_test*_ratio_test)
        test_samples = set(random.sample(_test_items, test_select))
    return train_samples, test_samples
