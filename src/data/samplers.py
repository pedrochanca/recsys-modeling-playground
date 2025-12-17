from typing import Protocol, Dict, Set, Optional
import numpy as np


class NegativeSampler(Protocol):

    def sample(
        self,
        user_id: int,
        n_negatives: int,
        current_timestamp: Optional[int] = None,
    ) -> np.ndarray: ...


class GlobalUniformNegativeSampler:
    """
    Time-agnostic sampler:
    - Avoids all items the user has EVER interacted with.
    """

    def __init__(self, n_items: int, user_pos_items: Dict[int, Set[int]]):
        self.n_items = n_items
        self.user_pos_items = user_pos_items

    def sample(
        self,
        user_id: int,
        n_negatives: int,
        current_timestamp: Optional[int] = None,
    ) -> np.ndarray:
        negatives = []
        pos_items = self.user_pos_items.get(user_id, set())

        while len(negatives) < n_negatives:
            j = np.random.randint(self.n_items)
            if j not in pos_items:
                negatives.append(j)

        return np.array(negatives, dtype=np.int64)


class TimeAwareNegativeSampler:
    """
    Skeleton for future time-aware negatives.
    Currently behaves like global sampler but keeps the interface ready.
    """

    def __init__(self, n_items: int, user_history, item_history):
        self.n_items = n_items
        self.user_history = user_history
        self.item_history = item_history

    def sample(
        self,
        user_id: int,
        n_negatives: int,
        current_timestamp: Optional[int] = None,
    ) -> np.ndarray:
        # TODO: later implement "up to current_timestamp" behavior
        items = self.item_history.get(user_id, [])
        pos_items = set(items)

        negatives = []
        while len(negatives) < n_negatives:
            j = np.random.randint(self.n_items)
            if j not in pos_items:
                negatives.append(j)

        return np.array(negatives, dtype=np.int64)
