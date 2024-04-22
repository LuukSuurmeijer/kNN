
from itertools import islice
from typing import Generator, Iterable, List
import math


def m_equal_batches(N: int, m: int) -> Generator[int, int, int]:
    """Divide iterable of size N into m equally sized batches. Yield start and end indices of each batch

    :param int N: Size of the iterable
    :param int m: Number of batches
    :yield: Start and end indices of each batch
    """
    split = math.ceil(N / m)
    for i in range(m):
        yield (i * split, min(split + (i * split), N))


def batches_of_m(N: int, m: int) -> Generator[int, int, int]:
    """Divide iterable of size N into batches of size m. Yield start and end indices of each batch.

    :param int N: Size of the iterable
    :param int m: Size of the batches
    :yield: Start and end indices of each batch
    """
    number_of_batches = math.ceil(N / m)
    for i in range(number_of_batches):
        yield (i * m, min(m + (i * m), N))


def batching_function(iterable: Iterable, n: int):
    """Batch data into tuples of length n. The last batch will be a remainder if not dividable. Yields directly from iterable."""
    if n < 1:
        raise ValueError("Batch size smaller than one specified")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def create_lookup_func(labels: List[str]):
    """Lookup function to go from index to label string"""

    def lookup_func_gen(i):
        """Return the ith label"""
        return labels[i]

    return lookup_func_gen
