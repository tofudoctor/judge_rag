#utils/batch.py
from itertools import islice

def batch_iter(it, size):
    it = iter(it)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch