import threading
from lru import LRU

THREAD_LOCAL = threading.local()

class ThreadLocalStorageLRU(object):
    LRU_SIZE = 10

    def __init__(self):
        if getattr(THREAD_LOCAL, 'lru', None) is None:
            THREAD_LOCAL.lru = LRU(ThreadLocalStorageLRU.LRU_SIZE)

    def __setitem__(self, key, item):
        THREAD_LOCAL.lru[key] = item

    def __getitem__(self, key):
        return THREAD_LOCAL.lru[key]

    def __repr__(self):
        return repr(THREAD_LOCAL.lru)

    def __len__(self):
        return THREAD_LOCAL.lru.len()

    def __delitem__(self, key):
        del THREAD_LOCAL.lru[key]

    def clear(self):
        return THREAD_LOCAL.lru.clear()

    def copy(self):
        return THREAD_LOCAL.lru.copy()

    def has_key(self, k):
        return k in THREAD_LOCAL.lru

    def update(self, *args, **kwargs):
        return THREAD_LOCAL.lru.update(*args, **kwargs)

    def keys(self):
        return THREAD_LOCAL.lru.keys()

    def values(self):
        return THREAD_LOCAL.lru.values()

    def items(self):
        return THREAD_LOCAL.lru.items()

    def pop(self, *args):
        return THREAD_LOCAL.lru.pop(*args)

    def __cmp__(self, lru_):
        return self._cmp(THREAD_LOCAL.lru, lru_)

    def __contains__(self, item):
        return item in THREAD_LOCAL.lru

    def __iter__(self):
        return iter(THREAD_LOCAL.lru)

    def __unicode__(self):
        return unicode(repr(THREAD_LOCAL.lru))


