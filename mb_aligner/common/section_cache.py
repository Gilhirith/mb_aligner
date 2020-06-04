# A dictionary based cache, similar to a map, that can work both for multiprocess and multithreaded applications
import multiprocessing as mp

class SectionCacheProcesses(object):

    def __init__(self):
        self._manager = mp.Manager()
        self._dict = self._manager.dict()


    def __setitem__(self, key, item):
        if isinstance(item, dict):
            self._dict[key] = self._manager.dict(item)
        elif isinstance(item, list):
            self._dict[key] = self._manager.list(item)
        else:
            self._dict[key] = item

    def __getitem__(self, key):
        return self._dict[key]

    def __repr__(self):
        return repr(self._dict)

    def __len__(self):
        return self._dict.len()

    def __delitem__(self, key):
        del self._dict[key]

    def clear(self):
        return self._dict.clear()

    def copy(self):
        return self._dict.copy()

    def has_key(self, k):
        return k in self._dict

    def update(self, *args, **kwargs):
        return self._dict.update(*args, **kwargs)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def pop(self, *args):
        return self._dict.pop(*args)

    def __cmp__(self, dict_):
        return self._cmp(self._dict, dict_)

    def __contains__(self, item):
        return item in self._dict

    def __iter__(self):
        return iter(self._dict)

    def __unicode__(self):
        return unicode(repr(self._dict))


