# An implementation of an image cache that stores up to a fixed number of images (tiles) in memory.
# Also allows to add a tag to each image, and evict from memory all images with the same tag.

import cv2
from lru import LRU
from collections import defaultdict

DEFAULT_MAX_TILES = 50


class Singleton(type):
    __instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls.__instances:
            cls.__instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls.__instances[cls]



class ImageCache(object, metaclass=Singleton):
    # The image cache is a singelton

    def __init__(self, max_images=DEFAULT_MAX_TILES):
        self._lru_dict = LRU(max_images)
        self._tags = defaultdict(set)

    def _load_file(self, img_fname):
        img = cv2.imread(fname, 0)
        self._lru_dict[img_fname] = img

    def get_file(self, img_fname, tag=None):
        # load the image if not present
        if img_fname not in self._lru_dict:
            self._load_file(img_fname)
        # set the image tag
        if tag is not None and tag not in self._tags:
            self._tags[tag].add(img_fname)

        return self._lru_dict[img_fname]

    def clear_entries_by_tag(self, tag):
        # Iterate through the items of the tag, and delete them from the cache (if they are still present)
        if tag in self._tags:
            for img_fname in tag:
                if img_fname in self._lru_dict:
                    del self._lru_dict[img_fname]
            # delete the tag from the tags list
            del self._tags[tag]


