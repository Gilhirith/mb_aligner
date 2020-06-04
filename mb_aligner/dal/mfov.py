import numpy as np
import os
import json
from .tile import Tile

class Mfov(object):
    """
    Represents a single Multibeam-Field-of-View (61 tiles) in the system
    """

    def __init__(self, tiles=[], **kwargs):
        self._tiles = tiles

        # Initialize default values
        self._mfov_idx = None
        self._layer = None
        self._bbox = None
        #print('Creating mfov')

        # initialize values using kwargs
        if len(kwargs) > 0:
            if "bbox" in kwargs:
                self._bbox = kwargs["bbox"] # [start_x, end_x, start_y, end_y] (after applying the transformations)
            if "layer" in kwargs:
                self._layer = kwargs["layer"]
            if "mfov_idx" in kwargs:
                self._mfov_idx = kwargs["mfov"]
        elif len(self._tiles) > 0:
            self._mfov_idx = self._tiles[0].mfov_index
            #print('Here', self._mfov_idx)
            self._layer = self._tiles[0].layer
            self._compute_bbox_from_tiles()
        #print('There', self._mfov_idx)


    @classmethod
    def create_from_tilespec(cls, tilespec):
        """
        Creates an mfov from a given tilespec
        """
        tiles = [Tile.create_from_tilespec(tile_ts) for tile_ts in tilespec]
        return Mfov(tiles)


    def _compute_bbox_from_tiles(self):
        """
        Computes an mfov bounding box of all the tiles in the mfov
        """
        if len(self._tiles) > 0:
            bboxes = [tile.bbox for tile in self._tiles]
            bboxes = [bbox for bbox in bboxes if bbox is not None] # Filter the tiles that don't have a bounding box
            if len(bboxes) > 0:
                bboxes = np.array(bboxes)
                self._bbox = [min(bboxes[:, 0]), max(bboxes[:, 1]), min(bboxes[:, 2]), max(bboxes[:, 3])]
        else:
            self._bbox = None


    @property
    def bbox(self):
        """
        Returns the bounding box of the tile in the following format [from_x, to_x, from_y, to_y]
        """
        return self._bbox

    @property
    def layer(self):
        """
        Returns the section layer number
        """
        return self._layer

    @property
    def mfov_index(self):
        """
        Returns the mfov index the tile is from in the section
        """
        return self._mfov_idx

    @property
    def tilespec(self):
        """
        Returns a tilespec representation of the mfov
        """
        return [tile.tilespec for tile in self._tiles]

    def save_as_json(self, out_fname):
        """
        Saves the mfov as a tilespec (used for debugging).
        """
        with open(out_fname, 'w') as out_f:
            json.dump(self.tilespec, out_f, sort_keys=True, indent=4)

    def tiles(self):
        '''
        A generator that iterates over all the tiles in the mfov
        '''
        for tile in self._tiles:
            yield tile

    def get_tile(self, tile_idx):
        '''
        Returns the tile with the given tile_idx
        '''
        for t in self.tiles():
            if t.tile_index == tile_idx:
                return t
        return None

    def remove_tile(self, tile_index):
        '''
        Removes a single tile from the mfov.
        '''
        to_remove_idx = None
        for t_idx, t in enumerate(self.tiles()):
            if t.tile_index == tile_index:
                to_remove_idx = t_idx
                break

        self._tiles.pop(to_remove_idx)

