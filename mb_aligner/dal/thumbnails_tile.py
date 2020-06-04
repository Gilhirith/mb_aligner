import numpy as np
import os
import json
import cv2

class ThumbnailsTile(object):
    """
    Represents a single image tile (consists of all thumbnailed tiles in an mfov) in the system
    """

    def __init__(self, mfov_img, mfov_mask, **kwargs):
        self._mfov_img = mfov_img
        self._mfov_mask = mfov_mask

        # Initialize default values
        self._width = None
        self._height = None
        self._bbox = [] # [min_x, max_x, min_y, max_y]
        self._layer = None
        self._mfov_idx = None
        self._tile_idx = None
        self._transforms_modelspecs = []
        self._transforms = None

        # initialize values using kwargs
        if "width" in kwargs and "height" in kwargs:
            self._width = kwargs["width"]
            self._height = kwargs["height"]
            if "bbox" not in kwargs:
                self._bbox = [0, self._width, 0, self._height]
        if "bbox" in kwargs:
            self._bbox = kwargs["bbox"] # [start_x, end_x, start_y, end_y] (after applying the transformations)
            if "width" not in kwargs or "height" not in kwargs:
                self._width = self._bbox[1] - self._bbox[0]
                self._height = self._bbox[3] - self._bbox[2]
        if "layer" in kwargs:
            self._layer = kwargs["layer"]
        if "mfov" in kwargs:
            self._mfov_idx = kwargs["mfov"]
        if "tile_index" in kwargs:
            self._tile_idx = kwargs["tile_index"]
        if "transforms" in kwargs:
            self._transforms_modelspecs = kwargs["transforms"]
           

    @classmethod
    def create_from_input(cls, img_fnames, tiles_size, tiles_locs, layer, mfov_idx):
        """
        Creates a thumbnails tile using the given parameters
        """
        min_xy_f = np.min(tiles_locs, axis=0)
        min_xy = np.floor(min_xy_f).astype(np.int)
        max_xy = np.ceil(np.max(tiles_locs, axis=0)).astype(np.int)

        # create the stitched mfov image
        mfov_shape = tuple((max_xy - min_xy)[::-1] + np.array(tiles_size))
        #print("mfov_shape: {}".format(mfov_shape))
        mfov_image = np.zeros(mfov_shape, dtype=np.uint8)
        mfov_mask = np.zeros_like(mfov_image)
        for img_fname, img_loc in zip(img_fnames, tiles_locs):
            img = cv2.imread(img_fname, 0)
            assert(img.shape == tiles_size)

#             # normalize and clamp values
#             img = ((img - np.mean(img))/np.std(img)*40 + 127).astype(np.int)
#             img[img < 0] = 0
#             img[img > 255] = 255
#             img = img.astype(np.uint8)

            img_loc = np.floor(img_loc - min_xy_f).astype(np.int)
            mfov_image[img_loc[1]:img_loc[1] + tiles_size[0], img_loc[0]:img_loc[0] + tiles_size[1]] = img
            mfov_mask[img_loc[1]:img_loc[1] + tiles_size[0], img_loc[0]:img_loc[0] + tiles_size[1]] = 255


        tilespec = {
            "layer" : layer,
            "transforms" : [{
                "className" : "mpicbg.trakem2.transform.TranslationModel2D",
                "dataString" : "{} {}".format(*min_xy)
                }],
            "width" : mfov_shape[1],
            "height" : mfov_shape[0],
            "mfov" : mfov_idx,
            "tile_index" : 1,
            # BoundingBox in the format "from_x to_x from_y to_y" (left right top bottom)
            "bbox" : [ min_xy[0], max_xy[0] + tiles_size[1],
                min_xy[1], max_xy[1] + tiles_size[0] ]
            }
        return ThumbnailsTile(mfov_image, mfov_mask, **tilespec)
        

    @property
    def img_fname(self):
        """
        Returns the image file name.
        As there isn't any name, just return the mfov number
        """
        return "No actual file, (layer, mfov_idx) ({}, {})".format(self._layer, self._mfov_idx)

    @property
    def image(self):
        """
        Returns the actual image (a 2d array of rows, cols)
        """
        return self._mfov_img

    @property
    def mask(self):
        """
        Returns the mfov image mask (a 2d array of rows, cols)
        """
        return self._mfov_mask

    @property
    def width(self):
        """
        Returns the width of the image
        """
        return self._width

    @property
    def height(self):
        """
        Returns the height of the image
        """
        return self._height

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
    def tile_index(self):
        """
        Returns the tile index in the mfov (1-61)
        """
        return self._tile_idx

    @property
    def transforms(self):
        """
        Returns the transformations that need to be applied to the tile
        """
        if self._transforms is None:
            self._transforms = [models.from_tilespec(ts_transform) for ts_transform in self._transforms_modelspecs]
        return self._transforms

    def add_ts_transform(self, modelspec):
        """
        Adds a given tilespec transform to the list of transformations
        """
        self._transforms_modelspecs.append(modelspec)
        if self._transforms is not None:
            self._transforms.append(models.from_tilespec(ts_transform))

    def add_transform(self, transform):
        """
        Adds a given transfromation (model) to the list of transformations
        """
        # add to the list of transformations (tilespec)
        modelspec = transform.to_modelspec()
        self._transforms_modelspecs.append(modelspec)
        if self_transforms is not None:
            self._transforms.append(transform)

    def set_transform(self, transform):
        """
        Sets the given transfromation (model) as the only transform for the tile
        """
        # add to the list of transformations (tilespec)
        modelspec = transform.to_modelspec()
        self._transforms_modelspecs = [modelspec]
        self._transforms = [transform]
        self._update_bbox()
 
        

#     @property
#     def tilespec(self):
#         """
#         Returns a tilespec representation of the tile
#         """
#         tilespec = {
#             "layer" : self._layer,
#             "transforms" : self._transforms_modelspecs,
#             "width" : self._width,
#             "height" : self._height,
#             "mfov" : self._mfov_idx,
#             "tile_index" : self._tile_idx,
#             # BoundingBox in the format "from_x to_x from_y to_y" (left right top bottom)
#             "bbox" : self._bbox
#             }
#         return tilespec
# 
#     def save_as_json(self, out_fname):
#         """
#         Saves the tile as a tilespec (used for debugging).
#         """
#         with open(out_fname, 'w') as out_f:
#             json.dump(self.tilespec, out_f, sort_keys=True, indent=4)
# 
    def __repr__(self):
        return "ThumbnailsSec{}_Mfov{}_Tile{}".format(self.layer, self.mfov_index, self.tile_index)

    def _update_bbox(self):
        """
        Updates the tile's bounding box by applying the transformations on the image corners, and taking the minimum and maximum x,y values.
        """
        corners = np.array([
            [0., 0.],
            [self._width, 0.],
            [self._width, self._height],
            [0., self._height]
        ])

        for t in self._transforms:
            corners = t.apply(corners)


        xy_min = np.min(corners, axis=0)
        xy_max = np.max(corners, axis=0)

        self._bbox = [xy_min[0], xy_max[0], xy_min[1], xy_max[1]]

        


