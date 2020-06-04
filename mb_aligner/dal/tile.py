import numpy as np
import os
import json
import cv2
import math
from rh_renderer import models
import rh_img_access_layer

class Tile(object):
    """
    Represents a single image tile in the system
    """

    def __init__(self, img_fname, **kwargs):
        self._img_fname = img_fname#.replace("file://", "")

        # Initialize default values
        self._img = None
        self._width = None
        self._height = None
        self._bbox = [] # [min_x, max_x, min_y, max_y]
        self._layer = None
        self._mfov_idx = None
        self._tile_idx = None
        self._transforms_modelspecs = []
        self._transforms = None
        self._non_affine_transform = False

        # initialize values using kwargs
        if len(kwargs) > 0:
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
        else:
            # load the image and populate some of the values
            img = self.image()
            self._width = img.shape[1]
            self._height = img.shape[0]
            self._bbox = [0, self._width, 0, self._height]
            

    @classmethod
    def create_from_tilespec(cls, tilespec):
        """
        Creates a tile from a given tilespec
        """
        return Tile(tilespec["mipmapLevels"]["0"]["imageUrl"], **tilespec)

    @classmethod
    def create_from_input(cls, img_fname, tile_size, tile_loc, layer, mfov_idx, tile_idx):
        """
        Creates a tile using the given parameters
        """
        if "://" in img_fname:
            img_url = img_fname
        else:
            img_url = "osfs://{}".format(img_fname)
        tilespec = {
            "mipmapLevels" : {
                "0" : {
                    "imageUrl" : img_url
                }
            },
            "minIntensity" : 0.0,
            "maxIntensity" : 255.0,
            "layer" : layer,
            "transforms" : [{
                "className" : "mpicbg.trakem2.transform.TranslationModel2D",
                "dataString" : "{0} {1}".format(tile_loc[0], tile_loc[1])
                }],
            "width" : tile_size[1],
            "height" : tile_size[0],
            "mfov" : mfov_idx,
            "tile_index" : tile_idx,
            # BoundingBox in the format "from_x to_x from_y to_y" (left right top bottom)
            "bbox" : [ tile_loc[0], tile_loc[0] + tile_size[1],
                tile_loc[1], tile_loc[1] + tile_size[0] ]
            }
        return Tile(tilespec["mipmapLevels"]["0"]["imageUrl"], **tilespec)
        

    @property
    def img_fname(self):
        """
        Returns the image file name
        """
        return self._img_fname

    @property
    def image(self):
        """
        Returns the actual image (a 2d array of rows, cols)
        """
        if self._img is None:
            self._img = self._load()
        return self._img

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
            self._transforms = [models.Transforms.from_tilespec(ts_transform) for ts_transform in self._transforms_modelspecs]
            self._non_affine_transform = not np.all([isinstance(t, models.AbstractAffineModel) for t in self._transforms])
        return self._transforms

    @property
    def transforms_non_affine(self):
        """
        Returns True iff at least one of the tile's transforms is non-affine
        """
        # compute the transforms if needed
        if self._transforms is None:
            self._transforms = [models.Transforms.from_tilespec(ts_transform) for ts_transform in self._transforms_modelspecs]
            self._non_affine_transform = not np.all([isinstance(t, models.AbstractAffineModel) for t in self._transforms])
        return self._non_affine_transform


    def _load(self):
        """
        Loads the image of the tile
        """
        return rh_img_access_layer.read_image_file(self._img_fname)

    def add_ts_transform(self, modelspec):
        """
        Adds a given tilespec transform to the list of transformations
        """
        self._transforms_modelspecs.append(modelspec)
        if self._transforms is not None:
            self._transforms.append(models.Transforms.from_tilespec(ts_transform))
            self._non_affine_transform = self._non_affine_transform | ~isinstance(self._transforms[-1], models.AbstractAffineModel)
        self._update_bbox()

    def add_transform(self, transform):
        """
        Adds a given transfromation (model) to the list of transformations
        """
        # add to the list of transformations (tilespec)
        modelspec = transform.to_modelspec()
        self._transforms_modelspecs.append(modelspec)
        if self._transforms is not None:
            self._transforms.append(transform)
            self._non_affine_transform = self._non_affine_transform | ~isinstance(transform, models.AbstractAffineModel)
        self._update_bbox()

    def set_transform(self, transform):
        """
        Adds a given transfromation (model) to the list of transformations
        """
        # add to the list of transformations (tilespec)
        modelspec = transform.to_modelspec()
        self._transforms_modelspecs = [modelspec]
        self._transforms = [transform]
        self._non_affine_transform = not isinstance(transform, models.AbstractAffineModel)
        self._update_bbox()
 
        

    @property
    def tilespec(self):
        """
        Returns a tilespec representation of the tile
        """
        if "://" in self._img_fname:
            img_url = self._img_fname
        else:
            img_url = "osfs://{}".format(self._img_fname)
        tilespec = {
            "mipmapLevels" : {
                "0" : {
                    "imageUrl" : img_url
                }
            },
            "minIntensity" : 0.0,
            "maxIntensity" : 255.0,
            "layer" : self._layer,
            "transforms" : self._transforms_modelspecs,
            "width" : self._width,
            "height" : self._height,
            "mfov" : self._mfov_idx,
            "tile_index" : self._tile_idx,
            # BoundingBox in the format "from_x to_x from_y to_y" (left right top bottom)
            "bbox" : self._bbox
            }
        return tilespec

    def save_as_json(self, out_fname):
        """
        Saves the tile as a tilespec (used for debugging).
        """
        with open(out_fname, 'w') as out_f:
            json.dump(self.tilespec, out_f, sort_keys=True, indent=4)

    def __repr__(self):
        return "Sec{}_Mfov{}_Tile{}".format(self.layer, self.mfov_index, self.tile_index)

    def _update_bbox(self):
        """
        Updates the tile's bounding box.
        If the tile's transforms are non-affine, update the bounding box by applying the transformations on the image corners, and taking the minimum and maximum x,y values.
        Otherwise, the boundary is transformed and the minimal and maximal X,Ys are computed.
        """
        if self.transforms_non_affine:
            # We  have a non-affine transformation, so compute the transformation of all the boundary pixels
            # using a forward transformation from the boundaries of the source image to the destination
            # Assumption: There won't be a pixel inside an image that goes out of the boundary
            boundaries = np.zeros((2 * self._width + 2 * self._height, 2), dtype=np.float)

            # Set boundary points with (X, 0)
            boundaries[:self._width, 0] = np.arange(self._width, dtype=float)
            # Set boundary points with (X, height-1)
            boundaries[self._width:2*self._width, 0] = np.arange(self._width, dtype=float)
            boundaries[self._width:2*self._width, 1] = float(self._height - 1)
            # Set boundary points with (0, Y)
            boundaries[2*self._width:2*self._width + self._height, 1] = np.arange(self._height, dtype=float)
            # Set boundary points with (width - 1, Y)
            boundaries[2*self._width + self._height:, 1] = np.arange(self._height, dtype=float)
            boundaries[2*self._width + self._height:, 0] = float(self._width - 1)

            for t in self._transforms:
                boundaries = t.apply(boundaries)

            # Find the bounding box of the boundaries
            xy_min = np.min(boundaries, axis=0)
            xy_max = np.max(boundaries, axis=0)
            # If the bounding box is incorrect because the tile hasn't got matches in its scope, remove the tile
            if np.any(np.isnan(xy_min)) or np.any(np.isnan(xy_max)):
                raise Exception("Tile has an invalid non-affine transform, and should be discarded")
            # Rounding to avoid float precision errors due to representation
            #new_bbox = [math.floor(round(min_XY[0], 5)), math.ceil(round(max_XY[0], 5)), math.floor(round(min_XY[1], 5)), math.ceil(round(max_XY[1], 5))]
            self._bbox = [int(math.floor(round(xy_min[0], 5))), int(math.ceil(round(xy_max[0], 5))), int(math.floor(round(xy_min[1], 5))), int(math.ceil(round(xy_max[1], 5)))]
        else:
            # affine only transform
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

        


