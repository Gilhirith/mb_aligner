# Receives an optimized mesh of a single section, and the section's original tilespec,
# and creates a new tilespec with the mesh transformation

from rh_logger.api import logger
import rh_logger
import logging
import numpy as np
from rh_renderer import models
import time
import tinyr

class MeshPointsModelExporter(object):
    def __init__(self):
        pass


    def update_section_points_model_transform(self, section, orig_pts, new_pts, mesh_spacing):
        """
        Update the given section's tiles' transformation to incorporate the alignment (post optimization) transformation.
        The orig_pts are points in the coordinate system of the stitched section, and the new_pts are the corresponding post
        optimization locations.
        Assumption: the given section is already stitched (i.e., each tile has a stitching transformation),
                    but not aligned.
        """

        assert(orig_pts.shape == new_pts.shape)

        # set the halo to twice the mesh_spacing
        halo = 2 * mesh_spacing
        logger.report_event("Points model halo: {}".format(halo), log_level=logging.DEBUG)

        # Create an r-tree for all the source points (so we can find the relevant control points for each tile)
        orig_pts_rtree = tinyr.RTree(interleaved=False, max_cap=5, min_cap=2)
        for p_idx, p in enumerate(orig_pts):
            # create a small rectangle for the point
            # (using the (x_min, x_max, y_min, y_max) notation)
            p_bbox = [p[0] - 0.5, p[0] + 0.5, p[1] - 0.5, p[1] + 0.5]
            orig_pts_rtree.insert(p_idx, p_bbox)

        tiles_to_remove = set()
        for tile_idx, tile in enumerate(section.tiles()):
            # Compute the tile's (post-stitching, pre-alignment) bbox with halo
            bbox_with_halo = list(tile.bbox)
            bbox_with_halo[0] -= halo
            bbox_with_halo[2] -= halo
            bbox_with_halo[1] += halo
            bbox_with_halo[3] += halo

            # find all orig_pts that in the bbox with halo
            filtered_pts_idxs = []
            rect_res = orig_pts_rtree.search(bbox_with_halo)
            for p_idx in rect_res:
                filtered_pts_idxs.append(p_idx)

            if len(filtered_pts_idxs) == 0:
                logger.report_event("Could not find any mesh points in bbox {}, skipping the tile {}".format(bbox_with_halo, tile.img_fname), log_level=logging.WARN)
                tiles_to_remove.append((tile_idx, tile))
                continue

            try:
                tile_model = models.PointsTransformModel((orig_pts[filtered_pts_idxs], new_pts[filtered_pts_idxs]))
                tile.add_transform(tile_model)
            except:
                logger.report_event("Found an error after applying the transformation on the boundaries of tile: {}, skipping the tile".format(tile.img_fname), log_level=logging.WARN)
                tiles_to_remove.add((tile.mfov_index, tile.tile_index))

        # remove tiles that no transformation was found for
        for mfov_tile_index in tiles_to_remove:
            logger.report_event("Removing tile {} from {}".format(mfov_tile_index, section.canonical_section_name_no_layer), log_level=logging.INFO)
            section.remove_tile(*mfov_tile_index)



# def compute_new_bounding_box(tile_ts):
#     """Computes a bounding box given the tile's transformations (if any),
#        and the new model to be applied last"""
#     # We must have a non-affine transformation, so compute the transformation of all the boundary pixels
#     # using a forward transformation from the boundaries of the source image to the destination
#     # Assumption: There won't be a pixel inside an image that goes out of the boundary
#     boundary1 = np.array([[float(p), 0.] for p in np.arange(tile_ts["width"])])
#     boundary2 = np.array([[float(p), float(tile_ts["height"] - 1)] for p in np.arange(tile_ts["width"])])
#     boundary3 = np.array([[0., float(p)] for p in np.arange(tile_ts["height"])])
#     boundary4 = np.array([[float(tile_ts["width"] - 1), float(p)] for p in np.arange(tile_ts["height"])])
#     boundaries = np.concatenate((boundary1, boundary2, boundary3, boundary4))
# 
#     for modelspec in tile_ts.get("transforms", []):
#         model = models.Transforms.from_tilespec(modelspec)
#         boundaries = model.apply(boundaries)
# 
#     # Find the bounding box of the boundaries
#     min_XY = np.min(boundaries, axis=0)
#     max_XY = np.max(boundaries, axis=0)
#     # If the boundig box is incorrect because the tile hasn't got matches in its scope, remove the tile
#     if np.any(np.isnan(min_XY)) or np.any(np.isnan(max_XY)):
#         return None
#     # Rounding to avoid float precision errors due to representation
#     new_bbox = [int(math.floor(round(min_XY[0], 5))), int(math.ceil(round(max_XY[0], 5))), int(math.floor(round(min_XY[1], 5))), int(math.ceil(round(max_XY[1], 5)))]
#     #new_bbox = [math.floor(round(min_XY[0], 5)), math.ceil(round(max_XY[0], 5)), math.floor(round(min_XY[1], 5)), math.ceil(round(max_XY[1], 5))]
#     return new_bbox


