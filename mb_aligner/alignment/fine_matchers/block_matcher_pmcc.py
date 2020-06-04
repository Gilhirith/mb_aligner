# Setup
from __future__ import print_function
from rh_logger.api import logger
import rh_logger
import logging
import os
import numpy as np
import time
import sys
from scipy.spatial import distance
from scipy import spatial
import cv2
import argparse
from mb_aligner.common import utils
from rh_renderer import models
from mb_aligner.alignment.fine_matchers import PMCC_filter
import multiprocessing as mp
from rh_renderer.tilespec_affine_renderer import TilespecAffineRenderer
import threading
from scipy.spatial import cKDTree as KDTree
from collections import defaultdict



# import pyximport
# pyximport.install()
# from ..common import cv_wrap_module


threadLocal = threading.local()

class BlockMatcherPMCCDispatcher(object):


    class BlockMatcherPMCC(object):
        def __init__(self, sec1, sec2, sec1_to_sec2_transform, **kwargs):
            self._scaling = kwargs.get("scaling", 0.2)
            self._template_size = kwargs.get("template_size", 200)
            self._search_window_size = kwargs.get("search_window_size", 8 * self._template_size)
            logger.report_event("Actual template size: {} and window search size: {} (after scaling)".format(self._template_size * self._scaling, self._search_window_size * self._scaling), log_level=logging.INFO)

            # Parameters for PMCC filtering
            self._min_corr = kwargs.get("min_correlation", 0.2)
            self._max_curvature = kwargs.get("maximal_curvature_ratio", 10)
            self._max_rod = kwargs.get("maximal_ROD", 0.9)
            self._use_clahe = kwargs.get("use_clahe", False)
            if self._use_clahe:
                self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

            #self._debug_dir = kwargs.get("debug_dir", None)
            self._debug_save_matches = None


            self._template_scaled_side = self._template_size * self._scaling / 2
            self._search_window_scaled_side = self._search_window_size * self._scaling / 2

            self._sec1 = sec1
            self._sec2 = sec2
            self._sec1_to_sec2_transform = sec1_to_sec2_transform
            
            self._scale_transformation = np.array([
                                        [ self._scaling, 0., 0. ],
                                        [ 0., self._scaling, 0. ]
                                    ])
            # For section1 there will be a single renderer with transformation and scaling
            self._sec1_scaled_renderer = TilespecAffineRenderer(self._sec1.tilespec)
            self._sec1_scaled_renderer.add_transformation(self._sec1_to_sec2_transform.get_matrix())
            self._sec1_scaled_renderer.add_transformation(self._scale_transformation)

            # for section2 there will only be a single renderer (no need to transform back to sec1)
            self._sec2_scaled_renderer = TilespecAffineRenderer(self._sec2.tilespec)
            self._sec2_scaled_renderer.add_transformation(self._scale_transformation)


        def set_debug_dir(self, debug_dir):
            self._debug_save_matches = True
            self._debug_dir = debug_dir

        
        def match_sec1_to_sec2_mfov(self, sec1_pts):
            # Apply the mfov transformation to compute estimated location on sec2
            sec1_mfov_pts_on_sec2 = self._sec1_to_sec2_transform.apply(np.atleast_2d(sec1_pts)) * self._scaling

            valid_matches = [[], [], []]
            invalid_matches = [[], []]
            for sec1_pt, sec2_pt_estimated in zip(sec1_pts, sec1_mfov_pts_on_sec2):

                # Fetch the template around img1_point (after transformation)
                from_x1, from_y1 = sec2_pt_estimated - self._template_scaled_side
                to_x1, to_y1 = sec2_pt_estimated + self._template_scaled_side
                sec1_template, sec1_template_start_point = self._sec1_scaled_renderer.crop(from_x1, from_y1, to_x1, to_y1)
            
                # Fetch a large sub-image around img2_point (using search_window_scaled_size)
                from_x2, from_y2 = sec2_pt_estimated - self._search_window_scaled_side
                to_x2, to_y2 = sec2_pt_estimated + self._search_window_scaled_side
                sec2_search_window, sec2_search_window_start_point = self._sec2_scaled_renderer.crop(from_x2, from_y2, to_x2, to_y2)
        
                # execute the PMCC match
                # Do template matching
                if np.any(np.array(sec2_search_window.shape) == 0) or np.any(np.array(sec1_template.shape) == 0):
                    continue
                if sec1_template.shape[0] >= sec2_search_window.shape[0] or sec1_template.shape[1] >= sec2_search_window.shape[1]:
                    continue
                if self._use_clahe:
                    sec2_search_window_clahe = self._clahe.apply(sec2_search_window)
                    sec1_template_clahe = self._clahe.apply(sec1_template)
                    pmcc_result, reason, match_val = PMCC_filter.PMCC_match(sec2_search_window_clahe, sec1_template_clahe, min_correlation=self._min_corr, maximal_curvature_ratio=self._max_curvature, maximal_ROD=self._max_rod)
                else:
                    pmcc_result, reason, match_val = PMCC_filter.PMCC_match(sec2_search_window, sec1_template, min_correlation=self._min_corr, maximal_curvature_ratio=self._max_curvature, maximal_ROD=self._max_rod)

                if pmcc_result is None:
                    invalid_matches[0].append(sec1_pt)
                    invalid_matches[1].append(reason)
#                     debug_out_fname1 = "temp_debug/debug_match_sec1{}-{}_template.png".format(int(sec1_pt[0]), int(sec1_pt[1]), int(sec2_pt_estimated[0]), int(sec2_pt_estimated[1]))
#                     debug_out_fname2 = "temp_debug/debug_match_sec1{}-{}_search_window.png".format(int(sec1_pt[0]), int(sec1_pt[1]), int(sec2_pt_estimated[0]), int(sec2_pt_estimated[1]))
#                     cv2.imwrite(debug_out_fname1, sec1_template)
#                     cv2.imwrite(debug_out_fname2, sec2_search_window)
                else:
                    # Compute the location of the matched point on img2 in non-scaled coordinates
                    matched_location_scaled = np.array([reason[1], reason[0]]) + np.array([from_x2, from_y2]) + self._template_scaled_side
                    sec2_pt = matched_location_scaled / self._scaling 
                    logger.report_event("{}: match found: {} and {} (orig assumption: {})".format(os.getpid(), sec1_pt, sec2_pt, sec2_pt_estimated / self._scaling), log_level=logging.DEBUG)
                    if self._debug_save_matches:
                        debug_out_fname1 = os.path.join(self._debug_dir, "debug_match_sec1_{}-{}_sec2_{}-{}_image1.png".format(int(sec1_pt[0]), int(sec1_pt[1]), int(sec2_pt[0]), int(sec2_pt[1])))
                        debug_out_fname2 = os.path.join(self._debug_dir, "debug_match_sec1_{}-{}_sec2_{}-{}_image2.png".format(int(sec1_pt[0]), int(sec1_pt[1]), int(sec2_pt[0]), int(sec2_pt[1])))
                        cv2.imwrite(debug_out_fname1, sec1_template)
                        sec2_cut_out = sec2_search_window[int(reason[0]):int(reason[0] + 2 * self._template_scaled_side), int(reason[1]):int(reason[1] + 2 * self._template_scaled_side)]
                        cv2.imwrite(debug_out_fname2, sec2_cut_out)
                    valid_matches[0].append(np.array(sec1_pt))
                    valid_matches[1].append(sec2_pt)
                    valid_matches[2].append(match_val)
            return valid_matches, invalid_matches
        

        def match_sec2_to_sec1_mfov(self, sec2_pts):
            # Assume that only sec1 renderer was transformed and not sec2 (and both scaled)
            sec2_pts = np.asarray(sec2_pts)
            sec2_pts_scaled = sec2_pts * self._scaling

            mat = self._sec1_to_sec2_transform.get_matrix()
            inverse_mat = np.linalg.inv(mat)
 
            #inverse_model = BlockMatcherPMCC.inverse_transform(self._sec1_to_sec2_transform)
            #sec2_pts_on_sec1 = inverse_model.apply(sec2_pts)

            valid_matches = [[], [], []]
            invalid_matches = [[], []]
            for sec2_pt, sec2_pt_scaled in zip(sec2_pts, sec2_pts_scaled):
                # sec1_pt_estimated is after the sec1_to_sec2 transform
                sec1_pt_estimated = sec2_pt_scaled

                # Fetch the template around sec2_pt_scaled (no transformation, just scaling)
                from_x2, from_y2 = sec2_pt_scaled - self._template_scaled_side
                to_x2, to_y2 = sec2_pt_scaled + self._template_scaled_side
                sec2_template, sec2_template_start_point = self._sec2_scaled_renderer.crop(from_x2, from_y2, to_x2, to_y2)
            
                # Fetch a large sub-image around sec1_pt_estimated (after transformation, using search_window_scaled_size)
                from_x1, from_y1 = sec1_pt_estimated - self._search_window_scaled_side
                to_x1, to_y1 = sec1_pt_estimated + self._search_window_scaled_side
                sec1_search_window, sec1_search_window_start_point = self._sec1_scaled_renderer.crop(from_x1, from_y1, to_x1, to_y1)
        
                # execute the PMCC match
                # Do template matching
                if np.any(np.array(sec1_search_window.shape) == 0) or np.any(np.array(sec2_template.shape) == 0):
                    continue
                if sec2_template.shape[0] >= sec1_search_window.shape[0] or sec2_template.shape[1] >= sec1_search_window.shape[1]:
                    continue
                if self._use_clahe:
                    sec1_search_window_clahe = self._clahe.apply(sec1_search_window)
                    sec2_template_clahe = self._clahe.apply(sec2_template)
                    pmcc_result, reason, match_val = PMCC_filter.PMCC_match(sec1_search_window_clahe, sec2_template_clahe, min_correlation=self._min_corr, maximal_curvature_ratio=self._max_curvature, maximal_ROD=self._max_rod)
                else:
                    pmcc_result, reason, match_val = PMCC_filter.PMCC_match(sec1_search_window, sec2_template, min_correlation=self._min_corr, maximal_curvature_ratio=self._max_curvature, maximal_ROD=self._max_rod)

                if pmcc_result is None:
                    invalid_matches[0].append(sec2_pt)
                    invalid_matches[1].append(reason)
#                     debug_out_fname1 = "temp_debug/debug_match_sec2{}-{}_template.png".format(int(sec2_pt[0]), int(sec2_pt[1]), int(sec1_pt_estimated[0]), int(sec1_pt_estimated[1]))
#                     debug_out_fname2 = "temp_debug/debug_match_sec2{}-{}_search_window.png".format(int(sec2_pt[0]), int(sec2_pt[1]), int(sec1_pt_estimated[0]), int(sec1_pt_estimated[1]))
#                     cv2.imwrite(debug_out_fname1, sec2_template)
#                     cv2.imwrite(debug_out_fname2, sec1_search_window)
                else:
                    # Compute the location of the matched point on img2 in non-scaled coordinates
                    matched_location_scaled = np.array([reason[1], reason[0]]) + np.array([from_x1, from_y1]) + self._template_scaled_side
                    sec1_pt = matched_location_scaled / self._scaling 
                    sec1_pt = np.dot(inverse_mat[:2,:2], sec1_pt) + inverse_mat[:2,2]
                    logger.report_event("{}: match found: {} and {} (orig assumption: {})".format(os.getpid(), sec2_pt, sec1_pt, np.dot(inverse_mat[:2,:2], sec1_pt_estimated / self._scaling) + inverse_mat[:2,2]), log_level=logging.DEBUG)
                    if self._debug_save_matches:
                        debug_out_fname1 = os.path.join(self._debug_dir, "debug_match_sec2_{}-{}_sec1_{}-{}_image1.png".format(int(sec2_pt[0]), int(sec2_pt[1]), int(sec1_pt[0]), int(sec1_pt[1])))
                        debug_out_fname2 = os.path.join(self._debug_dir, "debug_match_sec2_{}-{}_sec1_{}-{}_image2.png".format(int(sec2_pt[0]), int(sec2_pt[1]), int(sec1_pt[0]), int(sec1_pt[1])))
                        cv2.imwrite(debug_out_fname1, sec2_template)
                        sec1_cut_out = sec1_search_window[int(reason[0]):int(reason[0] + 2 * self._template_scaled_side), int(reason[1]):int(reason[1] + 2 * self._template_scaled_side)]
                        cv2.imwrite(debug_out_fname2, sec1_cut_out)
                    valid_matches[0].append(sec2_pt)
                    valid_matches[1].append(sec1_pt)
                    valid_matches[2].append(match_val)
            return valid_matches, invalid_matches
        

    def __init__(self, **kwargs):

        self._matcher_kwargs = kwargs
        self._mesh_spacing = kwargs.get("mesh_spacing", 1500)

#         self._scaling = kwargs.get("scaling", 0.2)
#         self._template_size = kwargs.get("template_size", 200)
#         self._search_window_size = kwargs.get("search_window_size", 8 * template_size)
#         logger.report_event("Actual template size: {} and window search size: {} (after scaling)".format(template_size * scaling, search_window_size * scaling), log_level=logging.INFO)
# 
#         # Parameters for PMCC filtering
#         self._min_corr = kwargs.get("min_correlation", 0.2)
#         self._max_curvature = kwargs.get("maximal_curvature_ratio", 10)
#         self._max_rod = kwargs.get("maximal_ROD", 0.9)
#         self._use_clahe = kwargs.get("use_clahe", False)

        self._debug_dir = kwargs.get("debug_dir", None)
        if self._debug_dir is not None:
            logger.report_event("Debug mode - on", log_level=logging.INFO)
            # Create a debug directory
            import datetime
            self._debug_dir = os.path.join(self._debug_dir, 'debug_matches_{}'.format(datetime.datetime.now().isoformat()))
            os.mkdirs(self._debug_dir)

               
    @staticmethod
    def _is_point_in_img(img_bbox, point):
        """Returns True if the given point lies inside the image as denoted by the given tile_tilespec"""
        # TODO - instead of checking inside the bbox, need to check inside the polygon after transformation
        if point[0] > img_bbox[0] and point[1] > img_bbox[2] and \
           point[0] < img_bbox[1] and point[1] < img_bbox[3]:
            return True
        return False

    @staticmethod
    def sum_invalid_matches(invalid_matches):
        if len(invalid_matches[1]) == 0:
            return [0] * 5
        hist, _ = np.histogram(invalid_matches[1], bins=5)
        return hist


    @staticmethod
    def _perform_matching(sec1_mfov_tile_idx, sec1, sec2, sec1_to_sec2_mfov_transform, sec1_mfov_mesh_pts, sec2_mfov_mesh_pts, debug_dir, matcher_args):
#         fine_matcher_key = "block_matcher_{},{},{}".format(sec1.canonical_section_name, sec2.canonical_section_name, sec1_mfov_tile_idx[0])
#         fine_matcher = getattr(threadLocal, fine_matcher_key, None)
#         if fine_matcher is None:
#             fine_matcher = BlockMatcherPMCCDispatcher.BlockMatcherPMCC(sec1, sec2, sec1_to_sec2_mfov_transform, **matcher_args)
#             if debug_dir is not None:
#                 fine_matcher.set_debug_dir(debug_dir)
# 
#             setattr(threadLocal, fine_matcher_key, fine_matcher)

        fine_matcher = BlockMatcherPMCCDispatcher.BlockMatcherPMCC(sec1, sec2, sec1_to_sec2_mfov_transform, **matcher_args)
        if debug_dir is not None:
            fine_matcher.set_debug_dir(debug_dir)

        logger.report_event("Block-Matching+PMCC layers: {} with {} (mfov1 {}) {} mesh points1, {} mesh points2".format(sec1.canonical_section_name, sec2.canonical_section_name, sec1_mfov_tile_idx, len(sec1_mfov_mesh_pts), len(sec2_mfov_mesh_pts)), log_level=logging.INFO)
        logger.report_event("Block-Matching+PMCC layers: {} -> {}".format(sec1.canonical_section_name, sec2.canonical_section_name), log_level=logging.INFO)
        valid_matches1, invalid_matches1 = fine_matcher.match_sec1_to_sec2_mfov(sec1_mfov_mesh_pts)
        logger.report_event("Block-Matching+PMCC layers: {} -> {} valid matches: {}, invalid_matches: {} {}".format(sec1.canonical_section_name, sec2.canonical_section_name, len(valid_matches1[0]), len(invalid_matches1[0]), BlockMatcherPMCCDispatcher.sum_invalid_matches(invalid_matches1)), log_level=logging.INFO)

        logger.report_event("Block-Matching+PMCC layers: {} <- {}".format(sec1.canonical_section_name, sec2.canonical_section_name), log_level=logging.INFO)
        valid_matches2, invalid_matches2 = fine_matcher.match_sec2_to_sec1_mfov(sec2_mfov_mesh_pts)
        logger.report_event("Block-Matching+PMCC layers: {} <- {} valid matches: {}, invalid_matches: {} {}".format(sec1.canonical_section_name, sec2.canonical_section_name, len(valid_matches2[0]), len(invalid_matches2[0]), BlockMatcherPMCCDispatcher.sum_invalid_matches(invalid_matches2)), log_level=logging.INFO)

        return sec1_mfov_tile_idx, valid_matches1, valid_matches2

#     def inverse_transform(model):
#         mat = model.get_matrix()
#         new_model = models.AffineModel(np.linalg.inv(mat))
#         return new_model

    def match_layers_fine_matching(self, sec1, sec2, sec1_cache, sec2_cache, sec1_to_sec2_mfovs_transforms, pool):

        starttime = time.time()
        logger.report_event("Block-Matching+PMCC layers: {} with {} (bidirectional)".format(sec1.canonical_section_name, sec2.canonical_section_name), log_level=logging.INFO)

        # take just the models (w/o the filtered match points)
        sec1_to_sec2_mfovs_transforms = {k:v[0] for k, v in sec1_to_sec2_mfovs_transforms.items()}

        # create a processes shared per-mfov transform from sec1 to sec2 (and from sec2 to sec1 too)
        mfovs1_centers_sec2centers = [[], [], []] # lists of mfovs indexes, mfovs centers, and mfovs centers after transformation to sec2
        missing_mfovs1_transforms_centers = [[], []] # lists of missing mfovs in sec1 and their centers
        for mfov1 in sec1.mfovs():
            mfov1_center = np.array([(mfov1.bbox[0] + mfov1.bbox[1])/2, (mfov1.bbox[2] + mfov1.bbox[3])/2])
            if mfov1.mfov_index in sec1_to_sec2_mfovs_transforms and sec1_to_sec2_mfovs_transforms[mfov1.mfov_index] is not None:
                mfovs1_centers_sec2centers[0].append(mfov1.mfov_index)
                mfovs1_centers_sec2centers[1].append(mfov1_center)
                sec1_mfov_model = sec1_to_sec2_mfovs_transforms[mfov1.mfov_index]
                mfovs1_centers_sec2centers[2].append(sec1_mfov_model.apply(mfov1_center)[0])
            else:
                missing_mfovs1_transforms_centers[0].append(mfov1.mfov_index)
                missing_mfovs1_transforms_centers[1].append(mfov1_center)

#         # find the transformations from sec2 to sec1
#         mfovs1_centers_sec2centers = [np.array(mfovs1_centers_sec2centers[0]), np.array(mfovs1_centers_sec2centers[1]), np.array(mfovs1_centers_sec2centers[2])]
#         mfovs1_centers_sec2_kdtree = KDTree(mfovs1_centers_sec2centers[2])
#         mfovs2_centers = [np.array([(mfov2.bbox[0] + mfov2.bbox[1])/2, (mfov2.bbox[2] + mfov2.bbox[3])/2]) for mfov2 in sec2.mfovs()]
#         mfovs2_closest_centers_mfovs1_idxs = mfovs1_centers_sec2_kdtree.query(mfovs2_centers)[1]
#         sec2_to_sec1_mfovs_transforms = {mfov2.mfov_index:
#                                             inverse_transform(
#                                                 sec1_to_sec2_mfovs_transforms[
#                                                     mfovs1_centers_sec2centers[0][mfovs2_closest_centers_mfovs1_idxs[i]]
#                                                 ]
#                                             )
#                                         for i, mfov2 in enumerate(sec2.mfovs())}
        

        # estimate the transformation for mfovs in sec1 that do not have one (look at closest neighbor)
        if len(missing_mfovs1_transforms_centers[0]) > 0:
            mfovs1_centers_sec1_kdtree = KDTree(mfovs1_centers_sec2centers[1])
            mfovs1_missing_closest_centers_mfovs1_idxs = mfovs1_centers_sec1_kdtree.query(missing_mfovs1_transforms_centers[1])[1]
            missing_mfovs1_sec2_centers = []
            for i, (mfov1_index, mfov1_closest_mfov_idx) in enumerate(zip(missing_mfovs1_transforms_centers[0], mfovs1_missing_closest_centers_mfovs1_idxs)):
                model = sec1_to_sec2_mfovs_transforms[
                            mfovs1_centers_sec2centers[0][mfov1_closest_mfov_idx]
                        ]
                sec1_to_sec2_mfovs_transforms[mfov1_index] = model
                missing_mfovs1_sec2_centers.append(model.apply(np.atleast_2d(missing_mfovs1_transforms_centers[1][i]))[0])

            # update the mfovs1_centers_sec2centers lists to include the missing mfovs and their corresponding values
            mfovs1_centers_sec2centers[0] = np.concatenate((mfovs1_centers_sec2centers[0], missing_mfovs1_transforms_centers[0]))
            mfovs1_centers_sec2centers[1] = np.concatenate((mfovs1_centers_sec2centers[1], missing_mfovs1_transforms_centers[1]))
            mfovs1_centers_sec2centers[2] = np.concatenate((mfovs1_centers_sec2centers[2], missing_mfovs1_sec2_centers))

        # Lay a grid on top of each section
        sec1_mesh_pts = utils.generate_hexagonal_grid(sec1.bbox, self._mesh_spacing)
        sec2_mesh_pts = utils.generate_hexagonal_grid(sec2.bbox, self._mesh_spacing)


        sec1_tiles_centers = [
                              [(t.bbox[0] + t.bbox[1])/2, (t.bbox[2] + t.bbox[3])/2]
                             for t in sec1.tiles()]
        sec1_tiles_centers_kdtree = KDTree(sec1_tiles_centers)
        sec1_tiles_mfov_tile_idxs = np.array([[t.mfov_index, t.tile_index] for t in sec1.tiles()])
        sec2_tiles_centers = [
                              [(t.bbox[0] + t.bbox[1])/2, (t.bbox[2] + t.bbox[3])/2]
                             for t in sec2.tiles()]
        sec2_tiles_centers_kdtree = KDTree(sec2_tiles_centers)
        sec2_tiles_mfov_tile_idxs = np.array([[t.mfov_index, t.tile_index] for t in sec2.tiles()])

        # TODO - split the work in a smart way between the processes
        # Group the mesh points of sec1 by its mfovs_tiles and make sure the points are in tiles
        sec1_mesh_pts_mfov_tile_idxs = sec1_tiles_mfov_tile_idxs[sec1_tiles_centers_kdtree.query(sec1_mesh_pts)[1]]
        sec1_per_region_mesh_pts = defaultdict(list)
        for sec1_pt, sec1_pt_mfov_tile_idx in zip(sec1_mesh_pts, sec1_mesh_pts_mfov_tile_idxs):
            sec1_tile = sec1.get_mfov(sec1_pt_mfov_tile_idx[0]).get_tile(sec1_pt_mfov_tile_idx[1])
            if BlockMatcherPMCCDispatcher._is_point_in_img(sec1_tile.bbox, sec1_pt):
                sec1_per_region_mesh_pts[tuple(sec1_pt_mfov_tile_idx)].append(sec1_pt)

        # Group the mesh pts of sec2 by the mfov on sec1 which they should end up on (mfov1 that after applying its transformation is closest to that point)
        # Transform sec1 tiles centers to their estimated location on sec2
        sec1_tiles_centers_per_mfov = defaultdict(list)
        for sec1_tile_center, sec1_tiles_mfov_tile_idx in zip(sec1_tiles_centers, sec1_tiles_mfov_tile_idxs):
            sec1_tiles_centers_per_mfov[sec1_tiles_mfov_tile_idx[0]].append(sec1_tile_center)
        sec1_tiles_centers_on_sec2 = [
                                        sec1_to_sec2_mfovs_transforms[mfov_index].apply(np.atleast_2d(mfov1_tiles_centers))
                                     for mfov_index, mfov1_tiles_centers in sec1_tiles_centers_per_mfov.items()
                                     ]
        sec1_tiles_centers_on_sec2 = np.vstack(tuple(sec1_tiles_centers_on_sec2))

        sec1_tiles_centers_on_sec2_kdtree = KDTree(sec1_tiles_centers_on_sec2)
        sec2_mesh_pts_sec1_closest_tile_idxs = sec1_tiles_centers_on_sec2_kdtree.query(sec2_mesh_pts)[1]
        sec2_mesh_pts_mfov_tile_idxs = sec2_tiles_mfov_tile_idxs[sec2_tiles_centers_kdtree.query(sec2_mesh_pts)[1]]
        sec2_per_region1_mesh_pts = defaultdict(list)
        for sec2_pt, (sec2_pt_mfov_idx, sec2_pt_tile_idx), sec1_tile_center_idx in zip(sec2_mesh_pts, sec2_mesh_pts_mfov_tile_idxs, sec2_mesh_pts_sec1_closest_tile_idxs):
            sec2_tile = sec2.get_mfov(sec2_pt_mfov_idx).get_tile(sec2_pt_tile_idx)
            if BlockMatcherPMCCDispatcher._is_point_in_img(sec2_tile.bbox, sec2_pt):
                sec2_per_region1_mesh_pts[tuple(sec1_tiles_mfov_tile_idxs[sec1_tile_center_idx])].append(sec2_pt)


        # Activate the actual matching        
        sec1_to_sec2_results = [[], []]
        sec2_to_sec1_results = [[], []]
        pool_results = []
        for region1_key, sec1_region_mesh_pts in sec1_per_region_mesh_pts.items():
            sec2_mesh_pts_cur_sec1_region = sec2_per_region1_mesh_pts[region1_key]
            #sec1_sec2_mfov_matches, sec2_sec1_mfov_matches = BlockMatcherPMCCDispatcher._perform_matching(sec1_mfov_index, sec1, sec2, sec1_to_sec2_mfovs_transforms[sec1_mfov_index], sec1_mfov_mesh_pts, sec2_mesh_pts_cur_sec1_mfov, self._debug_dir, **self._matcher_kwargs)
            res_pool = pool.apply_async(BlockMatcherPMCCDispatcher._perform_matching, (region1_key, sec1, sec2, sec1_to_sec2_mfovs_transforms[region1_key[0]], sec1_region_mesh_pts, sec2_mesh_pts_cur_sec1_region, self._debug_dir, self._matcher_kwargs))
            pool_results.append(res_pool)

        for res_pool in pool_results:
            sec1_region_index, sec1_sec2_region_matches, sec2_sec1_region_matches = res_pool.get()

            if len(sec1_sec2_region_matches[0]) > 0:
                sec1_to_sec2_results[0].append(sec1_sec2_region_matches[0])
                sec1_to_sec2_results[1].append(sec1_sec2_region_matches[1])
            if len(sec2_sec1_region_matches[0]) > 0:
                sec2_to_sec1_results[0].append(sec2_sec1_region_matches[0])
                sec2_to_sec1_results[1].append(sec2_sec1_region_matches[1])

        return np.array([np.vstack(sec1_to_sec2_results[0]), np.vstack(sec1_to_sec2_results[1])]), np.array([np.vstack(sec2_to_sec1_results[0]), np.vstack(sec2_to_sec1_results[1])])
        

