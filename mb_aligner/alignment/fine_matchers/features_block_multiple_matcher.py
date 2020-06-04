# Setup
from __future__ import print_function
from rh_logger.api import logger
import rh_logger
import logging
import os
import numpy as np
import time
import sys
from scipy import spatial
import cv2
import argparse
from mb_aligner.common import utils
from rh_renderer import models
import multiprocessing as mp
from mb_aligner.common.thread_local_storage_lru import ThreadLocalStorageLRU
from scipy.spatial import cKDTree as KDTree
from collections import defaultdict
from mb_aligner.common.detector import FeaturesDetector
from mb_aligner.common.matcher import FeaturesMatcher
#from mb_aligner.common.grid_dict import GridDict
import tinyr



# import pyximport
# pyximport.install()
# from ..common import cv_wrap_module


class FeaturesBlockMultipleMatcherDispatcher(object):

    DETECTOR_KEY = "features_multiple_matcher_detector"
    BLOCK_FEATURES_KEY = "block_features"

    class FeaturesBlockMultipleMatcher(object):
        def __init__(self, sec1, sec2, sec1_to_sec2_transform, sec1_cache_features, sec2_cache_features, **kwargs):
            #self._scaling = kwargs.get("scaling", 0.2)
            self._template_size = kwargs.get("template_size", 200)
            self._search_window_size = kwargs.get("search_window_size", 8 * self._template_size)
            #logger.report_event("Actual template size: {} and window search size: {}".format(self._template_size, self._search_window_size), log_level=logging.INFO)

            # Parameters for PMCC filtering
            # self._min_corr = kwargs.get("min_correlation", 0.2)
            # self._max_curvature = kwargs.get("maximal_curvature_ratio", 10)
            # self._max_rod = kwargs.get("maximal_ROD", 0.9)
            # self._use_clahe = kwargs.get("use_clahe", False)
            # if self._use_clahe:
            #     self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

            #self._debug_dir = kwargs.get("debug_dir", None)
            self._debug_save_matches = None

            detector_type = kwargs.get("detector_type", FeaturesDetector.Type.ORB.name)
            #self._detector = FeaturesDetector(detector_type, **kwargs.get("detector_params", {}))
            self._matcher = FeaturesMatcher(FeaturesDetector.get_matcher_init_fn(detector_type), **kwargs.get("matcher_params", {}))

            self._template_side = self._template_size / 2
            self._search_window_side = self._search_window_size / 2

            self._sec1 = sec1
            self._sec2 = sec2
            self._sec1_to_sec2_transform = sec1_to_sec2_transform
            self._inverse_model = FeaturesBlockMultipleMatcherDispatcher.FeaturesBlockMultipleMatcher.inverse_transform(self._sec1_to_sec2_transform)
            
            self._sec1_cache_features = sec1_cache_features
            self._sec2_cache_features = sec2_cache_features

            # Create an rtree for each section's tiles, to quickly find the relevant tiles
            self._sec1_tiles_rtree = FeaturesBlockMultipleMatcherDispatcher.FeaturesBlockMultipleMatcher._create_tiles_bbox_rtree(sec1)
            self._sec2_tiles_rtree = FeaturesBlockMultipleMatcherDispatcher.FeaturesBlockMultipleMatcher._create_tiles_bbox_rtree(sec2)
            
            ####self._sec1_scaled_renderer.add_transformation(self._sec1_to_sec2_transform.get_matrix())



        @staticmethod
        def _create_tiles_bbox_rtree(sec):
            sec_tiles_rtree = tinyr.RTree(interleaved=False, max_cap=5, min_cap=2)
            for t_idx, t in enumerate(sec.tiles()):
                bbox = t.bbox
                # using the (x_min, x_max, y_min, y_max) notation
                sec_tiles_rtree.insert(t_idx, bbox)
            return sec_tiles_rtree
            
        def set_debug_dir(self, debug_dir):
            self._debug_save_matches = True
            self._debug_dir = debug_dir

        @staticmethod
        def inverse_transform(model):
            mat = model.get_matrix()
            new_model = models.AffineModel(np.linalg.inv(mat))
            return new_model


        @staticmethod
        def _fetch_sec_features(sec, sec_tiles_rtree, sec_cache_features, bbox):
            relevant_features = [[], []]
            rect_res = sec_tiles_rtree.search(bbox)
            for t_idx in rect_res:
                k = "{}_t{}".format(sec.canonical_section_name, t_idx)
                assert(k in sec_cache_features)
                tile_features_kps, tile_features_descs = sec_cache_features[k]
                # find all the features that are overlap with the bbox
                bbox_mask = (tile_features_kps[:, 0] >= bbox[0]) & (tile_features_kps[:, 0] <= bbox[1]) &\
                            (tile_features_kps[:, 1] >= bbox[2]) & (tile_features_kps[:, 1] <= bbox[3])
                if np.any(bbox_mask):
                    relevant_features[0].append(tile_features_kps[bbox_mask])
                    relevant_features[1].append(tile_features_descs[bbox_mask])
            if len(relevant_features[0]) == 0:
                return relevant_features # [[], []]
            return np.vstack(relevant_features[0]), np.vstack(relevant_features[1])
            
        
        def match_sec1_to_sec2_mfov(self, sec1_pts):
            valid_matches = [[], [], []]
            invalid_matches = [[], []]
            if len(sec1_pts) == 0:
                return valid_matches, invalid_matches

            sec1_pts = np.atleast_2d(sec1_pts)
                
            # Apply the mfov transformation to compute estimated location on sec2
            sec1_mfov_pts_on_sec2 = self._sec1_to_sec2_transform.apply(sec1_pts)

            for sec1_pt, sec2_pt_estimated in zip(sec1_pts, sec1_mfov_pts_on_sec2):

                # Fetch the template around sec1_point (before transformation)
                from_x1, from_y1 = sec1_pt - self._template_side
                to_x1, to_y1 = sec1_pt + self._template_side
                sec1_pt_features_kps, sec1_pt_features_descs = FeaturesBlockMultipleMatcherDispatcher.FeaturesBlockMultipleMatcher._fetch_sec_features(self._sec1, self._sec1_tiles_rtree, self._sec1_cache_features, (from_x1, to_x1, from_y1, to_y1))

                if len(sec1_pt_features_kps) <= 1:
                    continue
            
                # Fetch a large sub-image around img2_point (using search_window_scaled_size)
                from_x2, from_y2 = sec2_pt_estimated - self._search_window_side
                to_x2, to_y2 = sec2_pt_estimated + self._search_window_side
                sec2_pt_est_features_kps, sec2_pt_est_features_descs = FeaturesBlockMultipleMatcherDispatcher.FeaturesBlockMultipleMatcher._fetch_sec_features(self._sec2, self._sec2_tiles_rtree, self._sec2_cache_features, (from_x2, to_x2, from_y2, to_y2))
        
                if len(sec2_pt_est_features_kps) <= 1:
                    continue

                # apply the transformation on sec1 feature points locations
                sec1_pt_features_kps = self._sec1_to_sec2_transform.apply(sec1_pt_features_kps)
                # Match the features
                transform_model, filtered_matches = self._matcher.match_and_filter(sec1_pt_features_kps, sec1_pt_features_descs, sec2_pt_est_features_kps, sec2_pt_est_features_descs)
                if transform_model is None:
                    invalid_matches[0].append(sec1_pt)
                    invalid_matches[1].append(1)
                else:
                    logger.report_event("{}: match found around area: {} (sec1) and {} (sec2) with {} matches".format(os.getpid(), sec1_pt, sec2_pt_estimated, len(filtered_matches[0])), log_level=logging.DEBUG)
                    # Take the matched points and apply the inverse transform on sec1_pts to get them back to sec1 locations
                    matches_pts_sec1 = self._inverse_model.apply(filtered_matches[0])
                    # add all the matched points
                    valid_matches[0].extend(matches_pts_sec1)
                    valid_matches[1].extend(filtered_matches[1])
                    valid_matches[2].extend([len(filtered_matches[0]) / len(sec1_pt_features_kps)] * len(matches_pts_sec1))



#                 if sec1_template.shape[0] >= sec2_search_window.shape[0] or sec1_template.shape[1] >= sec2_search_window.shape[1]:
#                     continue
#                 if self._use_clahe:
#                     sec2_search_window_clahe = self._clahe.apply(sec2_search_window)
#                     sec1_template_clahe = self._clahe.apply(sec1_template)
#                     pmcc_result, reason, match_val = PMCC_filter.PMCC_match(sec2_search_window_clahe, sec1_template_clahe, min_correlation=self._min_corr, maximal_curvature_ratio=self._max_curvature, maximal_ROD=self._max_rod)
#                 else:
#                     pmcc_result, reason, match_val = PMCC_filter.PMCC_match(sec2_search_window, sec1_template, min_correlation=self._min_corr, maximal_curvature_ratio=self._max_curvature, maximal_ROD=self._max_rod)
# 
#                 if pmcc_result is None:
#                     invalid_matches[0].append(sec1_pt)
#                     invalid_matches[1].append(reason)
# #                     debug_out_fname1 = "temp_debug/debug_match_sec1{}-{}_template.png".format(int(sec1_pt[0]), int(sec1_pt[1]), int(sec2_pt_estimated[0]), int(sec2_pt_estimated[1]))
# #                     debug_out_fname2 = "temp_debug/debug_match_sec1{}-{}_search_window.png".format(int(sec1_pt[0]), int(sec1_pt[1]), int(sec2_pt_estimated[0]), int(sec2_pt_estimated[1]))
# #                     cv2.imwrite(debug_out_fname1, sec1_template)
# #                     cv2.imwrite(debug_out_fname2, sec2_search_window)
#                 else:
#                     # Compute the location of the matched point on img2 in non-scaled coordinates
#                     matched_location_scaled = np.array([reason[1], reason[0]]) + np.array([from_x2, from_y2]) + self._template_scaled_side
#                     sec2_pt = matched_location_scaled / self._scaling 
#                     logger.report_event("{}: match found: {} and {} (orig assumption: {})".format(os.getpid(), sec1_pt, sec2_pt, sec2_pt_estimated / self._scaling), log_level=logging.DEBUG)
#                     if self._debug_save_matches:
#                         debug_out_fname1 = os.path.join(self._debug_dir, "debug_match_sec1_{}-{}_sec2_{}-{}_image1.png".format(int(sec1_pt[0]), int(sec1_pt[1]), int(sec2_pt[0]), int(sec2_pt[1])))
#                         debug_out_fname2 = os.path.join(self._debug_dir, "debug_match_sec1_{}-{}_sec2_{}-{}_image2.png".format(int(sec1_pt[0]), int(sec1_pt[1]), int(sec2_pt[0]), int(sec2_pt[1])))
#                         cv2.imwrite(debug_out_fname1, sec1_template)
#                         sec2_cut_out = sec2_search_window[int(reason[0]):int(reason[0] + 2 * self._template_scaled_side), int(reason[1]):int(reason[1] + 2 * self._template_scaled_side)]
#                         cv2.imwrite(debug_out_fname2, sec2_cut_out)
#                     valid_matches[0].append(np.array(sec1_pt))
#                     valid_matches[1].append(sec2_pt)
#                     valid_matches[2].append(match_val)

            return valid_matches, invalid_matches
        

        def match_sec2_to_sec1_mfov(self, sec2_pts):
            valid_matches = [[], [], []]
            invalid_matches = [[], []]
            if len(sec2_pts) == 0:
                return valid_matches, invalid_matches

            # Assume that only sec1 renderer was transformed and not sec2 (and both scaled)
            sec2_pts = np.atleast_2d(sec2_pts)

            #mat = self._sec1_to_sec2_transform.get_matrix()
            #inverse_mat = np.linalg.inv(mat)
 
            sec2_pts_on_sec1 = self._inverse_model.apply(sec2_pts)

            for sec2_pt, sec1_pt_estimated in zip(sec2_pts, sec2_pts_on_sec1):

                # Fetch the template around sec2_pt
                from_x2, from_y2 = sec2_pt - self._template_side
                to_x2, to_y2 = sec2_pt + self._template_side
                sec2_pt_features_kps, sec2_pt_features_descs = FeaturesBlockMultipleMatcherDispatcher.FeaturesBlockMultipleMatcher._fetch_sec_features(self._sec2, self._sec2_tiles_rtree, self._sec2_cache_features, (from_x2, to_x2, from_y2, to_y2))
            
                if len(sec2_pt_features_kps) <= 1:
                    continue

                # Fetch a large sub-image around sec1_pt_estimated (after transformation, using search_window_scaled_size)
                from_x1, from_y1 = sec1_pt_estimated - self._search_window_side
                to_x1, to_y1 = sec1_pt_estimated + self._search_window_side
                sec1_pt_est_features_kps, sec1_pt_est_features_descs = FeaturesBlockMultipleMatcherDispatcher.FeaturesBlockMultipleMatcher._fetch_sec_features(self._sec1, self._sec1_tiles_rtree, self._sec1_cache_features, (from_x1, to_x1, from_y1, to_y1))

                if len(sec1_pt_est_features_kps) <= 1:
                    continue

                # apply the inverse transformation on sec2 feature points locations
                sec2_pt_features_kps = self._inverse_model.apply(sec2_pt_features_kps)
                # Match the features
                transform_model, filtered_matches = self._matcher.match_and_filter(sec2_pt_features_kps, sec2_pt_features_descs, sec1_pt_est_features_kps, sec1_pt_est_features_descs)
                if transform_model is None:
                    invalid_matches[0].append(sec2_pt)
                    invalid_matches[1].append(1)
                else:
                    logger.report_event("{}: match found around area: {} (sec2) and {} (sec1) with {} matches".format(os.getpid(), sec2_pt, sec1_pt_estimated, len(filtered_matches[0])), log_level=logging.DEBUG)
                    # Take the matched points and apply the inverse transform on sec1_pts to get them back to sec1 locations
                    matches_pts_sec2 = self._sec1_to_sec2_transform.apply(filtered_matches[0])
                    # add all the matched points
                    valid_matches[0].extend(matches_pts_sec2)
                    valid_matches[1].extend(filtered_matches[1])
                    valid_matches[2].extend([len(filtered_matches[0]) / len(sec2_pt_features_kps)] * len(matches_pts_sec2))


            return valid_matches, invalid_matches
        

    def __init__(self, **kwargs):

        self._kwargs = kwargs
        self._mesh_spacing = kwargs.get("mesh_spacing", 1500)

        self._template_size = kwargs.get("template_size", 200)
        self._search_window_size = kwargs.get("search_window_size", 8 * self._template_size)
        logger.report_event("Actual template size: {} and window search size: {}".format(self._template_size, self._search_window_size), log_level=logging.INFO)

        self._detector_type = kwargs.get("detector_type", FeaturesDetector.Type.ORB.name)
        self._detector_kwargs = kwargs.get("detector_params", {})
        self._matcher_kwargs = kwargs.get("matcher_params", {})

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
    def _perform_matching(sec1_mfov_tile_idx, sec1, sec2, sec1_to_sec2_mfov_transform, sec1_cache_features, sec2_cache_features, sec1_mfov_mesh_pts, sec2_mfov_mesh_pts, debug_dir, matcher_args):
#         fine_matcher_key = "block_matcher_{},{},{}".format(sec1.canonical_section_name, sec2.canonical_section_name, sec1_mfov_tile_idx[0])
#         fine_matcher = getattr(threadLocal, fine_matcher_key, None)
#         if fine_matcher is None:
#             fine_matcher = BlockMatcherPMCCDispatcher.BlockMatcherPMCC(sec1, sec2, sec1_to_sec2_mfov_transform, **matcher_args)
#             if debug_dir is not None:
#                 fine_matcher.set_debug_dir(debug_dir)
# 
#             setattr(threadLocal, fine_matcher_key, fine_matcher)

        fine_matcher_key = "block_matcher_{},{},{}".format(sec1.canonical_section_name, sec2.canonical_section_name, sec1_mfov_tile_idx[0])
        thread_local_store = ThreadLocalStorageLRU()
        if fine_matcher_key in thread_local_store.keys():
            fine_matcher = thread_local_store[fine_matcher_key]
        else:
            fine_matcher = FeaturesBlockMultipleMatcherDispatcher.FeaturesBlockMultipleMatcher(sec1, sec2, sec1_to_sec2_mfov_transform, sec1_cache_features, sec2_cache_features, **matcher_args)
            if debug_dir is not None:
                fine_matcher.set_debug_dir(debug_dir)
            thread_local_store[fine_matcher_key] = fine_matcher

#         fine_matcher = getattr(threadLocal, fine_matcher_key, None)
#         if fine_matcher is None:
#             fine_matcher = FeaturesBlockMultipleMatcherDispatcher.FeaturesBlockMultipleMatcher(sec1, sec2, sec1_to_sec2_mfov_transform, sec1_cache_features, sec2_cache_features, **matcher_args)
#             if debug_dir is not None:
#                 fine_matcher.set_debug_dir(debug_dir)
#             setattr(threadLocal, fine_matcher_key, fine_matcher)

        logger.report_event("Features-Matching+PMCC layers: {} with {} (mfov1 {}) {} mesh points1, {} mesh points2".format(sec1.canonical_section_name, sec2.canonical_section_name, sec1_mfov_tile_idx, len(sec1_mfov_mesh_pts), len(sec2_mfov_mesh_pts)), log_level=logging.INFO)
        logger.report_event("Features-Matching+PMCC layers: {} -> {}".format(sec1.canonical_section_name, sec2.canonical_section_name), log_level=logging.INFO)
        valid_matches1, invalid_matches1 = fine_matcher.match_sec1_to_sec2_mfov(sec1_mfov_mesh_pts)
        logger.report_event("Features-Matching+PMCC layers: {} -> {} valid matches: {}, invalid_matches: {} {}".format(sec1.canonical_section_name, sec2.canonical_section_name, len(valid_matches1[0]), len(invalid_matches1[0]), FeaturesBlockMultipleMatcherDispatcher.sum_invalid_matches(invalid_matches1)), log_level=logging.INFO)

        logger.report_event("Features-Matching+PMCC layers: {} <- {}".format(sec1.canonical_section_name, sec2.canonical_section_name), log_level=logging.INFO)
        valid_matches2, invalid_matches2 = fine_matcher.match_sec2_to_sec1_mfov(sec2_mfov_mesh_pts)
        logger.report_event("Features-Matching+PMCC layers: {} <- {} valid matches: {}, invalid_matches: {} {}".format(sec1.canonical_section_name, sec2.canonical_section_name, len(valid_matches2[0]), len(invalid_matches2[0]), FeaturesBlockMultipleMatcherDispatcher.sum_invalid_matches(invalid_matches2)), log_level=logging.INFO)

        return sec1_mfov_tile_idx, valid_matches1, valid_matches2

#     def inverse_transform(model):
#         mat = model.get_matrix()
#         new_model = models.AffineModel(np.linalg.inv(mat))
#         return new_model

    @staticmethod
    def compute_features(tile, out_dict, out_dict_key, detector_type, detector_kwargs):
        thread_local_store = ThreadLocalStorageLRU()
        if FeaturesBlockMultipleMatcherDispatcher.DETECTOR_KEY in thread_local_store.keys():
            detector = thread_local_store[FeaturesBlockMultipleMatcherDispatcher.DETECTOR_KEY]
        else:
            #detector_type = FeaturesDetector.Type.ORB.name
            detector = FeaturesDetector(detector_type, **detector_kwargs)
            thread_local_store[FeaturesBlockMultipleMatcherDispatcher.DETECTOR_KEY] = detector

#         detector = getattr(threadLocal, FeaturesBlockMultipleMatcherDispatcher.DETECTOR_KEY, None)
#         if detector is None:
#             #detector_type = FeaturesDetector.Type.ORB.name
#             detector = FeaturesDetector(detector_type, **detector_kwargs)
# 
#             setattr(threadLocal, FeaturesBlockMultipleMatcherDispatcher.DETECTOR_KEY, detector)

        # Load the image
        img = tile.image

        # compute features
        kps, descs = detector.detect(img)

        # Replace the keypoints with a numpy array
        kps_pts = np.empty((len(kps), 2), dtype=np.float64)
        for kp_i, kp in enumerate(kps):
            kps_pts[kp_i][:] = kp.pt

        # Apply tile transformations to each point
        for transform in tile.transforms:
            kps_pts = transform.apply(kps_pts)

        out_dict[out_dict_key] = [kps_pts, np.array(descs)]


#     @staticmethod
#     def create_grid_from_pts(pts, grid_size):
#         """
#         Given 2D points and a grid_size, gathers all points into a dictionary that maps between
#         pt[0] // grid_size, pt[1] // grid_size    to    a numpy array of the points that land in that bucket
#         """
#         grid = defaultdict(list)
# 
#         locations = (pts / grid_size).astype(int)
#         for loc, pt in zip(locations, pts):
#             grid[loc].append(pt)
# 
#         for k, pts_list in grid.items():
#             grid[k] = np.array(pts_list)
# 
#         return grid

    def match_layers_fine_matching(self, sec1, sec2, sec1_cache, sec2_cache, sec1_to_sec2_mfovs_transforms, pool):

        starttime = time.time()
        logger.report_event("Features-Matching+PMCC layers: {} with {} (bidirectional)".format(sec1.canonical_section_name, sec2.canonical_section_name), log_level=logging.INFO)

        # take just the models (w/o the filtered match points)
        sec1_to_sec2_mfovs_transforms = {k:v[0] for k, v in sec1_to_sec2_mfovs_transforms.items()}

        if FeaturesBlockMultipleMatcherDispatcher.BLOCK_FEATURES_KEY not in sec1_cache:
            sec1_cache[FeaturesBlockMultipleMatcherDispatcher.BLOCK_FEATURES_KEY] = {}
        if FeaturesBlockMultipleMatcherDispatcher.BLOCK_FEATURES_KEY not in sec2_cache:
            sec2_cache[FeaturesBlockMultipleMatcherDispatcher.BLOCK_FEATURES_KEY] = {}

        # For each section, detect the features for each tile, and transform them to their stitched location (store in cache for future comparisons)
        logger.report_event("Computing per tile block features", log_level=logging.INFO)
        pool_results = []
        for sec1_t_idx, t in enumerate(sec1.tiles()):
            k = "{}_t{}".format(sec1.canonical_section_name, sec1_t_idx)
            if k not in sec1_cache[FeaturesBlockMultipleMatcherDispatcher.BLOCK_FEATURES_KEY]:
                res = pool.apply_async(FeaturesBlockMultipleMatcherDispatcher.compute_features, (t, sec1_cache[FeaturesBlockMultipleMatcherDispatcher.BLOCK_FEATURES_KEY], k, self._detector_type, self._detector_kwargs))
                pool_results.append(res)
        for sec2_t_idx, t in enumerate(sec2.tiles()):
            k = "{}_t{}".format(sec2.canonical_section_name, sec2_t_idx)
            if k not in sec2_cache[FeaturesBlockMultipleMatcherDispatcher.BLOCK_FEATURES_KEY]:
                res = pool.apply_async(FeaturesBlockMultipleMatcherDispatcher.compute_features, (t, sec2_cache[FeaturesBlockMultipleMatcherDispatcher.BLOCK_FEATURES_KEY], k, self._detector_type, self._detector_kwargs))
                pool_results.append(res)

        for res in pool_results:
            res.get()

        logger.report_event("Computing missing mfovs transformations", log_level=logging.INFO)
        # find the nearest transformations for mfovs1 that are missing in sec1_to_sec2_mfovs_transforms and for sec2 to sec1
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


#         # Put all features of each section in an rtree
#         #sec1_features_rtree = tinyr.RTree(interleaved=False, max_cap=5, min_cap=2)
#         sec1_features_grid = GridDict(self._template_size)
#         #sec2_features_rtree = tinyr.RTree(interleaved=False, max_cap=5, min_cap=2)
#         sec2_features_grid = GridDict(self._template_size)
#         # using the (x_min, x_max, y_min, y_max) notation
#         for sec1_t_idx, t in enumerate(sec1.tiles()):
#             k = "{}_t{}".format(sec1.canonical_section_name, sec1_t_idx)
#             t_kps, t_descs = sec1_cache["block_features"][k]
#             #t_mfov_index = t.mfov_index
#             #t_features_kps_sec1_on_sec2 = sec1_to_sec2_mfovs_transforms[t_mfov_index].apply(t_kps)
# #             for feature_idx, t_feature_kp_sec1 in enumerate(t_kps):
# #                 sec1_features_rtree.insert([k, feature_idx], [t_feature_kp_sec1[0], t_feature_kp_sec1[0]+0.5, t_feature_kp_sec1[1], t_feature_kp_sec1[1]+0.5])
#             for t_feature_kp_sec1, t_feature_desc_sec1 in zip(t_kps, t_descs):
#                 sec1_features_grid.add(t_feature_kp_sec1, t_feature_desc_sec1)
#                 
#          for sec2_t_idx, t in enumerate(sec2.tiles()):
#             k = "{}_t{}".format(sec2.canonical_section_name, sec2_t_idx)
#             t_kps, t_descs = sec2_cache["block_features"][k]
# #             for feature_idx, t_feature_kp_sec2 in enumerate(t_kps):
# #                 sec2_features_rtree.insert([k, feature_idx], [t_feature_kp_sec2[0], t_feature_kp_sec2[0]+0.5, t_feature_kp_sec2[1], t_feature_kp_sec2[1]+0.5])
#             for t_feature_kp_sec2, t_feature_desc_sec2 in zip(t_kps, t_descs):
#                 sec2_features_grid.add(t_feature_kp_sec2, t_feature_desc_sec2)


        logger.report_event("Computing grid points and distributing work", log_level=logging.INFO)
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
            if FeaturesBlockMultipleMatcherDispatcher._is_point_in_img(sec1_tile.bbox, sec1_pt):
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
            if FeaturesBlockMultipleMatcherDispatcher._is_point_in_img(sec2_tile.bbox, sec2_pt):
                sec2_per_region1_mesh_pts[tuple(sec1_tiles_mfov_tile_idxs[sec1_tile_center_idx])].append(sec2_pt)



        # Activate the actual matching        
        sec1_to_sec2_results = [[], []]
        sec2_to_sec1_results = [[], []]
        pool_results = []
#         temp_ctr = 0
        for region1_key, sec1_region_mesh_pts in sec1_per_region_mesh_pts.items():
            sec2_mesh_pts_cur_sec1_region = sec2_per_region1_mesh_pts[region1_key]
            #sec1_sec2_mfov_matches, sec2_sec1_mfov_matches = BlockMatcherPMCCDispatcher._perform_matching(sec1_mfov_index, sec1, sec2, sec1_to_sec2_mfovs_transforms[sec1_mfov_index], sec1_mfov_mesh_pts, sec2_mesh_pts_cur_sec1_mfov, self._debug_dir, **self._matcher_kwargs)
            res_pool = pool.apply_async(FeaturesBlockMultipleMatcherDispatcher._perform_matching, (region1_key, sec1, sec2, sec1_to_sec2_mfovs_transforms[region1_key[0]], sec1_cache[FeaturesBlockMultipleMatcherDispatcher.BLOCK_FEATURES_KEY], sec2_cache[FeaturesBlockMultipleMatcherDispatcher.BLOCK_FEATURES_KEY], sec1_region_mesh_pts, sec2_mesh_pts_cur_sec1_region, self._debug_dir, self._kwargs))
            pool_results.append(res_pool)
#             temp_ctr += 1
#             if temp_ctr == 50:
#                 break

        for res_pool in pool_results:
            sec1_region_index, sec1_sec2_region_matches, sec2_sec1_region_matches = res_pool.get()


            # TODO - remove matches that start from the same place and end up in different places

            if len(sec1_sec2_region_matches[0]) > 0:
                sec1_to_sec2_results[0].append(sec1_sec2_region_matches[0])
                sec1_to_sec2_results[1].append(sec1_sec2_region_matches[1])
            if len(sec2_sec1_region_matches[0]) > 0:
                sec2_to_sec1_results[0].append(sec2_sec1_region_matches[0])
                sec2_to_sec1_results[1].append(sec2_sec1_region_matches[1])

        if len(sec1_to_sec2_results[0]) == 0:
            return [], []
        return np.array([np.vstack(sec1_to_sec2_results[0]), np.vstack(sec1_to_sec2_results[1])]), np.array([np.vstack(sec2_to_sec1_results[0]), np.vstack(sec2_to_sec1_results[1])])
        

