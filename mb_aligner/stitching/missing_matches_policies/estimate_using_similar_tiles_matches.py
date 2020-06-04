import numpy as np
from rh_logger.api import logger
import logging
import rh_logger
from collections import defaultdict
from mb_aligner.common import ransac

class EstimateUsingSimilarTilesMatches(object):

    def __init__(self, **kwargs):
        self._missing_matches = {}

        if kwargs is None:
            kwargs = {}
        self._avoid_inter_mfov = kwargs.get("avoid_inter_mfov", False)
        self._avoid_inter_mfov_2nd_degree = kwargs.get("avoid_inter_mfov_2nd_degree", False)
        self._model_index = kwargs.get("model_index", 1) # Rigid
        self._min_matches = kwargs.get("min_matches", 3)
        self._iterations = kwargs.get("iterations", 1000)
        self._max_epsilon = kwargs.get("max_epsilon", 5)
        self._min_inlier_ratio = kwargs.get("min_inlier_ratio", 0)
        self._min_num_inlier = kwargs.get("min_num_inlier", 3)
        self._max_trust = kwargs.get("max_trust", 3)
        self._det_delta = kwargs.get("det_delta", 0.99)
        self._max_stretch = kwargs.get("max_stretch", 0.99)
        self._robust_filter = True if "robust_filter" in kwargs else False

    def add_missing_match(self, match):
        self._missing_matches.update(match)

    def fix_missing_matches(self, cur_matches):
        # If there are no missing matches, return an empty map
        if len(self._missing_matches) == 0:
            return {}

        # for each pair of matching tiles (from cur_matches), aggregate all matches to create median match
        # (split by intra mfov and inter mfov)
        intra_mfov_matches = defaultdict(list)
        inter_mfov_matches = defaultdict(list)
        for cur_match_k, cur_match_v in cur_matches.items():
            # cur_match_k = (tile1_unique_idx, tile2_unique_idx)
            # cur_match_v = filtered_matches : [pts1, pts2]
            # where: tile_unique_idx = (tile.layer, tile.mfov_index, tile.tile_index)
            tile1_layer, tile1_mfov_index, tile1_tile_index = cur_match_k[0]
            tile2_layer, tile2_mfov_index, tile2_tile_index = cur_match_k[1]
            
            if tile1_mfov_index == tile2_mfov_index:
                intra_mfov_matches[tile1_tile_index, tile2_tile_index].append(cur_match_v)
            elif self._avoid_inter_mfov:
                continue
            elif self._avoid_inter_mfov_2nd_degree:
                if tile1_tile_index < 38 or tile2_tile_index < 38:
                    continue
                else:
                    inter_mfov_matches[tile1_tile_index, tile2_tile_index].append(cur_match_v)
            else:
                inter_mfov_matches[tile1_tile_index, tile2_tile_index].append(cur_match_v)


        intra_mfov_fake_matches = {}
        inter_mfov_fake_matches = {}

        # Add missing matches
        new_matches = {}

        for missing_match_k, missing_match_v in self._missing_matches.items():
            # missing_match_k = (tile1_unique_idx, tile2_unique_idx)
            # missing_match_v = (tile1, tile2)
            tile1, tile2 = missing_match_v

            # set where to look, depending whether it's intra mfov or inter mfov
            if tile1.mfov_index == tile2.mfov_index:
                fake_matches_list = intra_mfov_matches[tile1.tile_index, tile2.tile_index]
                mfov_fake_matches = intra_mfov_fake_matches
            else:
                if self._avoid_inter_mfov:
                    continue
                if self._avoid_inter_mfov_2nd_degree:
                    if tile1.tile_index < 38 or tile2.tile_index < 38:
                        continue
                fake_matches_list = inter_mfov_matches[tile1.tile_index, tile2.tile_index]
                mfov_fake_matches = inter_mfov_fake_matches
            logger.report_event("Adding fake matches between: {} and {}".format((tile1.mfov_index, tile1.tile_index), (tile2.mfov_index, tile2.tile_index)), log_level=logging.INFO)

            if (tile1.tile_index, tile2.tile_index) not in mfov_fake_matches.keys():
                # Compute the best 
                mfov_fake_matches[tile1.tile_index, tile2.tile_index] = self._compute_fake_match(fake_matches_list)[0] # only keep the model

            fake_match_model = mfov_fake_matches[tile1.tile_index, tile2.tile_index]
            if fake_match_model is None:
                continue

            bbox1 = tile1.bbox
            bbox2 = tile2.bbox

            intersection = [max(bbox1[0], bbox2[0]),
                            min(bbox1[1], bbox2[1]),
                            max(bbox1[2], bbox2[2]),
                            min(bbox1[3], bbox2[3])]
            intersection_center = np.array([intersection[0] + intersection[1], intersection[2] + intersection[3]]) * 0.5

            fake_match_points_global = np.array([
                    [intersection_center[0] + intersection[0] - 2, intersection_center[1] + intersection[2] - 2],
                    [intersection_center[0] + intersection[1] + 4, intersection_center[1] + intersection[2] - 4],
                    [intersection_center[0] + intersection[0] + 2, intersection_center[1] + intersection[3] - 2],
                    [intersection_center[0] + intersection[1] - 4, intersection_center[1] + intersection[3] - 6]
                ]) * 0.5
            fake_pts_tile1 = fake_match_points_global - np.array([bbox1[0], bbox1[2]])
            fake_pts_tile2 = fake_match_model.apply(fake_pts_tile1)

            fake_new_matches = np.array([
                    fake_pts_tile1, fake_pts_tile2
                ])
            new_matches[missing_match_k] = fake_new_matches

        return new_matches

    def _compute_fake_match(self, fake_matches_list):
        # Input: list of matches, each of the form [pts1, pts2]:
        # output: a single fake match given by concatenating the lists, and ransacing it once
        all_pts1 = []
        all_pts2 = []
        for pts1, pts2 in fake_matches_list:
            all_pts1.extend(pts1)
            all_pts2.extend(pts2)

        all_matches = np.array([all_pts1, all_pts2])
        model, filtered_matches, filtered_matches_mask = ransac.filter_matches(all_matches, all_matches, self._model_index, self._iterations, self._max_epsilon, self._min_inlier_ratio, self._min_num_inlier, self._max_trust, self._det_delta, self._max_stretch, robust_filter=self._robust_filter)

        return model, filtered_matches, filtered_matches_mask

    def reset(self):
        self._missing_matches = {}
