import numpy as np
from rh_logger.api import logger
import logging
import rh_logger

class UseMicroscopeCoordinates(object):

    def __init__(self, **kwargs):
        self._missing_matches = {}

        # TODO - add policy (inner mofv or also between mfov)
        self._intra_mfov_only = kwargs.get("intra_mfov_only", True)

    def add_missing_match(self, match):
        self._missing_matches.update(match)

    def fix_missing_matches(self, cur_matches):
        # If there are no missing matches, return an empty map
        if len(self._missing_matches) == 0:
            return {}

        # Add missing matches
        new_matches = {}

        for missing_match_k, missing_match_v in self._missing_matches.items():
            # missing_match_k = (tile1_unique_idx, tile2_unique_idx)
            # missing_match_v = (tile1, tile2)
            tile1, tile2 = missing_match_v


            if not self._intra_mfov_only or tile1.mfov_index == tile2.mfov_index:
            #if filtered_matches is None:
                logger.report_event("Adding fake matches between: {} and {}".format((tile1.mfov_index, tile1.tile_index), (tile2.mfov_index, tile2.tile_index)), log_level=logging.INFO)
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
                fake_new_matches = np.array([
                        fake_match_points_global - np.array([bbox1[0], bbox1[2]]),
                        fake_match_points_global - np.array([bbox2[0], bbox2[2]])
                    ])
                new_matches[missing_match_k] = fake_new_matches

        return new_matches

    def reset(self):
        self._missing_matches = {}


