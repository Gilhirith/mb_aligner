from __future__ import print_function
'''
Recevies matches (entire matches for pair of sections),
and finds for each match (p1, p2), whether p2 agrees with the target points of the surrounding matches of p1.
'''
from rh_logger.api import logger
import rh_logger
import logging
import numpy as np
import tinyr
from mb_aligner.common import ransac
import argparse
from collections import defaultdict


class FineMatchesAffineSupportFilter(object):

    def __init__(self, **kwargs):
        if kwargs is None:
            kwargs = {}
        self._support_radius = kwargs.get("support_radius", 7500)
        self._model_index = kwargs.get("model_index", 3) # Affine
        self._min_matches = kwargs.get("min_matches", 3)
        self._iterations = kwargs.get("iterations", 50)
        self._max_epsilon = kwargs.get("max_epsilon", 30)
        self._min_inlier_ratio = kwargs.get("min_inlier_ratio", 0)
        self._min_num_inlier = kwargs.get("min_num_inlier", 3)
        self._max_trust = kwargs.get("max_trust", 3)
        self._det_delta = kwargs.get("det_delta", None)
        self._max_stretch = kwargs.get("max_stretch", None)
        self._robust_filter = True if "robust_filter" in kwargs else False
        self._window_size = 2
        #print("self._max_epsilon", self._max_epsilon)
        assert(self._support_radius > 1)
        #assert(self._min_matches >= 3)

    def _run_ransac(self, matches):
        # TODO: make the parameters modifiable from the c'tor
        model, filtered_matches, filtered_matches_mask = ransac.filter_matches(matches, matches, self._model_index, self._iterations, self._max_epsilon, self._min_inlier_ratio, self._min_num_inlier, self._max_trust, self._det_delta, self._max_stretch, robust_filter=self._robust_filter)

        return model, filtered_matches, filtered_matches_mask

    def filter_matches_in_buckets(self, relevant_buckets, col, row, window_size):
        all_matches = []
        #print("col: {}, row: {}".format(col, row))
        for k, val in relevant_buckets.items():
            all_matches.extend(val)

        all_matches = np.transpose(np.array(all_matches), (1,0,2))

        all_matches_relevant_mask = np.zeros((len(all_matches[0]), ), dtype=bool)
        start_idx = 0
        for k, val in relevant_buckets.items():
            mask_val = (col <= k[0] < col + window_size) & (row <= k[1] < row + window_size)
            #print("K: {}, mask_val: {}, len(val): {}".format(k, mask_val, len(val)))
            all_matches_relevant_mask[start_idx:start_idx + len(val)] = mask_val
            start_idx += len(val)

        # Build an r-tree of all the matches source points
        all_matches_rtree = tinyr.RTree(interleaved=True, max_cap=5, min_cap=2)
        for m_id, m_src_pt in enumerate(all_matches[0]):
            # using the (x_min, y_min, x_max, y_max) notation
            all_matches_rtree.insert(m_id, (m_src_pt[0], m_src_pt[1], m_src_pt[0]+1, m_src_pt[1]+1))

        # Only search for the matches in the buckets that are part of the windows (not the ones on the borders, if were added)
        matches_mask = np.zeros((len(all_matches[0]), ), dtype=bool)
        # For each match, search around for all matches in the supprt radius, compute the defined affine transform of these,
        # and points, and if it is part of the filtered matches keep it
        fail1_cnt = 0
        fail2_cnt = 0
        fail3_cnt = 0
        #total_cnt = 0
        #success_cnt = 0
        for m_id, (m_src_pt, mask_val) in enumerate(zip(all_matches[0], all_matches_relevant_mask)):
            if mask_val: # only if the match is of a valid bucket
                #total_cnt += 1
                rect_res = all_matches_rtree.search( (m_src_pt[0] - self._support_radius, m_src_pt[1] - self._support_radius, m_src_pt[0] + self._support_radius, m_src_pt[1] + self._support_radius) )
                m_other_ids = [m_other_id for m_other_id in rect_res]
                #logger.report_event("{}: found neighbors#:{}".format(m_id, len(m_other_ids)), log_level=logging.DEBUG)
                if len(m_other_ids) < self._min_matches:
                    # There aren't enough matches in the radius, filter out the match
                    #print("fail 1")
                    fail1_cnt += 1
                    continue
                support_matches = np.array([
                    all_matches[0][m_other_ids],
                    all_matches[1][m_other_ids]
                ])
                model, support_matches_filtered, support_matches_mask = self._run_ransac(support_matches)
                if model is None:
                    # Couldn't find a valid model, filter out the match point
                    fail2_cnt += 1
                    continue
                #logger.report_event("m_src_pt: {}, support_matches_filtered[0]: {}".format(m_src_pt, support_matches_filtered[0]), log_level=logging.DEBUG)
                # make sure the point is part of the valid model support matches
                # (checking that m_src_pt is inside support_matches_filtered[0])
                if not np.any(np.all(np.abs(support_matches_filtered[0] - m_src_pt) <= 0.0001, axis=1)):
                    fail3_cnt += 1
                    continue
    #             if support_matches_mask[m_id] == False:
    #                 fail3_cnt += 1
    #                 continue
                #logger.report_event("{}: model: {}".format(m_id, model.get_matrix()), log_level=logging.DEBUG)

                matches_mask[m_id] = True
                #success_cnt += 1

        #logger.report_event("col {}, row {} summary: total_cnt: {}, success_cnt: {}".format(col, row, total_cnt, success_cnt), log_level=logging.DEBUG)
        return all_matches[:, matches_mask, :], np.array([fail1_cnt, fail2_cnt, fail3_cnt])



    def filter_matches(self, in_matches, pool):
        assert(in_matches.shape[0] == 2)
        assert(in_matches.shape[2] == 2)


        # Build a grid of all source matches, where each bucket is of size support_radius**2
        grid = defaultdict(list)

        for m_src_pt, m_dst_pt in zip(in_matches[0], in_matches[1]):
            key = tuple((m_src_pt / self._support_radius).astype(int))
            grid[key].append((m_src_pt, m_dst_pt))


        # Find the min/max x,y of the keys
        grid_keys = np.array(list(grid.keys()))
        keys_min_xy = np.min(grid_keys, axis=0)
        keys_max_xy = np.max(grid_keys, axis=0) + 1

        async_res = []
        all_valid_matches = [[], []]
        sum_fail_counters = np.zeros((3, ), dtype=np.int)
        # Split the work to window_size*window_size for each process/thread (add 1 the the left/top and 1 to the bottom/right)
        for row in range(keys_min_xy[1], keys_max_xy[1], self._window_size):
            cur_min_y = row - 1
            cur_max_y = row + self._window_size + 1
            for col in range(keys_min_xy[0], keys_max_xy[0], self._window_size):
                cur_min_x = col - 1
                cur_max_x = col + self._window_size + 1

                # collect all the buckets in the range and submit the job to a process
                relevant_buckets = {k:v for k, v in grid.items() if cur_min_x <= k[0] < cur_max_x and cur_min_y <= k[1] < cur_max_y}

                if len(relevant_buckets) > 0:
#                     valid_matches, fail_counters = self.filter_matches_in_buckets(relevant_buckets, col, row, self._window_size)
#                     if len(valid_matches[0]) > 0:
#                         all_valid_matches[0].extend(valid_matches[0])
#                         all_valid_matches[1].extend(valid_matches[1])
#                     sum_fail_counters += fail_counters
                    res = pool.apply_async(FineMatchesAffineSupportFilter.filter_matches_in_buckets, (self, relevant_buckets, col, row, self._window_size))
                    async_res.append(res)

        for res in async_res:
            valid_matches, fail_counters = res.get()
            if len(valid_matches[0]) > 0:
                all_valid_matches[0].extend(valid_matches[0])
                all_valid_matches[1].extend(valid_matches[1])
            sum_fail_counters += fail_counters
        
        logger.report_event("Parsed {} matches, {} didn't have enough matches in the radius, {} failed finding good model for, {} had a surrounding model but weren't part of that".format(len(in_matches[0]), *sum_fail_counters), log_level=logging.INFO)

        return np.array(all_valid_matches)


