import numpy as np
import pickle
import os
import time
import scipy.sparse as spp
from scipy.sparse.linalg import lsqr
import scipy.optimize
from rh_renderer.models import TranslationModel
#import common

EPS = 0.000001

class TranslationDamping2DOptimizer(object):
    def __init__(self, **kwargs):
        self._damping = kwargs.get("damping", 0.0)

    @staticmethod
    def apply_translation_transform(pts, t_x, t_y):
        return pts + np.array([t_x, t_y])


    @staticmethod
    def compute_all_dists(matches, transforms, matches_num):
        dists = np.empty((matches_num, ), dtype=np.float32)
        start_idx = 0
        for pair_name, pair_matches in matches.items():
            pair_matches_len = len(pair_matches[0])

            transform1 = transforms[pair_name[0]]
            transform2 = transforms[pair_name[1]]
            pts1_transformed = TranslationDamping2DOptimizer.apply_translation_transform(pair_matches[0], *transform1)
            pts2_transformed = TranslationDamping2DOptimizer.apply_translation_transform(pair_matches[1], *transform2)

            # compute the L2 distance between the two sets of points
            deltas = pts1_transformed - pts2_transformed
            dists[start_idx:start_idx + pair_matches_len] = np.sqrt(np.sum(deltas**2, axis=1))
            start_idx += pair_matches_len
        return dists

    def optimize(self, orig_locs, matches):
        """
        The aim is to find for each tile a triplet: t_x, and t_y that will define the
        translation transformation that needs to be applied to that tile.
        The transformation needs to minimize the L2 distance between the matches of pairs of tiles.
        To this end, we define our optimizations as a non-linear least squares problem.
        Note that due to outliers, we would like to use a more robust method, such as huber loss.

        """

        tile_names = sorted(list(orig_locs.keys()))
        tile_names_map = {name:idx for idx, name in enumerate(tile_names)}
        matches_num = np.sum([len(m[0]) for m in matches.values()])

        # For debug:
        solution1 = {name:[orig_locs[name][0], orig_locs[name][1]] for name, idx in tile_names_map.items()}
        dists = TranslationDamping2DOptimizer.compute_all_dists(matches, solution1, matches_num)
        print("pre optimization distances: min={}, mean={}, median={}, max={}".format(np.min(dists), np.mean(dists), np.median(dists), np.max(dists)))

        st_time = time.time()
        # Find an initial translation only transformation for each tile (better than the initial assumption)
        # solve for X
        # Create a matrix A that is made of 1's, 0's and -1's of size matches_num*tiles_num,
        # and a vector b s.t. b = - matches[0].x + matches[1].x (actually b will be a matches_num*2 matrix, one column for x and the other for y)
        # We'll try to find x, s.t. A*x=b, and therefore each row (corresponding to a single match of a pair of tiles),
        # will have 1 for the first tile of the match, -1 for the second tile of the match, and 0 elsewhere
        #A = spp.csc_matrix( (matches_num, len(orig_locs)), dtype=np.float32 )
        A = spp.lil_matrix( (matches_num, len(orig_locs)), dtype=np.float32 )
        b = np.empty((matches_num, 2), dtype=np.float32)
        start_idx = 0
        for pair_name, pair_matches in matches.items():
            pair_matches_len = len(pair_matches[0])
            tile1_params_idx = tile_names_map[pair_name[0]]
            tile2_params_idx = tile_names_map[pair_name[1]]
            A[start_idx:start_idx + pair_matches_len, tile1_params_idx] = 1
            A[start_idx:start_idx + pair_matches_len, tile2_params_idx] = -1

            b[start_idx:start_idx + pair_matches_len] = - pair_matches[0] + pair_matches[1]

            start_idx += pair_matches_len
        # convert A to row sparse matrix, for faster computations
        A = A.tocsr()

        #p0_translate_x = np.array([orig_locs[k][0] for k in tile_names]) # [t_x1, t_x2, ...] with the original locations
        Tx = lsqr(A, b[:, 0], damp=self._damping)[0]
        Ty = lsqr(A, b[:, 1], damp=self._damping)[0]
        print("translation-only optimization time: {} seconds".format(time.time() - st_time))
        # Normalize all deltas to (0, 0)
        Tx -= np.min(Tx)
        Ty -= np.min(Ty)

        # For debug:
        #solution2 = {name:[p0[::2][idx], p0[1::2][idx]] for name, idx in tile_names_map.items()}
        solution2 = {name:[Tx[idx], Ty[idx]] for name, idx in tile_names_map.items()}
        dists = TranslationDamping2DOptimizer.compute_all_dists(matches, solution2, matches_num)
        print("post translation optimization distances: min={}, mean={}, median={}, max={}".format(np.min(dists), np.mean(dists), np.median(dists), np.max(dists)))

        # create the optimized models for each tile
        optimized_models = {name:TranslationModel([Tx[idx], Ty[idx]]) for name, idx in tile_names_map.items()}
        return optimized_models

    @staticmethod
    def fix_matches(orig_locs, matches, new_matches_num=4):
    #     # Create "false matches" in case non are there
    #     for pair_name, pair_matches in matches.values():
    #         if len(pair_matches[0]) < 2:
    #             print("Creating made up matches for pair: {} -> {}".format(os.path.basename(pair_name[0]), os.path.basename(pair_name[1])))
    #             pair_matches[0] = np.zeros((new_matches_num, 2))
    #             pair_matches[1] = np.zeros((new_matches_num, 2))
        # Remove any pair of matched tiles that don't have matches
        to_remove_keys = []
        for pair_name, pair_matches in matches.items():
            if len(pair_matches[0]) == 0:
                print("Removing no matches for pair: {} -> {}".format(os.path.basename(pair_name[0]), os.path.basename(pair_name[1])))
                to_remove_keys.append(pair_name)

        for k in to_remove_keys:
            del matches[k]
            

