from rh_logger.api import logger
import logging
import numpy as np
from scipy.optimize import least_squares
import pickle
import os
import time
import scipy.sparse as spp
from scipy.sparse.linalg import lsqr
import scipy.optimize
from rh_renderer.models import RigidModel
#import common

EPS = 0.000001

class Rigid2DOptimizerFaster(object):
    # TODO - make it a class
    def __init__(self, **kwargs):
        self._damping = float(kwargs.get("damping", 0.0))
        self._huber_delta = float(kwargs.get("huber_delta", 15))
        self._max_iterations = int(kwargs.get("max_iterations", 1000))
        self._init_gamma = float(kwargs.get("init_gamma", 0.00000000001))
        self._min_gamma = float(kwargs.get("min_gamma", 1e-30))
        self._eps = float(kwargs.get("eps", 1e-9))
        self._pre_translate = "pre_translate" in kwargs
        

    def _allocate_vectors(self, p0, tile_names, tile_names_map, matches, matches_num):
        """
        Allocates anbd initializes the arrays on the gpu that will be used for the optimization process
        """
        self._matches_num = matches_num
        self._params_num = p0.shape[0]
        self._tiles_num = p0.shape[0] // 3


        self._cur_params_vector = np.copy(p0)
        self._next_params_vector = np.empty_like(p0)
        self._gradients_vector = np.zeros_like(p0)
        self._diff_params_vector = np.empty_like(p0)

        counter = 0
        self._src_matches_all = np.empty((matches_num, 2), dtype=np.float32, order='C')
        self._dst_matches_all = np.empty((matches_num, 2), dtype=np.float32, order='C')
        self._src_tiles_idxs_all = np.empty((matches_num,), dtype=np.int32, order='C')
        self._dst_tiles_idxs_all = np.empty((matches_num,), dtype=np.int32, order='C')
        for pair_name, pair_matches in matches.items():
            pair_matches_len = len(pair_matches[0])
            self._src_matches_all[counter:counter + pair_matches_len] = pair_matches[0].astype(np.float32)
            self._dst_matches_all[counter:counter + pair_matches_len] = pair_matches[1].astype(np.float32)
            self._src_tiles_idxs_all[counter:counter + pair_matches_len] = tile_names_map[pair_name[0]]
            self._dst_tiles_idxs_all[counter:counter + pair_matches_len] = tile_names_map[pair_name[1]]
            counter += pair_matches_len

        self._residuals_vector = np.empty((matches_num, ), dtype=np.float32)


    @staticmethod
    def apply_rigid_transform(pts, theta, t_x, t_y):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return np.dot([[cos_theta, -sin_theta],
                   [sin_theta, cos_theta]],
                   pts.T).T + np.array([t_x, t_y])

    @staticmethod
    def transform_pts(params, tiles_idxs_all, pts):

        # create vectors of all the rigid transformations (r00 will be element 00 of the matrices, 01 will be element 01 of the matrices, etc...)
        cos_theta = np.cos(params[tiles_idxs_all*3 + 0])
        sin_theta = np.sin(params[tiles_idxs_all*3 + 0])
        pts_transformed = np.array([
                                cos_theta * pts[:, 0] - sin_theta * pts[:, 1] + params[tiles_idxs_all * 3 + 1],
                                sin_theta * pts[:, 0] + cos_theta * pts[:, 1] + params[tiles_idxs_all * 3 + 2]
                                  ]).T
        return pts_transformed



    def _optimize_func(self, params):
        pts1_transformed = Rigid2DOptimizerFaster.transform_pts(params, self._src_tiles_idxs_all, self._src_matches_all)
        pts2_transformed = Rigid2DOptimizerFaster.transform_pts(params, self._dst_tiles_idxs_all, self._dst_matches_all)

        deltas = pts1_transformed - pts2_transformed

        #np.sqrt(np.sum(deltas**2, axis=1), out=self._residuals)
        np.sqrt(np.sum(deltas**2, axis=1), self._residuals_vector)

        return self._residuals_vector

    def _compute_cost(self, cur_p):
        cost = np.sum(self._optimize_func(cur_p))
        return cost
        
    def _compute_cost_huber(self, cur_p, huber_delta):
        residuals = self._optimize_func(cur_p)
        cost = np.empty_like(residuals)
        residuals_huber_mask = residuals <= huber_delta
        cost[residuals_huber_mask] = 0.5 * residuals[residuals_huber_mask]**2
        cost[~residuals_huber_mask] = huber_delta * residuals[~residuals_huber_mask] - (0.5 * huber_delta**2)
        return np.sum(cost)



    @staticmethod
    def compute_all_dists(matches, transforms, matches_num):
        dists = np.empty((matches_num, ), dtype=np.float32)
        start_idx = 0
        for pair_name, pair_matches in matches.items():
            pair_matches_len = len(pair_matches[0])

            transform1 = transforms[pair_name[0]]
            transform2 = transforms[pair_name[1]]
            pts1_transformed = Rigid2DOptimizerFaster.apply_rigid_transform(pair_matches[0], *transform1)
            pts2_transformed = Rigid2DOptimizerFaster.apply_rigid_transform(pair_matches[1], *transform2)

            # compute the L2 distance between the two sets of points
            deltas = pts1_transformed - pts2_transformed
            dists[start_idx:start_idx + pair_matches_len] = np.sqrt(np.sum(deltas**2, axis=1))
            start_idx += pair_matches_len
        return dists
     

    def _grad_F_huber(self, params, huber_delta):
                
        #print("self._src_tiles_idxs_all.shape", self._src_tiles_idxs_all.shape)
        #print("self._src_matches_all.shape", self._src_matches_all.shape)

        pts1_transformed = Rigid2DOptimizerFaster.transform_pts(params, self._src_tiles_idxs_all, self._src_matches_all)
        pts2_transformed = Rigid2DOptimizerFaster.transform_pts(params, self._dst_tiles_idxs_all, self._dst_matches_all)

        #print("pts1_transformed.shape", pts1_transformed.shape)
        #print("pts2_transformed.shape", pts2_transformed.shape)
        deltas = pts1_transformed - pts2_transformed
        delta_x = deltas[:, 0]
        delta_y = deltas[:, 1]

        #np.sqrt(np.sum(deltas**2, axis=1), out=self._residuals)
        np.sqrt(np.sum(deltas**2, axis=1), out=self._residuals_vector)

        residuals_huber_mask = self._residuals_vector <= huber_delta

        # The gradient coefficient for anything that is below the huber delta is 1, and anything above should be:
        # (delta / R), where R is the distance between the two points
        grad_f_multiplier = np.ones_like(self._residuals_vector)
        grad_f_multiplier[~residuals_huber_mask] = huber_delta / self._residuals_vector[~residuals_huber_mask]

        # update the grad_f per each match (and later sum them per tile)
        theta1_idxs = self._src_tiles_idxs_all * 3
        #print("delta_x.shape", delta_x.shape)
        #print("self._src_matches_all[:, 0].shape", self._src_matches_all[:, 0].shape)
        #print("params[theta1_idxs].shape", params[theta1_idxs].shape)
        #print("self._src_matches_all[:, 1].shape", self._src_matches_all[:, 1].shape)
        #print("grad_f_multiplier.shape", grad_f_multiplier.shape)
        per_match_grad_f_theta1 = delta_x * (-self._src_matches_all[:, 0]*params[theta1_idxs] - self._src_matches_all[:, 1]) * grad_f_multiplier +\
                                  delta_y * (self._src_matches_all[:, 0] - self._src_matches_all[:, 1]*params[theta1_idxs]) * grad_f_multiplier
        per_match_grad_f_tx1 = delta_x * grad_f_multiplier
        per_match_grad_f_ty1 = delta_y * grad_f_multiplier

        theta2_idxs = self._dst_tiles_idxs_all * 3
        per_match_grad_f_theta2 = delta_x * (self._dst_matches_all[:, 0]*params[theta2_idxs] + self._dst_matches_all[:, 1]) * grad_f_multiplier +\
                                  delta_y * (-self._dst_matches_all[:, 0] + self._dst_matches_all[:, 1]*params[theta2_idxs]) * grad_f_multiplier
        per_match_grad_f_tx2 = -(delta_x * grad_f_multiplier)
        per_match_grad_f_ty2 = -(delta_y * grad_f_multiplier)

        # merge all updates for each param
        grad_f_result = np.zeros_like(params)
        np.add.at(grad_f_result, theta1_idxs, per_match_grad_f_theta1)
        np.add.at(grad_f_result, theta1_idxs + 1, per_match_grad_f_tx1)
        np.add.at(grad_f_result, theta1_idxs + 2, per_match_grad_f_ty1)
        np.add.at(grad_f_result, theta2_idxs, per_match_grad_f_theta2)
        np.add.at(grad_f_result, theta2_idxs + 1, per_match_grad_f_tx2)
        np.add.at(grad_f_result, theta2_idxs + 2, per_match_grad_f_ty2)

        return grad_f_result


    def _gradient_descent(self):
        
        cur_p = self._cur_params_vector
        #cur_cost = np.sum(optimize_func(cur_p, *args))
        cur_cost = self._compute_cost_huber(cur_p, self._huber_delta)
        logger.report_event("Initial cost: {}".format(cur_cost), log_level=logging.INFO)
        gamma = self._init_gamma

        for it in range(self._max_iterations):
            #print("Iteration {}".format(it))
            prev_p = cur_p
            prev_cost = cur_cost
            cur_p = prev_p - gamma * self._grad_F_huber(prev_p, self._huber_delta)
            #print("New params: {}".format(cur_p))
            #cur_cost = np.sum(optimize_func(cur_p, *args))
            cur_cost = self._compute_cost_huber(cur_p, self._huber_delta)
            #print("New cost: {}".format(cur_cost))
            if it % 100 == 0:
                logger.report_event("iter {}: C: {}".format(it, cur_cost), log_level=logging.INFO)
            if cur_cost > prev_cost: # we took a bad step: undo it, scale down gamma, and start over
                #print("Backtracking step")
                cur_p = prev_p
                cur_cost = prev_cost
                gamma *= 0.5
            elif np.all(np.abs(cur_p - prev_p) <= self._eps): # We took a good step, but the change to the parameters vector is negligible
                break
            else: # We took a good step, try to increase the step size a bit
                gamma *= 1.1
            if gamma < self._min_gamma:
                break

        #print("The local minimum occurs at", cur_p)
        logger.report_event("Post-opt cost: {}".format(cur_cost), log_level=logging.INFO)
        return cur_p


    def optimize(self, orig_locs, matches):
        """
        The aim is to find for each tile a triplet: tetha, t_x, and t_y that will define the
        rigid transformation that needs to be applied to that tile.
        The transformation needs to minimize the L2 distance between the matches of pairs of tiles.
        To this end, we define our optimizations as a non-linear least squares problem.
        Given that the number of tiles is N, and the total number of matches is M,
        we want to find the values for 3*N parameters, s.t., the sum of all distances is minimized.
        Note that due to outliers, we would like to use a more robust method, such as huber loss.

        """

        tile_names = sorted(list(orig_locs.keys()))
        tile_names_map = {name:idx for idx, name in enumerate(tile_names)}
        matches_num = np.sum([len(m[0]) for m in matches.values()])
        p0 = np.empty((len(orig_locs)*3, ), dtype=np.float32) # all triplets [theta1, t_x1, t_y1, theta2, t_x2, t_y2, ...]

        if self._pre_translate:

            # For debug:
            solution1 = {name:[0, orig_locs[name][0], orig_locs[name][1]] for name, idx in tile_names_map.items()}
            dists = Rigid2DOptimizerFaster.compute_all_dists(matches, solution1, matches_num)
            logger.report_event("pre optimization distances: min={}, mean={}, median={}, max={}".format(np.min(dists), np.mean(dists), np.median(dists), np.max(dists)), log_level=logging.INFO)

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
            logger.report_event("translation-only optimization time: {} seconds".format(time.time() - st_time), log_level=logging.INFO)
            
            # Normalize all deltas to (0, 0)
            Tx -= np.min(Tx)
            Ty -= np.min(Ty)
            p0[1::3] = Tx
            p0[2::3] = Ty

            # For debug:
            #solution2 = {name:[0, p0[1::3][idx], p0[2::3][idx]] for name, idx in tile_names_map.items()}
            solution2 = {name:[0, Tx[idx], Ty[idx]] for name, idx in tile_names_map.items()}
            dists = Rigid2DOptimizerFaster.compute_all_dists(matches, solution2, matches_num)
            logger.report_event("post translation optimization distances: min={}, mean={}, median={}, max={}".format(np.min(dists), np.mean(dists), np.median(dists), np.max(dists)), log_level=logging.INFO)
        else:
            p0[1::3] = [orig_locs[k][0] for k in tile_names] # set default X to original location's X
            p0[2::3] = [orig_locs[k][1] for k in tile_names] # set default Y to original location's Y
            

        p0[::3] = 0 # Set default theta to 0


        # Create a sparse matrix that has 
        st_time = time.time()

        # allocate memory
        self._allocate_vectors(p0, tile_names, tile_names_map, matches, matches_num)

        #res = least_squares(optimize_func, p0, args=(tile_names_map, matches, matches_num), verbose=2)
        #res = least_squares(optimize_func, p0, loss='huber', f_scale=15, args=(tile_names_map, matches, matches_num), verbose=2)
        #res = least_squares(optimize_func, p0, loss='soft_l1', f_scale=15, args=(tile_names_map, matches, matches_num), verbose=2)
    #     stepsize = 0.0001
    #     max_iterations = 1000
    #     res = gradient_descent(optimize_func, p0, max_iterations, stepsize, args=(tile_names_map, matches, matches_num))

        huber_delta = 15 # Maximal L2 distance for a match to be considered inlier
        res = self._gradient_descent()
        #res = self._gradient_descent(Rigid2DOptimizerFaster.optimize_func, p0, Rigid2DOptimizerFaster.grad_F_huber)
        end_time = time.time()

        logger.report_event("non-linear optimization time: {} seconds".format(end_time - st_time), log_level=logging.INFO)

        solution = {}
        if res is not None:
            for name, idx in tile_names_map.items():
                solution[name] = np.array(res[idx * 3:idx*3 + 3]) # Stores [theta, t_x, t_y] of the tile
        else:
            raise Exception("Could not find a valid solution to the optimization problem")

        dists = Rigid2DOptimizerFaster.compute_all_dists(matches, solution, matches_num)
        logger.report_event("post optimization distances: min={}, mean={}, median={}, max={}".format(np.min(dists), np.mean(dists), np.median(dists), np.max(dists)), log_level=logging.INFO)

        # create the optimized models for each tile
        optimized_models = {name:RigidModel(res[idx*3], res[idx*3+1:idx*3+3]) for name, idx in tile_names_map.items()}
        return optimized_models


#     def fix_matches(orig_locs, matches, new_matches_num=4):
#     #     # Create "false matches" in case non are there
#     #     for pair_name, pair_matches in matches.values():
#     #         if len(pair_matches[0]) < 2:
#     #             print("Creating made up matches for pair: {} -> {}".format(os.path.basename(pair_name[0]), os.path.basename(pair_name[1])))
#     #             pair_matches[0] = np.zeros((new_matches_num, 2))
#     #             pair_matches[1] = np.zeros((new_matches_num, 2))
#         # Remove any pair of matched tiles that don't have matches
#         to_remove_keys = []
#         for pair_name, pair_matches in matches.items():
#             if len(pair_matches[0]) == 0:
#                 print("Removing no matches for pair: {} -> {}".format(os.path.basename(pair_name[0]), os.path.basename(pair_name[1])))
#                 to_remove_keys.append(pair_name)
# 
#         for k in to_remove_keys:
#             del matches[k]
            

if __name__ == '__main__':
#     in_orig_locs_fname = 'data/W05_Sec001_ROI466_mfovs_475_476_orig_locs.pkl'
#     in_matches_fname = 'data/W05_Sec001_ROI466_mfovs_475_476.pkl'
#     in_ts_fname = 'data/W05_Sec001_ROI466_mfovs_475_476.json'
#     out_ts_fname = 'montaged_optimize3_W05_Sec001_ROI466_mfovs_475_476.json'
    in_orig_locs_fname = 'data/W05_Sec001_ROI466_orig_locs.pkl'
    in_matches_fname = 'data/W05_Sec001_ROI466.pkl'
    in_ts_fname = 'data/W05_Sec001_ROI466.json'
    out_ts_fname = 'montaged_optimize3_W05_Sec001_ROI466.json'

    # Read the files
    with open(in_orig_locs_fname, 'rb') as in_f:
        orig_locs = pickle.load(in_f)
    with open(in_matches_fname, 'rb') as in_f:
        matches = pickle.load(in_f)

    fix_matches(orig_locs, matches)
    solution = optimize(orig_locs, matches, pre_translate=True)
    #common.export_rigid_tilespec(in_ts_fname, out_ts_fname, solution)

