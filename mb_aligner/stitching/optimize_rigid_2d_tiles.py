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

class Rigid2DOptimizer(object):
    # TODO - make it a class
    def __init__(self, **kwargs):
        self._damping = float(kwargs.get("damping", 0.0))
        self._huber_delta = float(kwargs.get("huber_delta", 15))
        self._max_iterations = int(kwargs.get("max_iterations", 1000))
        self._init_gamma = float(kwargs.get("init_gamma", 0.00000000001))
        self._min_gamma = float(kwargs.get("min_gamma", 1e-30))
        self._eps = float(kwargs.get("eps", 1e-9))
        self._pre_translate = "pre_translate" in kwargs
        

    @staticmethod
    def apply_rigid_transform(pts, theta, t_x, t_y):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return np.dot([[cos_theta, -sin_theta],
                   [sin_theta, cos_theta]],
                   pts.T).T + np.array([t_x, t_y])

    @staticmethod
    def optimize_func(params, tile_names_map, matches, matches_num):
        # Compute the residuals of all the matches
        residuals = np.empty((matches_num, ), dtype=np.float32)
        start_idx = 0
        for pair_name, pair_matches in matches.items():
            pair_matches_len = len(pair_matches[0])

            tile1_params_start_idx = tile_names_map[pair_name[0]] * 3
            tile2_params_start_idx = tile_names_map[pair_name[1]] * 3
            pts1_transformed = Rigid2DOptimizer.apply_rigid_transform(pair_matches[0], *params[tile1_params_start_idx:tile1_params_start_idx+3])
            pts2_transformed = Rigid2DOptimizer.apply_rigid_transform(pair_matches[1], *params[tile2_params_start_idx:tile2_params_start_idx+3])

            # compute the L2 distance between the two sets of points
            deltas = pts1_transformed - pts2_transformed
            residuals[start_idx:start_idx + pair_matches_len] = np.sqrt(np.sum(deltas**2, axis=1))
            start_idx += pair_matches_len

        # Normalize the residuals by 2*median
        #med_residual = np.median(residuals)
        #residuals = residuals / (2*med_residual + EPS)

        return residuals
        

    @staticmethod
    def compute_all_dists(matches, transforms, matches_num):
        dists = np.empty((matches_num, ), dtype=np.float32)
        start_idx = 0
        for pair_name, pair_matches in matches.items():
            pair_matches_len = len(pair_matches[0])

            transform1 = transforms[pair_name[0]]
            transform2 = transforms[pair_name[1]]
            pts1_transformed = Rigid2DOptimizer.apply_rigid_transform(pair_matches[0], *transform1)
            pts2_transformed = Rigid2DOptimizer.apply_rigid_transform(pair_matches[1], *transform2)

            # compute the L2 distance between the two sets of points
            deltas = pts1_transformed - pts2_transformed
            dists[start_idx:start_idx + pair_matches_len] = np.sqrt(np.sum(deltas**2, axis=1))
            start_idx += pair_matches_len
        return dists
     

    @staticmethod
    def grad_F_huber(huber_delta, params, tile_names_map, matches, matches_num):
        
        # Compute the residuals of all the matches
        grad_f_result = np.zeros_like(params)
        #start_idx = 0
        for pair_name, pair_matches in matches.items():
            #pair_matches_len = len(pair_matches[0])


            tile1_params_start_idx = tile_names_map[pair_name[0]] * 3
            tile2_params_start_idx = tile_names_map[pair_name[1]] * 3

            pts1_transformed = Rigid2DOptimizer.apply_rigid_transform(pair_matches[0], *params[tile1_params_start_idx:tile1_params_start_idx+3])
            pts2_transformed = Rigid2DOptimizer.apply_rigid_transform(pair_matches[1], *params[tile2_params_start_idx:tile2_params_start_idx+3])
            deltas = pts1_transformed - pts2_transformed
            delta_x = deltas[:, 0]
            delta_y = deltas[:, 1]

            residuals = np.sqrt(np.sum(deltas**2, axis=1))
            residuals_huber_mask = residuals <= huber_delta
            # The gradient coefficient for anything that is below the huber delta is 1, and anything above should be:
            # (delta / R), where R is the distance between the two points
            grad_f_multiplier = np.ones_like(residuals)
            grad_f_multiplier[~residuals_huber_mask] = huber_delta / residuals[~residuals_huber_mask]
     
            # The current matches only add values to the gradient at the indices of the relevant parameters (don't change anything else)
            theta1_idx = tile1_params_start_idx
            tx1_idx = tile1_params_start_idx + 1
            ty1_idx = tile1_params_start_idx + 2
            theta2_idx = tile2_params_start_idx
            tx2_idx = tile2_params_start_idx + 1
            ty2_idx = tile2_params_start_idx + 2
            # Update grad(Theta_tile1) , grad(Tx_tile1), grad(Ty_tile1)
            grad_f_result[theta1_idx] += np.sum(
                                            np.dot(delta_x * (-pair_matches[0][:, 0]*params[theta1_idx] - pair_matches[0][:, 1]),
                                                   grad_f_multiplier)
                                         ) + np.sum(
                                            np.dot(delta_y * (pair_matches[0][:, 0] - pair_matches[0][:, 1]*params[theta1_idx]),
                                                   grad_f_multiplier)
                                         )
            grad_f_result[tx1_idx] += np.sum(np.dot(delta_x, grad_f_multiplier))
            grad_f_result[ty1_idx] += np.sum(np.dot(delta_y, grad_f_multiplier))
            # Update grad(Theta_tile2) , grad(Tx_tile2), grad(Ty_tile2)
            grad_f_result[theta2_idx] += np.sum(
                                            np.dot(delta_x * (pair_matches[1][:, 0]*params[theta2_idx] + pair_matches[1][:, 1]),
                                                   grad_f_multiplier)
                                         ) + np.sum(
                                            np.dot(delta_y * (-pair_matches[1][:, 0] + pair_matches[1][:, 1]*params[theta2_idx]),
                                                   grad_f_multiplier)
                                         )
            grad_f_result[tx2_idx] += -np.sum(np.dot(delta_x, grad_f_multiplier))
            grad_f_result[ty2_idx] += -np.sum(np.dot(delta_y, grad_f_multiplier))


        return grad_f_result


    def _gradient_descent(self, optimize_func, p0, grad_F_huber, args=None):
        
        def compute_cost_huber(optimize_func, cur_p, params, huber_delta):
            residuals = optimize_func(cur_p, *params)
            cost = np.empty_like(residuals)
            residuals_huber_mask = residuals <= huber_delta
            cost[residuals_huber_mask] = 0.5 * residuals[residuals_huber_mask]**2
            cost[~residuals_huber_mask] = huber_delta * residuals[~residuals_huber_mask] - (0.5 * huber_delta**2)
            return np.sum(cost)

        cur_p = p0
        #cur_cost = np.sum(optimize_func(cur_p, *args))
        cur_cost = compute_cost_huber(optimize_func, cur_p, args, self._huber_delta)
        print("Initial cost: {}".format(cur_cost))
        gamma = self._init_gamma

        for it in range(self._max_iterations):
            print("Iteration {}".format(it))
            prev_p = cur_p
            prev_cost = cur_cost
            cur_p = prev_p - gamma * grad_F_huber(self._huber_delta, prev_p, *args)
            #print("New params: {}".format(cur_p))
            #cur_cost = np.sum(optimize_func(cur_p, *args))
            cur_cost = compute_cost_huber(optimize_func, cur_p, args, self._huber_delta)
            print("New cost: {}".format(cur_cost))
            if cur_cost > prev_cost: # we took a bad step: undo it, scale down gamma, and start over
                print("Backtracking step")
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
            dists = Rigid2DOptimizer.compute_all_dists(matches, solution1, matches_num)
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
            p0[1::3] = Tx
            p0[2::3] = Ty

            # For debug:
            #solution2 = {name:[0, p0[1::3][idx], p0[2::3][idx]] for name, idx in tile_names_map.items()}
            solution2 = {name:[0, Tx[idx], Ty[idx]] for name, idx in tile_names_map.items()}
            dists = Rigid2DOptimizer.compute_all_dists(matches, solution2, matches_num)
            print("post translation optimization distances: min={}, mean={}, median={}, max={}".format(np.min(dists), np.mean(dists), np.median(dists), np.max(dists)))
        else:
            p0[1::3] = [orig_locs[k][0] for k in tile_names] # set default X to original location's X
            p0[2::3] = [orig_locs[k][1] for k in tile_names] # set default Y to original location's Y
            

        p0[::3] = 0 # Set default theta to 0


        # Create a sparse matrix that has 
        st_time = time.time()
        #res = least_squares(optimize_func, p0, args=(tile_names_map, matches, matches_num), verbose=2)
        #res = least_squares(optimize_func, p0, loss='huber', f_scale=15, args=(tile_names_map, matches, matches_num), verbose=2)
        #res = least_squares(optimize_func, p0, loss='soft_l1', f_scale=15, args=(tile_names_map, matches, matches_num), verbose=2)
    #     stepsize = 0.0001
    #     max_iterations = 1000
    #     res = gradient_descent(optimize_func, p0, max_iterations, stepsize, args=(tile_names_map, matches, matches_num))

        huber_delta = 15 # Maximal L2 distance for a match to be considered inlier
        res = self._gradient_descent(Rigid2DOptimizer.optimize_func, p0, Rigid2DOptimizer.grad_F_huber, args=(tile_names_map, matches, matches_num))
        end_time = time.time()

        print("non-linear optimization time: {} seconds".format(end_time - st_time))

        solution = {}
        if res is not None:
            for name, idx in tile_names_map.items():
                solution[name] = np.array(res[idx * 3:idx*3 + 3]) # Stores [theta, t_x, t_y] of the tile
        else:
            raise Exception("Could not find a valid solution to the optimization problem")

        dists = Rigid2DOptimizer.compute_all_dists(matches, solution, matches_num)
        print("post optimization distances: min={}, mean={}, median={}, max={}".format(np.min(dists), np.mean(dists), np.median(dists), np.max(dists)))

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

