import numpy as np
from scipy.optimize import least_squares
import pickle
import os
import time
import scipy.sparse as spp
from scipy.sparse.linalg import lsqr
import scipy.optimize
from rh_renderer.models import RigidModel
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.tools
import pycuda.autoinit
import pycuda.cumath
from pycuda.reduction import ReductionKernel
from pycuda.tools import dtype_to_ctype
from pycuda.compiler import SourceModule
#import common

class GPURigid2DOptimizer(object):
    # TODO - make it a class
    def __init__(self, **kwargs):
        self._damping = float(kwargs.get("damping", 0.0))
        self._huber_delta = kwargs.get("huber_delta", None)
        if self._huber_delta is not None:
            self._huber_delta = float(self._huber_delta)
        self._max_iterations = int(kwargs.get("max_iterations", 1000))
        self._init_gamma = float(kwargs.get("init_gamma", 0.00000000001))
        self._min_gamma = float(kwargs.get("min_gamma", 1e-30))
        self._eps = float(kwargs.get("eps", 1e-9))
        self._pre_translate = "pre_translate" in kwargs
        
        # initialize the access to the gpu, and the needed kernels
        self._init_kernels()

    def _init_kernels(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, 'optimize_rigid_2d_gpu.cu'), 'rt') as in_f:
            optimize_cu = in_f.read()

        mod_optimize_cu = SourceModule(optimize_cu)

        self._compute_cost_func = mod_optimize_cu.get_function("compute_cost")
        self._compute_cost_func.prepare("PPiPPPP")
        self._compute_cost_huber_func = mod_optimize_cu.get_function("compute_cost_huber")
        self._compute_cost_huber_func.prepare("PPiPPPfP")
        self._grad_f_contrib_func = mod_optimize_cu.get_function("grad_f_contrib")
        self._grad_f_contrib_func.prepare("PPiPPPP")
#         self._grad_f_contrib_func.prepare("PPiPPPPPP")
        self._grad_f_contrib_huber_func = mod_optimize_cu.get_function("grad_f_contrib_huber")
        self._grad_f_contrib_huber_func.prepare("PPiPPPfPP")
        self._compute_new_params_func = mod_optimize_cu.get_function("compute_new_params")
        self._compute_new_params_func.prepare("PPiPfP")

        self._reduce_sum_kernel = ReductionKernel(np.float32, "0", "a+b",
                            arguments="const %(tp)s *in" % {"tp": dtype_to_ctype(np.float32)})
#         self._transform_pts_func = mod_optimize_cu.get_function("transform_points")
#         self._transform_pts_func.prepare("PiPPP")

        if self._huber_delta is None:
            self._cost_func = self._compute_cost
            self._grad_func = self._compute_grad_f
        else:
            self._cost_func = self._compute_cost_huber
            self._grad_func = self._compute_grad_f_huber

    @staticmethod
    def _bdim_to_gdim(bdim, cols, rows):
        dx, mx = divmod(cols, bdim[0])
        dy, my = divmod(rows, bdim[1])

        gdim = ( int(dx + int(mx>0)), int(dy + int(my>0)) )
        return gdim


    def _allocate_gpu_vectors(self, p0, tile_names, tile_names_map, matches, matches_num):
        """
        Allocates anbd initializes the arrays on the gpu that will be used for the optimization process
        """
        self._matches_num = matches_num
        self._params_num = p0.shape[0]
        self._tiles_num = p0.shape[0] // 3

        # Allocate the parameters and gradients arrays, and copy the initial parameters
        self._cur_params_gpu = gpuarray.to_gpu(p0.astype(np.float32))
        self._next_params_gpu = gpuarray.empty(p0.shape,
                                               np.float32, order='C')
        self._gradients_gpu = gpuarray.zeros(p0.shape,
                                             np.float32, order='C')
        self._diff_params_gpu = gpuarray.empty(p0.shape,
                                               np.float32, order='C')

        # Allocate and copy matches and indexes mappers - TODO - should be async
        self._src_matches_gpu = cuda.mem_alloc(int(np.dtype(np.float32).itemsize * 2 * matches_num))
        assert(self._src_matches_gpu is not None)
        self._dst_matches_gpu = cuda.mem_alloc(int(np.dtype(np.float32).itemsize * 2 * matches_num))
        assert(self._dst_matches_gpu is not None)
        self._src_idx_to_tile_idx_gpu = cuda.mem_alloc(int(np.dtype(int).itemsize * matches_num))
        assert(self._src_idx_to_tile_idx_gpu is not None)
        self._dst_idx_to_tile_idx_gpu = cuda.mem_alloc(int(np.dtype(int).itemsize * matches_num))
        assert(self._dst_idx_to_tile_idx_gpu is not None)

#         counter = 0
#         for pair_name, pair_matches in matches.items():
#             pair_matches_len = len(pair_matches[0])
#             cuda.py_memcpy_htoa(self._src_matches_gpu, counter, pair_matches[0].astype(np.float32, order='C'))
#             cuda.py_memcpy_htoa(self._dst_matches_gpu, counter, pair_matches[1].astype(np.float32, order='C'))
#             # copy the mapping to tile idx to the gpu TODO - note that the numpy array is reused, so should be careful in async mode
#             tile_idx = np.empty((pair_matches_len, ), dtype=np.int32)
#             tile_idx.fill(tile_names_map[pair_name[0]]) # fill with src tile idx
#             cuda.py_memcpy_htoa(self._src_idx_to_tile_idx_gpu, counter, tile_idx)
#             tile_idx.fill(tile_names_map[pair_name[1]]) # fill with dst tile idx
#             cuda.py_memcpy_htoa(self._dst_idx_to_tile_idx_gpu, counter, tile_idx)
#             counter += pair_matches_len
        counter = 0
        src_matches_all = np.empty((matches_num, 2), dtype=np.float32, order='C')
        dst_matches_all = np.empty((matches_num, 2), dtype=np.float32, order='C')
        src_tiles_idxs_all = np.empty((matches_num,), dtype=np.int32, order='C')
        dst_tiles_idxs_all = np.empty((matches_num,), dtype=np.int32, order='C')
        for pair_name, pair_matches in matches.items():
            pair_matches_len = len(pair_matches[0])
            src_matches_all[counter:counter + pair_matches_len] = pair_matches[0].astype(np.float32)
            dst_matches_all[counter:counter + pair_matches_len] = pair_matches[1].astype(np.float32)
            src_tiles_idxs_all[counter:counter + pair_matches_len] = tile_names_map[pair_name[0]]
            dst_tiles_idxs_all[counter:counter + pair_matches_len] = tile_names_map[pair_name[1]]
            counter += pair_matches_len

        cuda.memcpy_htod(self._src_matches_gpu, src_matches_all)
        cuda.memcpy_htod(self._dst_matches_gpu, dst_matches_all)
        cuda.memcpy_htod(self._src_idx_to_tile_idx_gpu, src_tiles_idxs_all)
        cuda.memcpy_htod(self._dst_idx_to_tile_idx_gpu, dst_tiles_idxs_all)

        # Allocate memory for the residuals
        self._residuals_gpu = gpuarray.empty((matches_num, ),
                                             np.float32, order='C')


#         self._temp_pts_gpu = gpuarray.empty((matches_num, 2),
#                                             np.float32, order='C')
#         self._temp_src_grad_f_contrib_gpu = gpuarray.empty((matches_num,),
#                                             np.float32, order='C')
#         self._temp_dst_grad_f_contrib_gpu = gpuarray.empty((matches_num,),
#                                             np.float32, order='C')


    def _deallocate_gpu_vectors(self):
        del self._cur_params_gpu
        del self._next_params_gpu
        del self._gradients_gpu
        del self._diff_params_gpu
        self._src_matches_gpu.free()
        self._dst_matches_gpu.free()
        self._src_idx_to_tile_idx_gpu.free()
        self._dst_idx_to_tile_idx_gpu.free()
        del self._residuals_gpu


    @staticmethod
    def apply_rigid_transform(pts, theta, t_x, t_y):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return np.dot([[cos_theta, -sin_theta],
                   [sin_theta, cos_theta]],
                   pts.T).T + np.array([t_x, t_y])

    @staticmethod
    def compute_all_dists(matches, transforms, matches_num):
        dists = np.empty((matches_num, ), dtype=np.float32)
        start_idx = 0
        for pair_name, pair_matches in matches.items():
            pair_matches_len = len(pair_matches[0])

            transform1 = transforms[pair_name[0]]
            transform2 = transforms[pair_name[1]]
            pts1_transformed = GPURigid2DOptimizer.apply_rigid_transform(pair_matches[0], *transform1)
            pts2_transformed = GPURigid2DOptimizer.apply_rigid_transform(pair_matches[1], *transform2)

            # compute the L2 distance between the two sets of points
            deltas = pts1_transformed - pts2_transformed
            dists[start_idx:start_idx + pair_matches_len] = np.sqrt(np.sum(deltas**2, axis=1))
            start_idx += pair_matches_len
        return dists

    def _compute_cost_huber(self, params_gpu):
        bdim = (128, 1, 1)
        gdim = GPURigid2DOptimizer._bdim_to_gdim(bdim, self._matches_num, 1)
        self._compute_cost_huber_func.prepared_call(gdim, bdim,
                          self._src_matches_gpu, self._dst_matches_gpu, self._matches_num,
                          params_gpu.gpudata, self._src_idx_to_tile_idx_gpu, self._dst_idx_to_tile_idx_gpu,
                          self._huber_delta,
                          self._residuals_gpu.gpudata)
 
        # Memoization seems to cause a hang when using multiprocessing and pycuda.gpuarray.sum
#         cost_arr = pycuda.gpuarray.sum(self._residuals_gpu)
#         cost = float(cost_arr.get())
#         del cost_arr
        cost_arr = self._reduce_sum_kernel(self._residuals_gpu)
        cost = float(cost_arr.get())
        del cost_arr
        return cost


    def _compute_cost(self, params_gpu):
        bdim = (128, 1, 1)
        gdim = GPURigid2DOptimizer._bdim_to_gdim(bdim, self._matches_num, 1)
        self._compute_cost_func.prepared_call(gdim, bdim,
                          self._src_matches_gpu, self._dst_matches_gpu, self._matches_num,
                          params_gpu.gpudata, self._src_idx_to_tile_idx_gpu, self._dst_idx_to_tile_idx_gpu,
                          self._residuals_gpu.gpudata)
 
        # Memoization seems to cause a hang when using multiprocessing and pycuda.gpuarray.sum
#         cost_arr = pycuda.gpuarray.sum(self._residuals_gpu)
#         cost = float(cost_arr.get())
#         del cost_arr
        cost_arr = self._reduce_sum_kernel(self._residuals_gpu)
        cost = float(cost_arr.get())
        del cost_arr
        return cost

#     def _transform_points(self, pts_gpu, map_gpu):
#         bdim = (128, 1, 1)
#         gdim = GPURigid2DOptimizer._bdim_to_gdim(bdim, self._matches_num, 1)
#         self._transform_pts_func(pts_gpu, self._matches_num,
#                                 self._cur_params_gpu.gpudata, map_gpu,
#                                 self._temp_pts_gpu,
#                                 block=bdim, grid=gdim)
#         pts = self._temp_pts_gpu.get()
#         return pts


    def _compute_grad_f_huber(self):
        bdim = (128, 1, 1)
        gdim = GPURigid2DOptimizer._bdim_to_gdim(bdim, self._matches_num, 1)
        self._grad_f_contrib_huber_func.prepared_call(gdim, bdim,
                          self._src_matches_gpu, self._dst_matches_gpu, self._matches_num,
                          self._cur_params_gpu.gpudata, self._src_idx_to_tile_idx_gpu, self._dst_idx_to_tile_idx_gpu,
                          self._huber_delta, self._residuals_gpu.gpudata,
                          self._gradients_gpu.gpudata)


    def _compute_grad_f(self):
        bdim = (128, 1, 1)
        gdim = GPURigid2DOptimizer._bdim_to_gdim(bdim, self._matches_num, 1)
        self._grad_f_contrib_func.prepared_call(gdim, bdim,
                          self._src_matches_gpu, self._dst_matches_gpu, self._matches_num,
                          self._cur_params_gpu.gpudata, self._src_idx_to_tile_idx_gpu, self._dst_idx_to_tile_idx_gpu,
                          self._gradients_gpu.gpudata)
#                           self._temp_src_grad_f_contrib_gpu.gpudata, self._temp_dst_grad_f_contrib_gpu.gpudata)


    def _compute_new_params(self, gamma):
        bdim = (128, 1, 1)
        gdim = GPURigid2DOptimizer._bdim_to_gdim(bdim, self._tiles_num, 1)
        self._compute_new_params_func.prepared_call(gdim, bdim,
                                      self._cur_params_gpu.gpudata, self._next_params_gpu.gpudata, np.int32(self._tiles_num),
                                      self._gradients_gpu.gpudata, np.float32(gamma), self._diff_params_gpu.gpudata)


    def _gradient_descent(self):

        # compute the cost
        cur_cost = self._cost_func(self._cur_params_gpu)
        print("Initial cost: {}".format(cur_cost))
        # cur_residuals_cpu = optimize_fun(self._cur_params_gpu.get(), self._tile_names_map, self._matches, self._matches_num)
        # print("Initial cost-cpu: {}".format(np.sum(cur_residuals_cpu)))
        gamma = self._init_gamma

        for it in range(self._max_iterations):
            print("Iteration {}".format(it))
            #prev_p = cur_p
            prev_cost = cur_cost
            #cur_p = prev_p - gamma * grad_F_huber(huber_delta, prev_p, *args)
            self._grad_func()
#             grad_cpu, per_match_src_contrib, per_match_dst_contrib = grad_F_huber(500000, self._cur_params_gpu.get(), self._tile_names_map, self._matches, self._matches_num)
#             grad_gpu = self._gradients_gpu.get()
#             pts1_transformed_cpu, pts2_transformed_cpu = compute_all_pts_transformed(self._matches_num, self._matches, self._cur_params_gpu.get(), self._tile_names_map)
#
#             pts1_transformed_gpu = self._transform_points(self._src_matches_gpu, self._src_idx_to_tile_idx_gpu)
#             pts2_transformed_gpu = self._transform_points(self._dst_matches_gpu, self._dst_idx_to_tile_idx_gpu)
#             die
            self._compute_new_params(gamma)
            #print("New params: {}".format(cur_p))
            #cur_cost = np.sum(optimize_fun(cur_p, *args))
            #cur_cost = compute_cost_huber(optimize_fun, cur_p, args, huber_delta)
            cur_cost = self._cost_func(self._next_params_gpu)
            print("New cost: {}".format(cur_cost))
            if cur_cost > prev_cost: # we took a bad step: undo it, scale down gamma, and start over
                print("Backtracking step")
                #cur_p = prev_p
                cur_cost = prev_cost
                gamma *= 0.5
            #elif float(pycuda.gpuarray.max(pycuda.cumath.fabs(self._diff_params_gpu)).get()) <= self._eps:
            elif float(pycuda.gpuarray.max(self._diff_params_gpu).get()) <= self._eps:
                # We took a good step, but the change to the parameters vector is negligible
                temp = self._cur_params_gpu
                self._cur_params_gpu = self._next_params_gpu
                self._next_params_gpu = temp
                break
            else: # We took a good step, try to increase the step size a bit
                gamma *= 1.1
                # change between cur_params_gpu and next_params_gpu so next iteartion cur_params will be the next_params
                temp = self._cur_params_gpu
                self._cur_params_gpu = self._next_params_gpu
                self._next_params_gpu = temp
            if gamma < self._min_gamma:
                break

        #print("The local minimum occurs at", cur_p)
        cur_p = self._cur_params_gpu.get()
        return cur_p




    def optimize(self, orig_locs, matches, pre_translate=False):
        """
        The aim is to find for each tile a triplet: tetha, t_x, and t_y that will define the
        rigid transformation that needs to be applied to that tile.
        The transformation needs to minimize the L2 distance between the matches of pairs of tiles.
        To this end, we define our optimizations as a non-linear least squares problem.
        Given that the number of tiles is N, and the total number of matches is M,
        we want to find the values for 3*N parameters, s.t., the sum of all distances is minimized.
        Note that due to outliers, we would like to use a more robust method, such as soft_L1.

        """

        tile_names = sorted(list(orig_locs.keys()))
        tile_names_map = {name:idx for idx, name in enumerate(tile_names)}
        matches_num = np.sum([len(m[0]) for m in matches.values()])
        p0 = np.empty((len(orig_locs)*3, ), dtype=np.float32) # all triplets [theta1, t_x1, t_y1, theta2, t_x2, t_y2, ...]

        # FOR DEBUG:
        #self._matches = matches
        #self._tile_names_map = tile_names_map

        if self._pre_translate:

            # For debug:
            solution1 = {name:[0, orig_locs[name][0], orig_locs[name][1]] for name, idx in tile_names_map.items()}
            dists = GPURigid2DOptimizer.compute_all_dists(matches, solution1, matches_num)
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
            dists = GPURigid2DOptimizer.compute_all_dists(matches, solution2, matches_num)
            print("post translation optimization distances: min={}, mean={}, median={}, max={}".format(np.min(dists), np.mean(dists), np.median(dists), np.max(dists)))
        else:
            p0[1::3] = [orig_locs[k][0] for k in tile_names] # set default X to original location's X
            p0[2::3] = [orig_locs[k][1] for k in tile_names] # set default Y to original location's Y
            

        p0[::3] = 0 # Set default theta to 0


        # Create a sparse matrix that has
        st_time = time.time()

        # allocate gpu memory
        self._allocate_gpu_vectors(p0, tile_names, tile_names_map, matches, matches_num)


#         solution_init = {}
#         res_init = self._cur_params_gpu.get()
#         for name, idx in tile_names_map.items():
#             solution_init[name] = np.array(res_init[idx * 3:idx*3 + 3]) # Stores [theta, t_x, t_y] of the tile
#         dists = GPURigid2DOptimizer.compute_all_dists(matches, solution_init, matches_num)
#         print("test-gpu distances: min={}, mean={}, median={}, max={}".format(np.min(dists), np.mean(dists), np.median(dists), np.max(dists)))

        #res = least_squares(optimize_fun, p0, args=(tile_names_map, matches, matches_num), verbose=2)
        #res = least_squares(optimize_fun, p0, loss='huber', f_scale=15, args=(tile_names_map, matches, matches_num), verbose=2)
        #res = least_squares(optimize_fun, p0, loss='soft_l1', f_scale=15, args=(tile_names_map, matches, matches_num), verbose=2)
    #     stepsize = 0.0001
    #     max_iterations = 1000
    #     res = gradient_descent(optimize_fun, p0, max_iterations, stepsize, args=(tile_names_map, matches, matches_num))

        huber_delta = 15 # Maximal L2 distance for a match to be considered inlier
        res = self._gradient_descent()
        #res = gradient_descent(optimize_fun, p0, grad_F_huber, huber_delta, args=(tile_names_map, matches, matches_num))

        self._deallocate_gpu_vectors()

        end_time = time.time()
        print("non-linear optimization time: {} seconds".format(end_time - st_time))

        solution = {}
        if res is not None:
            for name, idx in tile_names_map.items():
                solution[name] = np.array(res[idx * 3:idx*3 + 3]) # Stores [theta, t_x, t_y] of the tile
        else:
            raise Exception("Could not find a valid solution to the optimization problem")

        dists = GPURigid2DOptimizer.compute_all_dists(matches, solution, matches_num)
        print("post optimization distances: min={}, mean={}, median={}, max={}".format(np.min(dists), np.mean(dists), np.median(dists), np.max(dists)))

        # create the optimized models for each tile
        optimized_models = {name:RigidModel(res[idx*3], res[idx*3+1:idx*3+3]) for name, idx in tile_names_map.items()}
        return optimized_models






