from rh_logger.api import logger
import logging
import sys
import json
import os
import os.path
import numpy as np
import pickle
from scipy.spatial import Delaunay
from sklearn.utils.extmath import randomized_svd
#import matplotlib
#matplotlib.use('GTK')
#import pylab
#from matplotlib import collections as mc
import gc
from mb_aligner.common import utils
import datetime
from collections import defaultdict
import glob
import re
#from rh_aligner.dal.serializer import get_serializer_from_conf
#from rh_aligner.common.parallel_reader import ParallelProcessesFilesReader
#from rh_aligner.common import ransac
from rh_renderer import models
from mb_aligner.common.tps.tps import ThinPlateSplines

import pyximport
pyximport.install()
import mb_aligner.alignment.optimizers.mesh_derivs_elastic as mesh_derivs_elastic

FLOAT_TYPE = np.float64

SHOW_FINAL_MO = True
SHOW_BLOCK_FINAL_MO = True
MAX_SAMPLED_MATCHES = 5000

sys.setrecursionlimit(10000)  # for grad

class Mesh(object):
    def __init__(self, points):
        # load the mesh
        #self.orig_pts = np.array(points, dtype=FLOAT_TYPE).reshape((-1, 2)).copy()
        self.pts = np.array(points, dtype=FLOAT_TYPE).reshape((-1, 2)).copy()
#        center = self.pts.mean(axis=0)
#        self.pts -= center
#        self.pts *= 1.1
#        self.pts += center
        self.orig_pts = self.pts.copy()

        logger.report_event("# points in base mesh {}".format(self.pts.shape[0]), log_level=logging.DEBUG)

        # for neighbor searching and internal mesh
        self.triangulation = Delaunay(self.pts)

    def internal_structural_mesh(self):
        simplices = self.triangulation.simplices.astype(np.uint32)
        # find unique edges
        edge_indices = np.vstack((simplices[:, :2],
                                  simplices[:, 1:],
                                  simplices[:, [0, 2]]))
        edge_indices = np.vstack({tuple(sorted(tuple(row))) for row in edge_indices}).astype(np.uint32)
        # mesh.pts[edge_indices, :].shape =(#edges, #pts-per-edge, #values-per-pt)
        edge_lengths = np.sqrt((np.diff(self.pts[edge_indices], axis=1) ** 2).sum(axis=2)).ravel()
        #print("Median edge length:", np.median(edge_lengths), "Max edge length:", np.max(edge_lengths))
        triangles_as_pts = self.pts[simplices]
        triangle_areas = 0.5 * np.cross(triangles_as_pts[:, 2, :] - triangles_as_pts[:, 0, :],
                                        triangles_as_pts[:, 1, :] - triangles_as_pts[:, 0, :])
        return edge_indices, edge_lengths, simplices, triangle_areas

    def remove_unneeded_points(self, pts1, pts2):
        """Removes points that cause query_cross_barycentrics to fail"""
        p1 = pts1.copy()
        p1[p1 < 0] = 0.01
        simplex_indices = self.triangulation.find_simplex(p1)
        if np.any(simplex_indices == -1):
            locs = np.where(simplex_indices == -1)
            logger.report_event("locations: {}".format(locs), log_level=logging.DEBUG)
            logger.report_event("points: {}".format(pts1[locs]), log_level=logging.DEBUG)
            logger.report_event("removing the above points", log_level=logging.DEBUG)
            pts1 = np.delete(pts1, locs, 0)
            pts2 = np.delete(pts2, locs, 0)
        return pts1, pts2

    def query_barycentrics(self, points):
        """Returns the mesh indices that surround a point, and the barycentric weights of those points"""
        p = points.copy()
        p[p < 0] = 0.01
        simplex_indices = self.triangulation.find_simplex(p)
        assert not np.any(simplex_indices == -1)

        # http://codereview.stackexchange.com/questions/41024/faster-computation-of-barycentric-coordinates-for-many-points
        X = self.triangulation.transform[simplex_indices, :2]
        Y = points - self.triangulation.transform[simplex_indices, 2]
        b = np.einsum('ijk,ik->ij', X, Y)
        pt_indices = self.triangulation.simplices[simplex_indices].astype(np.uint32)
        barys = np.c_[b, 1 - b.sum(axis=1)]

        pts_test = np.einsum('ijk,ij->ik', self.pts[pt_indices], barys)

        return self.triangulation.simplices[simplex_indices].astype(np.uint32), barys

class ElasticTPSMeshOptimizer(object):

    def __init__(self, **kwargs):
        self._checkpoints_dir = kwargs.get('checkpoints_dir', 'opt_checkpoints')
        self._mesh_spacing = kwargs.get('mesh_spacing', 1500)

        # set default values
        self._cross_slice_weight = kwargs.get("cross_slice_weight", 1.0)
        self._cross_slice_winsor = kwargs.get("cross_slice_winsor", 20)
        self._intra_slice_weight = kwargs.get("intra_slice_weight", 1.0)
        self._intra_slice_winsor = kwargs.get("intra_slice_winsor", 200)
        #intra_slice_weight = 3.0

        self._block_size = kwargs.get("block_size", 35)
        self._block_step = kwargs.get("block_step", 25)
        # min_iterations = kwargs.get("min_iterations", 200)
        self._max_iterations = kwargs.get("max_iterations", 5000)
        # mean_offset_threshold = kwargs.get("mean_offset_threshold", 5)
        # num_threads = kwargs.get("optimization_threads", 8)
        self._min_stepsize = kwargs.get("min_stepsize", 1e-20)
        self._assumed_model = kwargs.get("assumed_model", 3) # 0 - Translation (not supported), 1 - Rigid, 2 - Similarity (not supported), 3 - Affine
    #     filter_distance_multiplier = kwargs.get("filter_distance_multiplier", None)

        self._compute_mesh_area = not kwargs.get("avoid_mesh_area", False)
        self._debugged_layers = kwargs.get("debugged_layers", [])

    def _save_checkpoint_data(self, block_num, relevant_sec_idxs, meshes, links, norm_weights, structural_meshes):
        cp_fname = os.path.join(self._checkpoints_dir, "checkpoint_block_{}.pkl".format(str(block_num).zfill(5)))
        cp_fname_partial = "{}.partial".format(cp_fname)
        logger.report_event("Saving checkpoint block data to {}".format(cp_fname), log_level=logging.INFO)
        #print("saving tilespecs: {}".format(sorted(list(relevant_sec_idxs))))

        #  only store the relevant_sec_idxs data
        meshes = {sec_idx: m for sec_idx, m in meshes.items() if sec_idx in relevant_sec_idxs}
        structural_meshes = {sec_idx: m for sec_idx, m in structural_meshes.items() if sec_idx in relevant_sec_idxs}
        links = {(sec1_idx, sec2_idx): v for (sec1_idx, sec2_idx), v in links.items() if sec1_idx in relevant_sec_idxs or sec2_idx in relevant_sec_idxs}
        norm_weights = {(sec1_idx, sec2_idx): v for (sec1_idx, sec2_idx), v in norm_weights.items() if sec1_idx in relevant_sec_idxs or sec2_idx in relevant_sec_idxs}

        with open(cp_fname_partial, 'wb') as out:
            #pickle.dump([block_num, meshes, links, norm_weights, structural_meshes], out)
            pickle.dump([block_num, meshes, links, norm_weights, structural_meshes], out, pickle.HIGHEST_PROTOCOL)
        os.rename(cp_fname_partial, cp_fname)


    def _load_most_recent_checkpoint(self):
        block_num = 0
        meshes = {}
        links = {}
        norm_weights = {}
        structural_meshes = {}

        possible_fnames = sorted(glob.glob(os.path.join(self._checkpoints_dir, 'checkpoint_block_*.pkl')))
        # Find the latest valid checkpoint file
        cp_fname = None
        for fname in reversed(possible_fnames):
            if re.search(r'checkpoint_block_([0-9]+)\.pkl', os.path.basename(fname)):
                cp_fname = fname
                break

        if cp_fname is not None:
            # Found a valid checkpoint file
            logger.report_event("Loading checkpoint block data from {}".format(cp_fname), log_level=logging.INFO)
            with open(cp_fname, 'rb') as in_file:
                block_num, meshes, links, norm_weights, structural_meshes = pickle.load(in_file)
            block_num += 1 # increase block num (because the next iteration starts from the next block)
        else:
            logger.report_event("No checkpoint block data found in {}".format(self._checkpoints_dir), log_level=logging.INFO)

        return block_num, meshes, links, norm_weights, structural_meshes


    def _create_mesh(self, sec_idx, section, meshes, structural_meshes):
        if sec_idx not in meshes:
            logger.report_event("Creating mesh for section: {}".format(section.layer), log_level=logging.DEBUG)
            meshes[sec_idx] = Mesh(utils.generate_hexagonal_grid(section.bbox, self._mesh_spacing))

            # Build internal structural mesh
            # (edge_indices, edge_lengths, face_indices, face_areas)
            structural_meshes[sec_idx] = meshes[sec_idx].internal_structural_mesh()




    #def create_meshes_from_layout(layout, tilespecs_fnames, meshes, links, norm_weights, structural_meshes, ts_to_normed_layer, hex_spacing, dal_serializer):
    def _create_meshes_from_layout(self, layout, matches, block_lo, block_hi, meshes, links, norm_weights, structural_meshes):
        """Updates the given meshes and links dictionaries to include the meshes and links of the new sectoions (only new ones are loaded)"""
        for sec_idx in range(block_lo, block_hi):
            # Create all the meshes of the missing sections
            self._create_mesh(sec_idx, layout['sections'][sec_idx], meshes, structural_meshes)

        # Keep a list of relevant sections, so we can remove anything that is unneeded (reserve memory)
        relevant_sec_idxs = set()

        # Update the links and the normalized weights
        for sec1_idx in range(block_lo, block_hi):
            relevant_sec_idxs.add(sec1_idx)
            section1 = layout['sections'][sec1_idx]
            for sec2_idx in layout['neighbors'][sec1_idx]:
                relevant_sec_idxs.add(sec2_idx)
                section2 = layout['sections'][sec2_idx]

                # Load the mesh of sec2 (if it is not already there)
                self._create_mesh(sec2_idx, layout['sections'][sec2_idx], meshes, structural_meshes)

                # add the links (sec1_idx, sec2_idx) even if sec2 was in a previous batch
                if (sec1_idx, sec2_idx) not in links:

                    pts1 = matches[sec1_idx, sec2_idx][0]
                    pts2 = matches[sec1_idx, sec2_idx][1]

                    if len(pts1) > 0:
                        pts1, pts2 = meshes[sec1_idx].remove_unneeded_points(pts1, pts2)
                        pts2, pts1 = meshes[sec2_idx].remove_unneeded_points(pts2, pts1)

                        links[sec1_idx, sec2_idx] = (meshes[sec1_idx].query_barycentrics(pts1),
                                                     meshes[sec2_idx].query_barycentrics(pts2))
                        # update the weights between the neighboring sections
                        # Initialize the weights to the distance of each layer from the layer at ts1 (not including skipped layers)
                        norm_weights[sec1_idx, sec2_idx] = 1.0 / abs(section1.layer - section2.layer)

#         # Adjust the weights so that neighbors with the same distance will have the same adjusted weights
#         for ts1 in tilespecs_fnames:
#             # Assuming the neighbors are sorted
#             
#             # find the index of the first neighbor that is after ts1 (= the number of neighbors that have smaller layer#)
#             prev_neighbors_cnt = len([ts for ts in layout['neighbors'][ts1][0] if ts_to_normed_layer[ts] < ts_to_normed_layer[ts1]])
#                
#             for i in range(min(prev_neighbors_cnt, len(layout['neighbors'][ts1][1]) - prev_neighbors_cnt)):
#                 # section at (prev_neighbors_cnt - i - 1) needs to be matched to (prev_neighbors_cnt + i) in the sorted neighbors array
#                 pre_idx = (prev_neighbors_cnt - i - 1)
#                 post_idx = (prev_neighbors_cnt + i)
#                 pre_ts = layout['neighbors'][ts1][0][pre_idx]
#                 post_ts = layout['neighbors'][ts1][0][post_idx]
#                 pre_links_num = len(links[ts1, pre_ts][0][0])
#                 post_links_num = len(links[ts1, post_ts][0][0])
# 
#                 if pre_links_num > 0 and post_links_num > 0:
#                     max_links_num = max(pre_links_num, post_links_num)
#                     print("pre_links_num {}, post_links_num {}".format(pre_links_num, post_links_num))
#                     #norm_weights[ts1, pre_ts] = norm_weights[ts1, pre_ts] * (float(max_links_num) / pre_links_num)
#                     norm_weights[ts1, pre_ts] = 1.0 / (ts_to_normed_layer[ts1] - ts_to_normed_layer[pre_ts]) * (float(max_links_num) / pre_links_num)
#                     ##norm_weights[pre_ts, ts1] = norm_weights[pre_ts, ts1] * (float(max_links_num) / pre_links_num)
#                     #norm_weights[ts1, post_ts] = norm_weights[ts1, post_ts] * (float(max_links_num) / post_links_num)
#                     norm_weights[ts1, post_ts] = 1.0 / (ts_to_normed_layer[post_ts] - ts_to_normed_layer[ts1]) * (float(max_links_num) / post_links_num)
#                     ##norm_weights[post_ts, ts1] = norm_weights[post_ts, ts1] * (float(max_links_num) / post_links_num)
#                     print("norm_weights[{}, {}] = {}, norm_weights[{}, {}] = {}".format(
#                         os.path.basename(ts1), os.path.basename(pre_ts), norm_weights[ts1, pre_ts],
#                         os.path.basename(ts1), os.path.basename(post_ts), norm_weights[ts1, post_ts]))
#                     #print("norm_weights[{}, {}] = {}, norm_weights[{}, {}] = {}".format(
#                     #    os.path.basename(ts1), os.path.basename(pre_ts), norm_weights[pre_ts, ts1],
#                     #    os.path.basename(ts1), os.path.basename(post_ts), norm_weights[post_ts, ts1]))


        # Remove the unneeded tilespecs from the dictionarys
        meshes = {sec_idx: m for sec_idx, m in meshes.items() if sec_idx in relevant_sec_idxs}
        structural_meshes = {sec_idx: m for sec_idx, m in structural_meshes.items() if sec_idx in relevant_sec_idxs}
        links = {(sec1_idx, sec2_idx): v for (sec1_idx, sec2_idx), v in links.items() if sec1_idx in relevant_sec_idxs or sec2_idx in relevant_sec_idxs}
        norm_weights = {(sec1_idx, sec2_idx): v for (sec1_idx, sec2_idx), v in norm_weights.items() if sec1_idx in relevant_sec_idxs or sec2_idx in relevant_sec_idxs}

        return meshes, links, norm_weights, structural_meshes

    def _get_transform_matrix(self, pts1, pts2):
        model = self._assumed_model
        if model == 1:
            return align_rigid(pts1, pts2)
        elif model == 3:
            return Haffine_from_points(pts1, pts2)
        else:
            logger.report_event("Unsupported transformation model type", log_level=logging.ERROR)
            return None

    def _transform_mesh_tps(self, pts_src_lists, pts_dst_lists, weights_list, mesh_pts):
        transformed_mesh_pts = np.zeros_like(mesh_pts)
        # Apply each transformation by the given matched points (weighted), and sum it all into transformed_mesh_pts
        print("1")
        logger.report_event("starting thinplate splines 1", log_level=logging.DEBUG)
        for pts_src, pts_dst, w in zip(pts_src_lists, pts_dst_lists, weights_list):
            #model = models.PointsTransformModel((pts_src, pts_dst))
            #transformed_mesh_pts += model.apply(mesh_pts) * w
            logger.report_event("Creating thinplate splines 1.1: {} control pts".format(len(pts_src)), log_level=logging.DEBUG)
            model = ThinPlateSplines(pts_src, pts_dst)
            print("1.1")
            logger.report_event("Applying thinplate splines 1.2: {} mesh pts".format(len(mesh_pts)), log_level=logging.DEBUG)
            transformed_mesh_pts += model.apply(mesh_pts, 20000) * w
            logger.report_event("Done applying thinplate splines 1.2: {} mesh pts".format(len(mesh_pts)), log_level=logging.DEBUG)
            print("1.2")

        # Normalize the transformed points
        transformed_mesh_pts /= np.sum(weights_list)
        print("2")
        return transformed_mesh_pts


    def _pre_optimize(self, layout, block_lo, block_hi, meshes, links, norm_weights, structural_meshes):
        # Compute the initial affine pre-alignment for each of the relevant sections
        pre_alignment_block_lo = max(1, block_lo)
        # on all blocks after the first, should avoid a pre affine transformation of anything that was already pre-aligned
        if block_lo > 0:
            pre_alignment_block_lo = min(block_lo + (self._block_size - self._block_step), block_hi)
        for active_sec_idx in range(pre_alignment_block_lo, block_hi):
            active_sec_name = layout['sections'][active_sec_idx].canonical_section_name
            logger.report_event("Before affine (sec {}): {}".format(active_sec_name, ts_mean_offsets(meshes, links, active_sec_idx, plot=False)), log_level=logging.INFO)
            rot = 0
            tran = 0
            count = 0

            #all_H = np.zeros((3,3))
            for neighbor_sec_idx in layout['neighbors'][active_sec_idx]:
                if neighbor_sec_idx < active_sec_idx:
                    # take both (active_sec, neighbor_sec) and (neighbor_sec, active_sec) into account
                    for (sec1_idx, sec2_idx), ((idx1, w1), (idx2, w2)) in [((active_sec_idx, neighbor_sec_idx), links[active_sec_idx, neighbor_sec_idx]), ((neighbor_sec_idx, active_sec_idx), links[neighbor_sec_idx, active_sec_idx])]:
                        #if active_ts in (ts1, ts2) and (layers[ts1] <= layers[active_ts]) and (layers[ts2] <= layers[active_ts]):
                        pts1 = np.einsum('ijk,ij->ik', meshes[sec1_idx].pts[idx1], w1)
                        pts2 = np.einsum('ijk,ij->ik', meshes[sec2_idx].pts[idx2], w2)
                        logger.report_event("Matches # (sections {}->{}): {}.".format(sec1_idx, sec2_idx, pts1.shape[0]), log_level=logging.INFO)#DEBUG)
                        if sec1_idx == active_sec_idx:
                            cur_rot, cur_tran = self._get_transform_matrix(pts1, pts2)
                            cur_norm_weights = norm_weights[sec1_idx, sec2_idx]
                        else: # sec2_idx == active_sec_idx
                            cur_rot, cur_tran = self._get_transform_matrix(pts2, pts1)
                            cur_norm_weights = norm_weights[sec2_idx, sec1_idx]

                        # Average the affine transformation by the number of matches between the two sections
                        rot += pts1.shape[0] * cur_norm_weights * cur_rot
                        tran += pts1.shape[0] * cur_norm_weights * cur_tran
                        count += pts1.shape[0] * cur_norm_weights

            if count == 0:
                logger.report_event("Error: no matches found for section {}.".format(active_sec_name), log_level=logging.ERROR)
                sys.exit(1)

            # normalize the transformation
            rot = rot * (1.0 / count)
            tran = tran * (1.0 / count)
            #print("rot:\n{}\ntran:\n{}".format(rot, tran))
            # transform the points
            meshes[active_sec_idx].pts = np.dot(meshes[active_sec_idx].pts, rot) + tran
            logger.report_event("After affine (sec {}): {}".format(active_sec_name, ts_mean_offsets(meshes, links, active_sec_idx, plot=False)), log_level=logging.INFO)


    def _grad_descent_optimize(self, layout, block_lo, block_hi, meshes, links, norm_weights, structural_meshes):
        stepsize = 0.0001
        momentum = 0.5
        prev_cost = np.inf
        gradients_with_momentum = {sec_idx: 0.0 for sec_idx in range(block_lo, block_hi)}
        old_pts = None

        old_pts = {sec_idx: m.pts.copy() for sec_idx, m in meshes.items() if sec_idx in range(block_lo, block_hi)}

        for iter in range(self._max_iterations):
            cost = 0.0

            gradients = {sec_idx: np.zeros_like(meshes[sec_idx].pts) for sec_idx in range(block_lo, block_hi)}

            # Compute the cost of the internal and external links
            for sec_idx in range(block_lo, block_hi):
                cost += mesh_derivs_elastic.internal_grad(self._compute_mesh_area, meshes[sec_idx].pts, gradients[sec_idx],
                                                  *((structural_meshes[sec_idx]) +
                                                    (self._intra_slice_weight, self._intra_slice_winsor)))


            for active_sec_idx in range(block_lo, block_hi):
                active_sec_layer = layout['sections'][active_sec_idx].layer
                for neighbor_sec_idx in layout['neighbors'][active_sec_idx]:
                    neighbor_layer = layout['sections'][neighbor_sec_idx].layer
                    if neighbor_layer < active_sec_layer:
                        # take both (active_sec, neighbor_sec) and (neighbor_sec, active_sec) into account
                        for (sec1_idx, sec2_idx), ((idx1, w1), (idx2, w2)) in [((active_sec_idx, neighbor_sec_idx), links[active_sec_idx, neighbor_sec_idx]), ((neighbor_sec_idx, active_sec_idx), links[neighbor_sec_idx, active_sec_idx])]:
#                             if ts1 not in block_tss and ts2 not in block_tss:
#                                 continue
#                             # only take the previous sections into account (if one of the sections is in the next block, disregard its cost)
#                             if ts1 not in block_tss and layers[ts1] >= layers[block_tss[-1]]:
#                                 continue
#                             if ts2 not in block_tss and layers[ts2] >= layers[block_tss[-1]]:
#                                 continue
                            # Add sections that are not part of the block_tss (the sections that have connections to the current block_tss)
                            if sec1_idx not in gradients:
                                gradients[sec1_idx] = np.zeros_like(meshes[sec1_idx].pts)
                            if sec2_idx not in gradients:
                                gradients[sec2_idx] = np.zeros_like(meshes[sec2_idx].pts)
                                
                            cost += mesh_derivs_elastic.external_grad(meshes[sec1_idx].pts, meshes[sec2_idx].pts,
                                                              gradients[sec1_idx], gradients[sec2_idx],
                                                              idx1, w1,
                                                              idx2, w2,
                                                              self._cross_slice_weight * norm_weights[sec1_idx, sec2_idx], self._cross_slice_winsor)

            if cost < prev_cost and not np.isinf(cost):
                prev_cost = cost
                stepsize *= 1.1
                if stepsize > 1.0:
                    stepsize = 1.0
                # update with new gradients
                for sec_idx in gradients_with_momentum:
                    gradients_with_momentum[sec_idx] = gradients[sec_idx] + momentum * gradients_with_momentum[sec_idx]
                old_pts = {sec_idx: m.pts.copy() for sec_idx, m in meshes.items() if sec_idx in range(block_lo, block_hi)}
                for sec_idx in range(block_lo, block_hi):
                    meshes[sec_idx].pts -= stepsize * gradients_with_momentum[sec_idx]
                # if iter % 500 == 0:
                #     print("{} Good step: cost {}  stepsize {}".format(iter, cost, stepsize))
            else:  # we took a bad step: undo it, scale down stepsize, and start over
                for sec_idx in range(block_lo, block_hi):
                    meshes[sec_idx].pts = old_pts[sec_idx]
                stepsize *= 0.5
                gradients_with_momentum = {sec_idx: 0 for sec_idx in range(block_lo, block_hi)}
                # if iter % 500 == 0:
                #     print("{} Bad step: stepsize {}".format(iter, stepsize))
            if iter % 100 == 0:
                logger.report_event("region [{}, {}], iter {}: C: {}, MO: {}, S: {}".format(block_lo, block_hi, iter, cost, mean_offsets(meshes, links, block_hi, plot=False, from_sec_idx=block_lo), stepsize), log_level=logging.INFO)

#             # Save animation for debugging
#             if len(debugged_layers) > 0:
#                 for debugged_layer in debugged_layers:
#                     active_ts = layer_to_ts[debugged_layer]
#                     if active_ts in layout['tilespecs'][block_lo:min(block_lo+block_step, len(layout['tilespecs']))]: # only the layers that are part of the current block (not in the non-overlapping part of the next block)
#                         pickle.dump([active_ts, meshes[active_ts].pts], open(os.path.join(DEBUG_DIR, "post_iter{}_{}.pickle".format(str(iter).zfill(5), os.path.basename(active_ts).replace(' ', '_'))), "w"))
#                         #plot_points(meshes[active_ts].pts, os.path.join(DEBUG_DIR, "post_iter{}_{}.png".format(str(iter).zfill(5), os.path.basename(active_ts).replace(' ', '_'))))


            for sec_idx in range(block_lo, block_hi):
                assert not np.any(~ np.isfinite(meshes[sec_idx].pts))

            # If stepsize is too small (won't make any difference), stop the iterations
            if stepsize < self._min_stepsize:
                break
        logger.report_event("region [{}, {}], last MO: {}\n".format(block_lo, block_hi, mean_offsets(meshes, links, block_hi, plot=False, from_sec_idx=block_lo)), log_level=logging.INFO)






    def optimize(self, layout, matches, export_mesh_func, pool):
#         logger.report_event("Debugged layers: {}".format(debugged_layers), log_level=logging.INFO)
#         DEBUG_DIR = None
#         debugged_layers = []
#         if len(self._debugged_layers) > 0:
#             debugged_layers = map(int, debugged_layers.split(','))
#             DEBUG_DIR = os.path.join("debug_optimization", "logs_{}".format(datetime.datetime.now().isoformat()))
#             if not os.path.exists(DEBUG_DIR):
#                 os.makedirs(DEBUG_DIR)

        # Create parallel files reader
        # files_reader = ParallelProcessesFilesReader(threads_num)

        meshes = {}
        links = {}
        norm_weights = {}

        # Build internal structural mesh
        # (edge_indices, edge_lengths, face_indices, face_areas)
        ##structural_meshes = {ts: mesh.internal_structural_mesh() for ts, mesh in meshes.items()}
        structural_meshes = {}


        # Create checkpoints directory if needed
        cp_block_num = 0
        if self._checkpoints_dir is not None:
            if not os.path.exists(self._checkpoints_dir):
                os.makedirs(self._checkpoints_dir)
            # Load checkpoint if there is one
            cp_block_num, meshes, links, norm_weights, structural_meshes = self._load_most_recent_checkpoint()

            # Filter out anything that's no longer in the layout
            links_to_remove = []
            for (sec1_idx, sec2_idx) in links.keys():
                if sec2_idx not in layout['neighbors'][sec1_idx] and sec1_idx not in layout['neighbors'][sec2_idx]:
                    links_to_remove.append((sec1_idx, sec2_idx))
            for k in links_to_remove:
                logger.report_event("Filtering {} from previoulsy loaded links".format(k), log_level=logging.INFO)
                del links[k]

#         # Save the initial pickle file for the debugged layers
#         for debugged_layer in debugged_layers:
#             active_ts = layer_to_ts[debugged_layer]
#             pickle.dump([active_ts, meshes[active_ts].pts], open(os.path.join(DEBUG_DIR, "pre_affine_{}.pickle".format(os.path.basename(active_ts).replace(' ', '_'))), "w"))
#             #plot_points(meshes[active_ts].pts, os.path.join(DEBUG_DIR, "pre_affine_{}.png".format(os.path.basename(active_ts).replace(' ', '_'))))

        # Work in blocks, and load only relevant meshes
        for block_lo in (range(cp_block_num * self._block_step, max(1, len(layout['sections'])), self._block_step)):
            block_hi = min(block_lo + self._block_size, len(layout['sections']))
            ## FOR DEBUG!!!
            #block_hi = min(block_lo + 3, len(layout['sections']))
            last_block = block_hi == len(layout['sections'])

            # Load needed meshes
            meshes, links, norm_weights, structural_meshes = self._create_meshes_from_layout(layout, matches, block_lo, block_hi, meshes, links, norm_weights, structural_meshes)
            

            # Compute the initial affine pre-alignment for each of the relevant sections
            pre_alignment_block_lo = max(1, block_lo)
            # on all blocks after the first, should avoid a pre affine transformation of anything that was already pre-aligned
            if block_lo > 0:
                pre_alignment_block_lo = min(block_lo + (self._block_size - self._block_step), block_hi)
            for active_sec_idx in range(pre_alignment_block_lo, block_hi):
                active_sec_name = layout['sections'][active_sec_idx].canonical_section_name
                logger.report_event("Before tps (sec {}): {}".format(active_sec_name, ts_mean_offsets(meshes, links, active_sec_idx, plot=False)), log_level=logging.INFO)

                src_pts_lists = []
                dst_pts_lists = []
                weights_list = []
                match_sets_num = 0

                for neighbor_sec_idx in layout['neighbors'][active_sec_idx]:
                    if neighbor_sec_idx < active_sec_idx:
                        # take both (active_sec, neighbor_sec) and (neighbor_sec, active_sec) into account
                        for (sec1_idx, sec2_idx), ((idx1, w1), (idx2, w2)) in [((active_sec_idx, neighbor_sec_idx), links[active_sec_idx, neighbor_sec_idx]), ((neighbor_sec_idx, active_sec_idx), links[neighbor_sec_idx, active_sec_idx])]:
                            #if active_ts in (ts1, ts2) and (layers[ts1] <= layers[active_ts]) and (layers[ts2] <= layers[active_ts]):
                            pts1 = np.einsum('ijk,ij->ik', meshes[sec1_idx].pts[idx1], w1)
                            pts2 = np.einsum('ijk,ij->ik', meshes[sec2_idx].pts[idx2], w2)
                            logger.report_event("Matches # (sections {}->{}): {}.".format(sec1_idx, sec2_idx, pts1.shape[0]), log_level=logging.INFO)#DEBUG)
                            if sec1_idx == active_sec_idx:
                                src_pts_lists.append(pts1)
                                dst_pts_lists.append(pts2)
                                weights_list.append(norm_weights[sec1_idx, sec2_idx])
                                cur_norm_weights = norm_weights[sec1_idx, sec2_idx]
                                match_sets_num += 1
                            else: # sec2_idx == active_sec_idx
                                src_pts_lists.append(pts2)
                                dst_pts_lists.append(pts1)
                                weights_list.append(norm_weights[sec2_idx, sec1_idx])
                                cur_norm_weights = norm_weights[sec2_idx, sec1_idx]
                                match_sets_num += 1

                if match_sets_num == 0:
                    logger.report_event("Error: no matches found for section {}.".format(active_ts), log_level=logging.ERROR)
                    sys.exit(1)
                # Apply the transformation to the mesh points
                meshes[active_sec_idx].pts = self._transform_mesh_tps(src_pts_lists, dst_pts_lists, weights_list, meshes[active_sec_idx].pts)
                logger.report_event("After tps (sec {}): {}".format(active_sec_name, ts_mean_offsets(meshes, links, active_sec_idx, plot=False)), log_level=logging.INFO)



            if SHOW_BLOCK_FINAL_MO:
                logger.report_event("Final Block MO:", log_level=logging.INFO)
                for active_sec_idx in range(block_lo, block_hi):
                    active_sec_name = layout['sections'][active_sec_idx].canonical_section_name
                    logger.report_event(" Section {}: {}".format(active_sec_name, ts_mean_offsets(meshes, links, active_sec_idx, plot=False)), log_level=logging.INFO)




            # Exporting the final meshes of the block
            logger.report_event("Saving the meshes of the finalized sections in the previous block", log_level=logging.INFO)
            if last_block:
                exported_sec_idxs = range(block_lo, block_hi)
            else:
                exported_sec_idxs = range(block_lo, block_hi - (self._block_size - self._block_step))
            for active_sec_idx in exported_sec_idxs:
                #out_positions = [meshes[active_sec_idx].orig_pts,
                #                 meshes[active_sec_idx].pts]
                export_mesh_func(layout['sections'][active_sec_idx], meshes[active_sec_idx].orig_pts, meshes[active_sec_idx].pts, self._mesh_spacing)

            # Checkpoint the current block (if it is not the last)
            if self._checkpoints_dir is not None and not last_block:
                # Only save the ones that are in the overlap or are connected to the overlapping tilespecs
                relevant_sec_idxs = set(list(range(block_hi - (self._block_size - self._block_step), block_hi)))
                for sec_idx in range(block_hi - (self._block_size - self._block_step), block_hi):
                    for neighbor_sec_idx in layout['neighbors'][sec_idx]:
                        relevant_sec_idxs.add(neighbor_sec_idx)
                self._save_checkpoint_data(block_lo / self._block_step, relevant_sec_idxs, meshes, links, norm_weights, structural_meshes)



    #    if SHOW_FINAL_MO:
    #        logger.report_event("Final MO:", log_level=logging.INFO)
    #        for active_ts in layout['tilespecs']:
    #            logger.report_event(" Section {}: {}".format(os.path.basename(active_ts), ts_mean_offsets(meshes, links, active_ts, plot=False)), log_level=logging.INFO)

        # Prepare per-layer output
        out_positions = {}

#         for i, ts in enumerate(meshes):
#             out_positions[ts] = [meshes[ts].orig_pts,
#                                  meshes[ts].pts]
#             if "DUMP_LOCATIONS" in os.environ:
#                 pickle.dump([ts, out_positions[ts]], open("newpos{}.pickle".format(i), "w"))

    #    if tsfile_to_layerid is not None:
    #        for tsfile, layerid in tsfile_to_layerid.items():
    #            if layerid in present_slices:
    #                meshidx = mesh_pt_offsets[layerid]
    #                out_positions[tsfile] = [(mesh.pts / mesh.layer_scale).tolist(),
    #                                         (all_mesh_pts[meshidx, :] / mesh.layer_scale).tolist()]
    #                if "DUMP_LOCATIONS" in os.environ:
    #                    pickle.dump([tsfile, out_positions[tsfile]], open("newpos{}.pickle".format(meshidx), "w"))
    #            else:
    #                out_positions[tsfile] = [(mesh.pts / mesh.layer_scale).tolist(),
    #                                         (mesh.pts / mesh.layer_scale).tolist()]

        return out_positions





def mean_offsets(meshes, links, stop_at_sec_idx, plot=False, from_sec_idx=None):
    means = []
    for (sec1_idx, sec2_idx), ((idx1, w1), (idx2, w2)) in links.items():
        if sec1_idx > stop_at_sec_idx or sec2_idx > stop_at_sec_idx:
            continue
        if from_sec_idx is not None:
            if sec1_idx < from_sec_idx or sec2_idx < from_sec_idx:
                continue
        pts1 = np.einsum('ijk,ij->ik', meshes[sec1_idx].pts[idx1], w1)
        pts2 = np.einsum('ijk,ij->ik', meshes[sec2_idx].pts[idx2], w2)
        if plot:
            lines = [[p1, p2] for p1, p2 in zip(pts1, pts2)]

            lc = mc.LineCollection(lines)
            pylab.figure()
            pylab.title(ts1 + ' ' + ts2)
            pylab.gca().add_collection(lc)
            pylab.scatter(pts1[:, 0], pts1[:, 1])
            pylab.gca().autoscale()
            
        lens = np.sqrt(((pts1 - pts2) ** 2).sum(axis=1))
        #print(np.mean(lens), np.min(lens), np.max(lens))
        means.append(np.median(lens))
    if len(means) == 0:
        return 0
    return np.mean(means)

def ts_mean_offsets(meshes, links, sec_idx, plot=False):
    means = []
    for (sec1_idx, sec2_idx), ((idx1, w1), (idx2, w2)) in links.items():
        if sec_idx not in (sec1_idx, sec2_idx):
            continue
        if sec1_idx > sec_idx or sec2_idx > sec_idx:
            continue
        pts1 = np.einsum('ijk,ij->ik', meshes[sec1_idx].pts[idx1], w1)
        pts2 = np.einsum('ijk,ij->ik', meshes[sec2_idx].pts[idx2], w2)
        if plot:
            lines = [[p1, p2] for p1, p2 in zip(pts1, pts2)]

            lc = mc.LineCollection(lines)
            pylab.figure()
            pylab.title(ts1 + ' ' + ts2)
            pylab.gca().add_collection(lc)
            pylab.scatter(pts1[:, 0], pts1[:, 1])
            pylab.gca().autoscale()
            
        lens = np.sqrt(((pts1 - pts2) ** 2).sum(axis=1))
        #print(np.mean(lens), np.min(lens), np.max(lens))
        means.append(np.median(lens))
    if len(means) == 0:
        return 0
    return np.mean(means)


def plot_offsets(meshes, links, sec_idx, fname_prefix):
    for (sec1_idx, sec2_idx), ((idx1, w1), (idx2, w2)) in links.items():
        if sec1_idx != sec_idx:
            continue
        pts1 = np.einsum('ijk,ij->ik', meshes[sec1_idx].pts[idx1], w1)
        pts2 = np.einsum('ijk,ij->ik', meshes[sec2_idx].pts[idx2], w2)
        diffs2 = pts2 - pts1
        #lines = [[p1, p2] for p1, p2 in zip(pts1, pts2)]

        fname = "{}{}.png".format(fname_prefix, str(sec1_idx) + '_' + str(sec2_idx))
        #lc = mc.LineCollection(lines)
        pylab.figure()
        pylab.title(sec1_idx + ' ' + sec2_idx)
        #pylab.gca().add_collection(lc)
        pylab.quiver(pts1[:, 0], pts1[:, 1], diffs2[:, 0], diffs2[:, 1])
        #pylab.scatter(pts1[:, 0], pts1[:, 1])
        #pylab.gca().autoscale()
        pylab.savefig(fname)

def plot_points(pts, fname):
    pylab.figure()
    pylab.scatter(pts[:, 0], pts[:, 1])
    #pylab.gca().autoscale()
    pylab.savefig(fname)

def align_rigid(pts1, pts2):
    # find R,T that bring pts1 to pts2

    # convert to column vectors
    pts1 = pts1.T
    pts2 = pts2.T
    
    m1 = pts1.mean(axis=1, keepdims=True)
    m2 = pts2.mean(axis=1, keepdims=True)
    U, S, VT = np.linalg.svd(np.dot((pts1 - m1), (pts2 - m2).T))
    R = np.dot(VT.T, U.T)
    T = - np.dot(R, m1) + m2

    # convert to form used for left-multiplying row vectors
    return R.T, T.T


def Haffine_from_points(fp, tp):
    """Find H, affine transformation, s.t. tp is affine transformation of fp.
       Taken from 'Programming Computer Vision with Python: Tools and algorithms for analyzing images'
    """

    assert(fp.shape == tp.shape)

    fp = fp.T
    tp = tp.T

    # condition points
    # --- from points ---
    m = fp.mean(axis=1, keepdims=True)
    maxstd = np.max(np.std(fp, axis=1)) + 1e-9
    C1 = np.diag([1.0/maxstd, 1.0/maxstd, 1.0])
    C1[0][2] = -m[0] / maxstd
    C1[1][2] = -m[1] / maxstd
    fp_cond = np.dot(C1[:2, :2], fp) + np.asarray(C1[:2, 2]).reshape((2, 1))
    
    # --- to points ---
    m = tp.mean(axis=1, keepdims=True)
    C2 = C1.copy() # must use same scaling for both point sets
    C2[0][2] = -m[0] / maxstd
    C2[1][2] = -m[1] / maxstd
    tp_cond = np.dot(C2[:2, :2], tp) + np.asarray(C2[:2, 2]).reshape((2, 1))

    # conditioned points have mean zero, so translation is zero
    A = np.concatenate((fp_cond, tp_cond), axis=0)
    #U,S,V = np.linalg.svd(A.T, full_matrices=True)
    #U,S,V = spLA.svd(A.T)
    #U,S,V = spLA.svds(A.T, k=3, ncv=50, maxiter=1e9, return_singular_vectors='vh') # doesn't work well
    # Randomized svd uses much less memory and is much faster than the numpy/scipy versions
    n_oversamples = min(10, max(MAX_SAMPLED_MATCHES, fp.shape[1] // 10))
    U, S, V = randomized_svd(A.T, 4, n_oversamples)

    # create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = np.concatenate((np.dot(C, np.linalg.pinv(B)), np.zeros((2, 1))), axis=1)
    H = np.vstack((tmp2, [0, 0, 1]))

    # decondition
    H = np.dot(np.linalg.inv(C2), np.dot(H, C1))

    m = H / H[2][2]
    return m[:2,:2].T, m[:2, 2] # rotation part (transposed) and translation


# def affine_from_points_ransac(fp, tp):
#     """
#     Find H, affine transformation, s.t. tp is affine transformation of fp and get rid of outliers using RANSAC.
#     """
# 
#     assert(fp.shape == tp.shape)
# 
#     if len(fp) < 3:
#         return None, None
# 
#     match_points = np.array([fp, tp])
# 
#     model_index = 3
#     iterations = 2000
#     max_epsilon = 40
#     min_inlier_ratio = 0.01
#     min_num_inlier = 8
#     max_trust = 3
#     det_delta = 0.95
#     max_stretch = 0.92
#     model, filtered_matches = ransac.filter_matches(match_points, match_points, model_index, iterations, max_epsilon, min_inlier_ratio, min_num_inlier, max_trust, det_delta, max_stretch, robust_filter=False)
# 
#     if model is None:
#         return None, None
# 
#     m = model.get_matrix()
# 
#     f_matches_transformed = np.dot(filtered_matches[0], m[:2, :2].T) + m[:2, 2]
#     f_matches_mean_dist = np.mean(np.sqrt(((f_matches_transformed - filtered_matches[1]) ** 2).sum(axis=1)))
#     all_matches_transformed = np.dot(fp, m[:2, :2].T) + m[:2, 2]
#     all_matches_mean_dist = np.mean(np.sqrt(((all_matches_transformed - tp) ** 2).sum(axis=1)))
# 
#     print("filtered_matches mean dist: {} match_points mean_dist: {}".format(f_matches_mean_dist, all_matches_mean_dist))
#     return m[:2, :2].T, m[:2, 2]


