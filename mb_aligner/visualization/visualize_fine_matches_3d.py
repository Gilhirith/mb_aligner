import numpy as np
import cv2
import sys
import shapely.geometry
import shapely.ops
from descartes.patch import PolygonPatch
import matplotlib
#matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from collections import defaultdict
from sklearn.utils.extmath import randomized_svd
import os
from mb_aligner.common.intermediate_results_dal_pickle import IntermediateResultsDALPickle
from enum import Enum
import ujson as json
from mb_aligner.dal.section import Section
from mb_aligner.visualization.visualize_pre_match_3d_affine_result import PreMatch3DAffineResultVisualizer
import rh_renderer.models


class FineMatches3DVisualizer(object):

    class ColorMode(Enum):
        ANGLES = 0
        DISTANCES = 1

    def __init__(self, sec_src, sec_dst, color_mode=ColorMode.ANGLES):
        self._sec_src = sec_src
        self._sec_dst = sec_dst
        self._color_mode = color_mode
        self._all_points_src = []
        self._all_points_dest = []


    def _visualize_matches_hsv_after_affine(self, title=None, use_affine=False):
        # create a single list of points from source to dest
        assert len(self._all_points_src) > 0

        print(len(self._all_points_src))
        from_mesh = np.array(self._all_points_src, dtype=np.float64)
        to_mesh = np.array(self._all_points_dest, dtype=np.float64)

        if use_affine:
            # apply an affine transformation to get rid of large displacements/rotations/scale
            affine_mat = Haffine_from_points(from_mesh, to_mesh)
            print(affine_mat)
            #print("Inverted affine mat:")
            #print(np.linalg.inv(affine_mat))
            from_mesh = np.dot(affine_mat[:2, :2],
                               from_mesh.T).T + np.asarray(affine_mat[:2, 2]).reshape((1, 2))
        else:
            #from_mesh_shift = np.min(from_mesh, axis=0)
            from_mesh_shift = np.mean(from_mesh, axis=0)
            print("Shifiting source points: {}".format(from_mesh_shift))
            from_mesh -= from_mesh_shift
            #to_mesh_shift = np.min(to_mesh, axis=0)
            to_mesh_shift = np.mean(to_mesh, axis=0)
            print("Shifiting target points: {}".format(to_mesh_shift))
            to_mesh -= to_mesh_shift
        
        X = from_mesh[:, 0]
        Y = from_mesh[:, 1]
        U = to_mesh[:, 0] - X
        V = to_mesh[:, 1] - Y
        mag, ang = cv2.cartToPolar(U, V)
        ang_deg = ang *180/np.pi
        print("ang_deg - min: {}, mean: {}, median: {}, max: {}".format(np.min(ang_deg), np.mean(ang_deg), np.median(ang_deg), np.max(ang_deg)))
        print("mag - min: {}, mean: {}, median: {}, max: {}".format(np.min(mag), np.mean(mag), np.median(mag), np.max(mag)))
        mag_hist = np.histogram(mag, bins=10)
        print("mag hist%: {}".format(mag_hist[0]/float(len(mag))))
        print("mag hist bin_edges: {}".format(mag_hist[1]))
        print("mag percentiles [0, 10, 25, 50, 75]: {}".format(np.percentile(mag, [0, 10, 25, 50, 75])))

        # cap the magintude by 2*median(mag) so we will be able to see the non-outlier orientations
        mag_threshold = 4 * np.median(mag)
        mag_outlier_masks = mag > mag_threshold
        mag_capped = np.copy(mag)
        mag_capped[mag_outlier_masks] = mag_threshold

        # normalize the capped magnitude
        mag_capped_norm = cv2.normalize(mag_capped, None, 0, 1, cv2.NORM_MINMAX)
        if self._color_mode == FineMatches3DVisualizer.ColorMode.ANGLES:
            my_colors = hs_to_rgb(ang_deg, mag_capped_norm)
            # Set the colors of the capped vectors to black
            my_colors[np.where(mag_outlier_masks == True)[0]] = np.array([1.0, 1.0, 1.0])
        elif self._color_mode == FineMatches3DVisualizer.ColorMode.DISTANCES:
            my_colors = cm.rainbow(mag_capped_norm.flatten())
       
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.quiver(X, Y, U, V,        # data
                  color=my_colors,
                  #ang_deg,                   # colour the arrows based on this array
                  #cmap=cm.hsv,     # colour map
                  headlength=7)        # length of the arrows

        #plt.colorbar()                  # adds the colour bar

        ax.invert_yaxis()

        if title is not None:
            ax.set_title(title)
        ax.set_axis_bgcolor('black')

        self._add_mfovs_layout(fig, ax, self._sec_src, affine_mat)
        return fig, ax
    

    def _add_mfovs_layout(self, fig, ax, sec, affine_mat):
        affine_model = rh_renderer.models.AffineModel(affine_mat)
        global_affine_transformation = {cur_mfov.mfov_index:affine_model for cur_mfov in sec.mfovs()}
        sec_mfovs_tiles_proj, sec_min_xy_proj = PreMatch3DAffineResultVisualizer.create_section_tiles_projections(sec, global_affine_transformation)

#         # scale the projections
#         for cur_mfov in sec.mfovs():
#             cur_mfov_index = cur_mfov.mfov_index
#             for idx in range(len(sec_mfovs_tiles_proj[cur_mfov_index])):
#                 #sec_mfovs_tiles_proj[cur_mfov_index][idx] -= min_xy_proj
#                 sec_mfovs_tiles_proj[cur_mfov_index][idx] *= self._scale

        sec_polygons_and_fill = [(PreMatch3DAffineResultVisualizer._get_pts_unified_polygon(mfov_pts_list), False) for mfov_pts_list in sec_mfovs_tiles_proj.values()]

        PreMatch3DAffineResultVisualizer._add_polygons(ax, sec_polygons_and_fill)

        # Add the mfov text
        sec_centers_proj = {mfov_index: PreMatch3DAffineResultVisualizer._find_center(mfov_pts_list) for mfov_index, mfov_pts_list in sec_mfovs_tiles_proj.items()}
        PreMatch3DAffineResultVisualizer._add_mfovs_text(ax, sec_centers_proj)
        


    def add_matches(self, points_src, points_dest):
        assert(len(points_src) == len(points_dest))

        if len(points_src) > 0:
            self._all_points_src.extend(points_src)
            self._all_points_dest.extend(points_dest)

#     def add_matches_file(self, fine_matches_fname):
#         """Given a 3d pre-match output file that has a per-mfov affine transformation,
#            generates two figures, one of the outline of the mfovs of the first section (the original mfov outline),
#            and the outline of the mfovs after transforming them to the second section
#         """
#         print "adding file: {}".format(fine_matches_fname)
#         exists, inter_results_dal = IntermediateResultsDALPickle.load_prev_single_file_results(fine_matches_fname)
# 
#         ts1 = inter_results_dal["metadata"]['sec1']
#         ts2 = inter_results_dal["metadata"]['sec2']
# 
# #         if self._ts_src is None and self._ts_dest is None:
# #             self._ts_src = ts1
# #             self._ts_dest = ts2
# #         
# #         assert self._ts_src == ts1 and self._ts_dest == ts2
# 
#         # Add all point matches
#         points_src = []
#         points_dest = []
#         for m in data["pointmatches"]:
#             points_src.append(np.array(m["point1"]))
#             points_dest.append(np.array(m["point2"]))
# 
#         self.add_matches(points_src, points_dest)


    @staticmethod
    def _load_tilespec(fname):
        with open(fname, 'rb') as in_f:
            tilespec = json.load(in_f)
        return Section.create_from_tilespec(tilespec)

    def visualize_fine_matches_3d(self, title=None, use_affine=False):
        """Generates the figure that show the flow of matches after an initial affine transformation.
           Shows the information from all the previously loaded matches (files/matches).
        """
        if title is None:
            title = "{} -> {} ({})".format(self._sec_src.canonical_section_name_no_layer, self._sec_dst.canonical_section_name_no_layer, 'affine' if use_affine else 'translation')

        return self._visualize_matches_hsv_after_affine(title, use_affine)



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
    n_oversamples = min(10, max(5000, fp.shape[1] // 10))
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
    return m


def hs_to_rgb(h_array, s_array):
    # Turns our own representation of hsv values (h_array that is the angle, and s_array that is the normalized magnitude) to an rgb array
    # Assumption V is always 1.0
    assert(len(h_array) == len(s_array))
    V = 1.0
    rgb = np.zeros((len(h_array), 3), dtype=np.float32)
    zero_idxs = s_array <= 0.0
    for idx, val in enumerate(zero_idxs):
        if val:
            rgb[idx, 0] = V
            rgb[idx, 1] = V
            rgb[idx, 2] = V

    
    hh_array = np.copy(h_array)
    hh_array[hh_array >= 360.0] = 0.0
    hh_array /= 60.0
    i_array = hh_array.astype(np.long)
    ff_array = hh_array - i_array
    
    p_array = V * (1.0 - s_array)
    q_array = V * (1.0 - (s_array * ff_array))
    t_array = V * (1.0 - (s_array * (1.0 - ff_array)))
    
    p_array[p_array < 0.0] = 0.0
    p_array[p_array > 1.0] = 1.0
    q_array[q_array < 0.0] = 0.0
    q_array[q_array > 1.0] = 1.0
    t_array[t_array < 0.0] = 0.0
    t_array[t_array > 1.0] = 1.0



    for j in range(len(rgb)):
        if rgb[j][0] < 0.5: # Basically, if we haven't set it yet
            if i_array[j] == 0:
                rgb[j][0] = V
                rgb[j][1] = t_array[j]
                rgb[j][2] = p_array[j]
            elif i_array[j] == 1:
                rgb[j][0] = q_array[j]
                rgb[j][1] = V
                rgb[j][2] = p_array[j]
            elif i_array[j] == 2:
                rgb[j][0] = p_array[j]
                rgb[j][1] = V
                rgb[j][2] = t_array[j]
            elif i_array[j] == 3:
                rgb[j][0] = p_array[j]
                rgb[j][1] = q_array[j]
                rgb[j][2] = V
            elif i_array[j] == 4:
                rgb[j][0] = t_array[j]
                rgb[j][1] = p_array[j]
                rgb[j][2] = V
            else:
                rgb[j][0] = V
                rgb[j][1] = p_array[j]
                rgb[j][2] = q_array[j]

    return rgb



if __name__ == '__main__':
    use_affine = True
    #show_mode = FineMatches3DVisualizer.ColorMode.ANGLES
    show_mode = FineMatches3DVisualizer.ColorMode.DISTANCES
    fine_matches_fname = sys.argv[1]
    ts1_fname = sys.argv[2]
    ts2_fname = sys.argv[3]

    print("Parsing file: {}".format(fine_matches_fname))
    exists, inter_results_dal = IntermediateResultsDALPickle.load_prev_single_file_results(fine_matches_fname)

    ts1 = inter_results_dal["metadata"]['sec1']
    ts2 = inter_results_dal["metadata"]['sec2']
    print("ts1: {}, ts2: {}".format(ts1, ts2))

    sec1 = FineMatches3DVisualizer._load_tilespec(ts1_fname)
    sec2 = FineMatches3DVisualizer._load_tilespec(ts2_fname)

    figs = []
    for i, (sec_src, sec_dst) in enumerate([(sec1, sec2), (sec2, sec1)]):
        src_pts, dst_pts = inter_results_dal["contents"][i]
        visualizer = FineMatches3DVisualizer(sec_src, sec_dst, show_mode)
        visualizer.add_matches(src_pts, dst_pts)
        cur_fig, ax = visualizer.visualize_fine_matches_3d(use_affine=use_affine)
        figs.append(cur_fig)

    for fig in figs:
        plt.show(fig)

