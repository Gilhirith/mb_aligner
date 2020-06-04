import cv2
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from mb_aligner.common import utils
from mb_aligner.common.wrinkle_avoidance.wrinkle_detector import WrinkleDetector
from mb_aligner.common.wrinkle_avoidance.edges_filter import EdgesFilter
from collections import Counter

DEBUG_NO_TRANSFOM = True

class SectionMeshRefiner(object):

    def __init__(self, section, mesh_spacing, refined_mesh_spacing, **kwargs):

        self._mesh_spacing = mesh_spacing
        self._refined_mesh_spacing = refined_mesh_spacing

        # create all the points of the original mesh
        self._orig_mesh_pts = utils.generate_hexagonal_grid(section.bbox, self._mesh_spacing)
        if DEBUG_NO_TRANSFOM:
            self._orig_mesh_pts = utils.generate_hexagonal_grid([0, 3128, 0, 2724], self._mesh_spacing)

        # detect all wrinkles
        self._edges_filter = EdgesFilter()
        self._section = section

        wrinkle_detector_params = kwargs.get("wrinkle_detector_params", {})        
        #self._wrinkle_detector = WrinkleDetector(**wrinkle_detector_params)
        self._wrinkle_detector = WrinkleDetector(kernel_size=3, threshold=200, min_line_length=100)

        self._detect_wrinkles()


        # refine the mesh around the wrinkles
        self._new_points_lists = self._refine_wrinkles_mesh()

        self._refined_mesh_points = self._merge_points()


    def _detect_wrinkles(self):
        # for each image of the section detect the wrinkles
        # TODO - parallelize
        per_image_contours = [self._wrinkle_detector.detect(tile.image) for tile in self._section.tiles()]
        # apply the per-image rigid transformation on all the contours
        per_image_contours = [[] if len(per_image_contours) == 0 else SectionMeshRefiner._apply_tile_transform(tile, contours) for tile, contours in zip(self._section.tiles(), per_image_contours)]


        # unify the contours
        #all_image_contours = np.vstack(per_image_contours)
        self._all_image_contours = []
        for image_contours in per_image_contours:
            self._all_image_contours.extend(image_contours)

        self._edges_filter.set_contours(self._all_image_contours)

    def _refine_wrinkles_mesh(self):
        # add points in the bounding box of each wrinkle
        new_points_lists = []
#         for image_contours in self._per_image_contours:
#             if len(image_contours) == 0:
#                 continue
            
        for image_contours in self._all_image_contours:
            wrinkle_min_xy = np.min(image_contours, axis=0)[0]
            wrinkle_max_xy = np.max(image_contours, axis=0)[0]
            wrinkle_bbox = (wrinkle_min_xy[0], wrinkle_max_xy[0], wrinkle_min_xy[1], wrinkle_max_xy[1])
            cur_bbox_points = utils.generate_hexagonal_grid_interior(wrinkle_bbox, self._refined_mesh_spacing)
            if len(cur_bbox_points) > 0:
                print("Created new points around bbox:", wrinkle_bbox)
                new_points_lists.append(cur_bbox_points)
        return new_points_lists

    def _merge_points(self):
        if len(self._new_points_lists) == 0:
            return self._orig_mesh_pts

        # find all points that are too close to each other, and mask them out
        
        proposed_pts = np.vstack((self._orig_mesh_pts, *self._new_points_lists))
        proposed_pts_inv_mask = np.zeros((len(proposed_pts), ), dtype=np.bool)
        proposed_pts_kdtree = cKDTree(proposed_pts)

        
        bad_pairs = proposed_pts_kdtree.query_pairs(self._refined_mesh_spacing / 2, output_type='ndarray')
        to_remove_idxs = set()
        # create a graph from the bad pairs, and greedily remove nodes that have the largest number of 
        
        # create histogram from bad_pairs
        def _compute_bad_pairs_histogram(bad_pairs, to_remove_idxs):
            hist = Counter()
            for pair in bad_pairs:
                if pair[0] in to_remove_idxs or pair[1] in to_remove_idxs:
                    continue
                hist[pair[0]] += 1
                hist[pair[1]] += 1
            
            #print("bad_pairs:", hist)
            return hist

        bad_pairs_hist = _compute_bad_pairs_histogram(bad_pairs, to_remove_idxs)
        if len(bad_pairs_hist) == 0:
            return proposed_pts

        cur_most_common = bad_pairs_hist.most_common(1)[0]
        while cur_most_common[1] > 0:
            #print("adding", cur_most_common[0])
            to_remove_idxs.add(cur_most_common[0])
            bad_pairs_hist = _compute_bad_pairs_histogram(bad_pairs, to_remove_idxs)
            if len(bad_pairs_hist) == 0:
                break
            cur_most_common = bad_pairs_hist.most_common(1)[0]

        #print("to_remove_idxs:", to_remove_idxs)
        proposed_pts_inv_mask[list(to_remove_idxs)] = True
        return proposed_pts[~proposed_pts_inv_mask]

    @staticmethod
    def _apply_tile_transform(tile, contours):
        # assuming there's a single transform for the tile
        assert(len(tile.transforms) == 1)

        transform = tile.transforms[0]
        if DEBUG_NO_TRANSFOM:
            from rh_renderer import models
            transform = models.AffineModel()

        out_contours = np.empty_like(contours)
        for cnt_idx, cnt in enumerate(contours):
            cnt_pts = cnt.reshape((len(cnt), 2))
            cnt_pts_transformed = transform.apply(cnt_pts)
            cnt_pts_transformed = cnt_pts_transformed.reshape((len(cnt_pts_transformed), 1, 2))
            out_contours[cnt_idx] = cnt_pts_transformed
        return out_contours

    def get_refined_mesh_points(self):
        return self._refined_mesh_points

    def filter_mesh_edges_and_simplices(self, section_mesh_tri):
        
        filtered_edges_indices, filtered_simplices, orig_simplices_mask = edges_filter.filter_mesh_by_wrinkles(section_mesh_tri)

        #visualize_filtered_mesh(mesh_tri, filtered_edges_indices, filtered_simplices, img)
        return filtered_edges_indices, filtered_simplices, orig_simplices_mask

    def filter_edges(self, edges):
        filtered_edges_mask = self._edges_filter.filter_edges_by_wrinkles(edges)
        return ~filtered_edges_mask

    def get_min_dist_from_wrinkles(self, pts):
        return np.array(self._edges_filter.get_min_lengths(pts))



def visualize_refined_mesh(mesh_tri, filtered_simplices, img):
    import matplotlib.pyplot as plt

    if img is not None:
        plt.imshow(255-img, cmap='gray')

    points = mesh_tri.points
    # find unique edges
    edge_indices = np.vstack((filtered_simplices[:, :2],
                              filtered_simplices[:, 1:],
                              filtered_simplices[:, [0, 2]]))
    edge_indices = np.vstack({tuple(sorted(tuple(row))) for row in edge_indices}).astype(np.uint32)

    filtered_edges_pts = points[edge_indices]
    plt.triplot(points[:,0], points[:,1], filtered_simplices, color='r')
    plt.plot(points[:,0], points[:,1], 'ro', markersize=2)
    #plt.plot(filtered_edges_pts[:, 0], filtered_edges_pts[:, 1], 'ro-', linewidth=2, markersize=2)

#    plt.triplot(points[:,0], points[:,1], mesh_tri.simplices.copy())
#    plt.plot(points[:,0], points[:,1], 'o')
    #plt.savefig('out_mesh_refined.png', dpi=1000)
    plt.show()

if __name__ == '__main__':
    import sys
    from mb_aligner.dal.section import Section
    import ujson
    import os

    if len(sys.argv) > 1:
        ts_fname = sys.argv[1]
    else:
        ts_fname = '/n/boslfs/LABS/lichtman_lab/adisuis/alignments/Zebrafish_Mariela_HindBrainROI/2d_W19_single_tiles_HBROI_gpu_opt_output_dir/W19_Sec016_montaged.json'
        #img_fname = '/n/lichtmangpfs01/Instrument_drop/U19_Zebrafish/EM/w019/w019_h02_20190326_00-43-52/002_S16R1/000021/002_000021_040_2019-03-26T0046443382649.bmp'

    
    with open(ts_fname, 'rt') as in_f:
        tilespec = ujson.load(in_f)

    wafer_num = int(os.path.basename(ts_fname).split('_')[0].split('W')[1])
    sec_num = int(os.path.basename(ts_fname).split('.')[0].split('_')[1].split('Sec')[1])
    section = Section.create_from_tilespec(tilespec, wafer_section=(wafer_num, sec_num))

    mesh_spacing = 500
    refined_mesh_spacing = 50
    mesh_refiner = SectionMeshRefiner(section, mesh_spacing, refined_mesh_spacing)

    mesh_tri = Delaunay(mesh_refiner.get_refined_mesh_points())
    #edges_filter = MeshEdgesFilter(mesh_tri)
    #filtered_edges_indices, filtered_simplices = edges_filter.filter_by_wrinkles(contours)

    filtered_simplices = mesh_tri.simplices

    assert(len(section.tilespec) == 1)
    img = None
    if DEBUG_NO_TRANSFOM:
        img = list(section.tiles())[0].image
    visualize_refined_mesh(mesh_tri, filtered_simplices, img)


    die
    

