import cv2
import numpy as np
from scipy.spatial import Delaunay
from mb_aligner.common import utils
from mb_aligner.common.wrinkle_avoidance.wrinkle_detector import WrinkleDetector
from mb_aligner.common.wrinkle_avoidance.edges_filter import EdgesFilter

class SectionEdgesFilter(object):

    def __init__(self, section, **kwargs):


        self._edges_filter = EdgesFilter()
        self._section = section

        wrinkle_detector_params = kwargs.get("wrinkle_detector_params", {})        
        #self._wrinkle_detector = WrinkleDetector(**wrinkle_detector_params)
        self._wrinkle_detector = WrinkleDetector(kernel_size=3, threshold=200, min_line_length=100)

        self._detect_wrinkles()


    def _detect_wrinkles(self):
        # for each image of the section detect the wrinkles
        # TODO - parallelize
        per_image_contours = [self._wrinkle_detector.detect(tile.image) for tile in section.tiles()]
        # apply the per-image rigid transformation on all the contours
        per_image_contours = [[] if len(per_image_contours) == 0 else SectionMeshEdgesFilter._apply_tile_transform(tile, contours) for tile, contours in zip(section.tiles(), per_image_contours)]


        # unify the contours
        #all_image_contours = np.vstack(per_image_contours)
        all_image_contours = []
        for image_contours in per_image_contours:
            all_image_contours.extend(image_contours)

        self._edges_filter.set_contours(all_image_contours)


    @staticmethod
    def _apply_tile_transform(tile, contours):
        # assuming there's a single transform for the tile
        assert(len(tile.transforms) == 1)

        transform = tile.transforms[0]

        out_contours = np.empty_like(contours)
        for cnt_idx, cnt in enumerate(contours):
            cnt_pts = cnt.reshape((len(cnt), 2))
            cnt_pts_transformed = transform.apply(cnt_pts)
            cnt_pts_transformed = cnt_pts_transformed.reshape((len(cnt_pts_transformed), 1, 2))
            out_contours[cnt_idx] = cnt_pts_transformed
        return out_contours

    def filter_mesh_edges_and_simplices(self, section_mesh_tri):
        
        filtered_edges_indices, filtered_simplices, orig_simplices_mask = edges_filter.filter_mesh_by_wrinkles(section_mesh_tri)

        #visualize_filtered_mesh(mesh_tri, filtered_edges_indices, filtered_simplices, img)
        return filtered_edges_indices, filtered_simplices, orig_simplices_mask

    def filter_edges(self, edges):
        filtered_edges_mask = edges_filter.filter_edges_by_wrinkles(edges)
        return ~filtered_edges_mask



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

    edges_filter = SectionMeshEdgesFilter(500)

    filtered_edges_indices, filtered_simplices = edges_filter.filter_edges_and_simplices(section)

    die
    

