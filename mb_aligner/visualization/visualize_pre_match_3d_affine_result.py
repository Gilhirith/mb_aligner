import numpy as np
import sys
import shapely.geometry
import shapely.ops
import ujson as json
from descartes.patch import PolygonPatch
import matplotlib
#matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from mb_aligner.common.intermediate_results_dal_pickle import IntermediateResultsDALPickle
from mb_aligner.dal.section import Section
from scipy.spatial import cKDTree as KDTree

class PreMatch3DAffineResultVisualizer(object):

    def __init__(self, scale=0.1):
        self._scale = 0.1
        self._sec1_color = '#6699cc'
        self._sec2_color = '#cc9966'


    @staticmethod
    def find_best_transformations(sec1, sec1_to_sec2_mfovs_transforms):
        new_sec1_to_sec2_mfovs_transforms = {}
        # find the nearest transformations for mfovs1 that are missing in sec1_to_sec2_mfovs_transforms and for sec2 to sec1
        mfovs1_centers = [[], []] # lists of mfovs indexes, mfovs centers
        missing_mfovs1_transforms_centers = [[], []] # lists of missing mfovs in sec1 and their centers
        for mfov1 in sec1.mfovs():
            mfov1_center = np.array([(mfov1.bbox[0] + mfov1.bbox[1])/2, (mfov1.bbox[2] + mfov1.bbox[3])/2])
            if mfov1.mfov_index in sec1_to_sec2_mfovs_transforms and sec1_to_sec2_mfovs_transforms[mfov1.mfov_index] is not None:
                mfovs1_centers[0].append(mfov1.mfov_index)
                mfovs1_centers[1].append(mfov1_center)
                new_sec1_to_sec2_mfovs_transforms[mfov1.mfov_index] = sec1_to_sec2_mfovs_transforms[mfov1.mfov_index]
            else:
                missing_mfovs1_transforms_centers[0].append(mfov1.mfov_index)
                missing_mfovs1_transforms_centers[1].append(mfov1_center)

        # estimate the transformation for mfovs in sec1 that do not have one (look at closest neighbor)
        if len(missing_mfovs1_transforms_centers[0]) > 0:
            mfovs1_centers_sec1_kdtree = KDTree(mfovs1_centers[1])
            mfovs1_missing_closest_centers_mfovs1_idxs = mfovs1_centers_sec1_kdtree.query(missing_mfovs1_transforms_centers[1])[1]
            for i, (mfov1_index, mfov1_closest_mfov_idx) in enumerate(zip(missing_mfovs1_transforms_centers[0], mfovs1_missing_closest_centers_mfovs1_idxs)):
                model = sec1_to_sec2_mfovs_transforms[
                            mfovs1_centers[0][mfov1_closest_mfov_idx]
                        ]
                new_sec1_to_sec2_mfovs_transforms[mfov1_index] = model
                print("Mfov {} from neighbor ({}) transformation\n{}".format(mfov1_index, mfovs1_centers[0][mfov1_closest_mfov_idx], model.to_str()))

        return new_sec1_to_sec2_mfovs_transforms, set(missing_mfovs1_transforms_centers[0])

    @staticmethod
    def create_section_tiles_projections(sec, best_transformations=None):
        min_x_proj = np.finfo(float).max
        min_y_proj = np.finfo(float).max
        mfovs_tiles_proj = defaultdict(list)
        for tile in sec.tiles():
            tile_mfov = tile.mfov_index

            bbox_pts = np.array([
                    [tile.bbox[0], tile.bbox[2]],
                    [tile.bbox[1], tile.bbox[2]],
                    [tile.bbox[1], tile.bbox[3]],
                    [tile.bbox[0], tile.bbox[3]]
                ], dtype=np.float64)
            if best_transformations is not None:
                bbox_pts = best_transformations[tile_mfov].apply(bbox_pts)

            mfovs_tiles_proj[tile_mfov].append(bbox_pts)
            min_x_proj = min(min_x_proj, np.min(bbox_pts[:, 0]))
            min_y_proj = min(min_y_proj, np.min(bbox_pts[:, 1]))

        return mfovs_tiles_proj, np.array([min_x_proj, min_y_proj])

    def _normalize_and_scale_projections(self, sec, mfovs_tiles_proj, min_xy_proj):
        for cur_mfov in sec.mfovs():
            cur_mfov_index = cur_mfov.mfov_index
            for idx in range(len(mfovs_tiles_proj[cur_mfov_index])):
                mfovs_tiles_proj[cur_mfov_index][idx] -= min_xy_proj
                mfovs_tiles_proj[cur_mfov_index][idx] *= self._scale
 
    @staticmethod
    def _add_polygons(ax, polygons_and_fill, facecolor=None):

        multi = shapely.geometry.MultiPolygon([p[0] for p in polygons_and_fill])

        for p, poly_pts_and_fill in zip(multi, polygons_and_fill):
            # plot coordinates system
            x, y = p.exterior.xy
            ax.plot(x, y, 'o', color='#999999', zorder=1)

            #patch = PolygonPatch(p, facecolor='#6699cc', edgecolor='#6699cc', alpha=0.5, zorder=2)
            #patch = PolygonPatch(p, edgecolor=edgecolor, alpha=0.5, zorder=2, fill=not poly_pts_and_fill[1])
            if facecolor is None:
                patch = PolygonPatch(p, edgecolor='#225511', alpha=0.2, zorder=2)
            else:
                patch = PolygonPatch(p, edgecolor='#225511', facecolor=facecolor, alpha=0.5, zorder=2)
            ax.add_patch(patch)

    @staticmethod
    def _add_mfovs_text(ax, sec_centers_proj, colors_map=None):
        for mfov_index, mfov_center in sec_centers_proj.items():
            color = 'green'
            if colors_map is not None:
                color = colors_map[mfov_index]
            ax.text(mfov_center[0], mfov_center[1], '{}'.format(mfov_index), color=color)
        

    @staticmethod
    def _get_pts_polygons(all_tiles_pts):
        polygons = [
            shapely.geometry.Polygon([
                t_pts[0],
                t_pts[1],
                t_pts[2],
                t_pts[3],
                t_pts[0]])
            for t_pts in all_tiles_pts]

        return polygons

    @staticmethod
    def _get_pts_unified_polygon(all_tiles_pts):
        polygons = PreMatch3DAffineResultVisualizer._get_pts_polygons(all_tiles_pts)
        return shapely.ops.cascaded_union(polygons)

    def _apply_transformation(self, matrix, p):
        pts = np.atleast_2d(p)
        return np.dot(matrix[:2,:2],
                       pts.T).T + np.asarray(matrix.T[2][:2]).reshape((1, 2))

    def _plot_polygons(self, polygons):

        fig = plt.figure()
        plt.gca().invert_yaxis()
        ax = fig.add_subplot(111)
        #multi = shapely.geometry.MultiPolygon([[p, []] for p in polygons])
        multi = shapely.geometry.MultiPolygon(polygons)

        for p in multi:
            # plot coordinates system
            x, y = p.exterior.xy
            ax.plot(x, y, 'o', color='#999999', zorder=1)

            #patch = PolygonPatch(p, facecolor='#6699cc', edgecolor='#6699cc', alpha=0.5, zorder=2)
            patch = PolygonPatch(p, edgecolor='#6699cc', alpha=0.5, zorder=2)
            ax.add_patch(patch)
        #polygons = [for ts in tilespecs]

        return fig, ax

    @staticmethod
    def _find_center(all_tiles_pts):
        # receives a list of a (tile) points list
        # find the minimial and maximal x and y values, and return the center of the mfov
        min_x = min([np.min(tile_pts[:, 0]) for tile_pts in all_tiles_pts])
        min_y = min([np.min(tile_pts[:, 1]) for tile_pts in all_tiles_pts])
        max_x = max([np.max(tile_pts[:, 0]) for tile_pts in all_tiles_pts])
        max_y = max([np.max(tile_pts[:, 1]) for tile_pts in all_tiles_pts])

        return np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0]).astype(np.int)

    def visualize_pre_match_3d(self, pre_match_results, sec1, sec2, title):

        # get the per-mfov transformation
        mfov_transformation = {k: v[0] for k, v in pre_match_results.items() if v[0] is not None} # mfov_num -> transformation_model

        for mfov_num in sorted(mfov_transformation.keys()):
            print("Mfov {} transformation\n{}".format(mfov_num, mfov_transformation[mfov_num].to_str()))
        # the original and projected images need to be normalized (start from the (0,0) coordinate)
        min_x_orig = np.finfo(float).max
        min_y_orig = np.finfo(float).max

        # Get the tilespecs boundaries for each mfov that we have a transformation for
        mfovs_tiles_orig = defaultdict(list)
        mfovs_tiles_proj = defaultdict(list)

        # find best transformations (even for missing mfovs transformations)
        best_transformations, missing_sec1_transforms = PreMatch3DAffineResultVisualizer.find_best_transformations(sec1, mfov_transformation)
        
        # compute the tiles projections (w/ transformations for sec1)
        sec1_mfovs_tiles_proj, sec1_min_xy_proj = PreMatch3DAffineResultVisualizer.create_section_tiles_projections(sec1, best_transformations)
        sec2_mfovs_tiles_proj, sec2_min_xy_proj = PreMatch3DAffineResultVisualizer.create_section_tiles_projections(sec2)

        min_xy_proj = np.array([min(sec1_min_xy_proj[0], sec2_min_xy_proj[0]), min(sec1_min_xy_proj[1], sec2_min_xy_proj[1])])

        # normalize the points of the original and projected images, and scale
        self._normalize_and_scale_projections(sec1, sec1_mfovs_tiles_proj, min_xy_proj)
        self._normalize_and_scale_projections(sec2, sec2_mfovs_tiles_proj, min_xy_proj)

        # Create the polygons for each of the mfovs for each of the sections
        sec1_polygons_and_fill = [(PreMatch3DAffineResultVisualizer._get_pts_unified_polygon(mfov_pts_list), not mfov_index in missing_sec1_transforms) for mfov_index, mfov_pts_list in sec1_mfovs_tiles_proj.items()]
        sec2_polygons_and_fill = [(PreMatch3DAffineResultVisualizer._get_pts_unified_polygon(mfov_pts_list), True) for mfov_pts_list in sec2_mfovs_tiles_proj.values()]
        
        # Create the figure and lay down the polygons for each mfov of each section
        fig = plt.figure()
        plt.gca().invert_yaxis()
        ax = fig.add_subplot(111)

        PreMatch3DAffineResultVisualizer._add_polygons(ax, sec1_polygons_and_fill, self._sec1_color)
        PreMatch3DAffineResultVisualizer._add_polygons(ax, sec2_polygons_and_fill, self._sec2_color)

        # Add mfov indices to the center of each relevant mfov
        # first, find normalized centers
        sec1_centers_proj = {mfov_index: PreMatch3DAffineResultVisualizer._find_center(mfov_pts_list) for mfov_index, mfov_pts_list in sec1_mfovs_tiles_proj.items()}
        sec2_centers_proj = {mfov_index: PreMatch3DAffineResultVisualizer._find_center(mfov_pts_list) for mfov_index, mfov_pts_list in sec2_mfovs_tiles_proj.items()}

        sec1_centers_colors_map = {mfov_index:'red' if mfov_index in mfov_transformation.keys() else 'black' for mfov_index in  sec1_centers_proj.keys()}
        sec2_centers_colors_map = {mfov_index:'green' for mfov_index in sec2_centers_proj.keys()}

        PreMatch3DAffineResultVisualizer._add_mfovs_text(ax, sec1_centers_proj, sec1_centers_colors_map)
        PreMatch3DAffineResultVisualizer._add_mfovs_text(ax, sec2_centers_proj, sec2_centers_colors_map)
        
        ax.set_title(title)

        return fig
 
       
        
    @staticmethod
    def _load_tilespec(fname):
        with open(fname, 'rb') as in_f:
            tilespec = json.load(in_f)
        return Section.create_from_tilespec(tilespec)

    def visualize_pre_match_3d_file(self, pre_matches_fname, ts1_fname, ts2_fname):
        """Given a 3d pre-match output file that has a per-mfov affine transformation,
           generates two figures, one of the outline of the mfovs of the first section (the original mfov outline),
           and the outline of the mfovs after transforming them to the second section
        """
        # Load the preliminary matches
        exists, self._inter_results_dal = IntermediateResultsDALPickle.load_prev_single_file_results(pre_matches_fname)

        if not exists:
            print("Error: file {} does not exist".format(pre_matches_fname))
            return None

        metadata = self._inter_results_dal['metadata']
        pre_match_results = self._inter_results_dal['contents']

        if pre_match_results is None or len(pre_match_results) == 0:
            print("Error: no matches in file {}".format(pre_matches_fname))
            return None

        # load tilespecs files
        ts1 = PreMatch3DAffineResultVisualizer._load_tilespec(ts1_fname)
        ts2 = PreMatch3DAffineResultVisualizer._load_tilespec(ts2_fname)

        title = "{} (blue) to {} (green)".format(metadata['sec1'], metadata['sec2'])
        return self.visualize_pre_match_3d(pre_match_results, ts1, ts2, title)

if __name__ == '__main__':
    pre_matches_fname = sys.argv[1]
    ts1_fname = sys.argv[2]
    ts2_fname = sys.argv[3]

    visualizer = PreMatch3DAffineResultVisualizer()
    fig = visualizer.visualize_pre_match_3d_file(pre_matches_fname, ts1_fname, ts2_fname)
    plt.show()

