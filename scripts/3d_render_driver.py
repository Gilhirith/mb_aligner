#
# Executes the rendering process on the 3d output transformations
# It takes a collection of tilespec files, each describing a montage of a single section,
# where all sections are aligned to a single coordinate system,
# and outputs a per-section image (or images/tiles).
# The input is a directory with tilespec files in json format (each file for a single layer).
# For tiles output, splits the image into tiles, and renders each in a separate process
#

import sys
import os.path
import os
import datetime
import time
from collections import defaultdict
import argparse
import glob
import ujson as json
#from rh_aligner.common.bounding_box import BoundingBox
import math
import tinyr
from rh_renderer.multiple_tiles_renderer import BlendType
import multiprocessing as mp
import render_core
import common





# class RenderSection(Job):
# #    def __init__(self, tiles_fname, output_fname, output_type, scale, from_x, from_y, to_x, to_y, invert_image, tile_size=None, from_to_cols_rows=None, hist_adjuster_fname=None, hist_adjuster_alg_type=None, threads_num=1):
#     def __init__(self, tiles_fname, output_fname, output_type, scale, from_x, from_y, to_x, to_y, invert_image, tile_size=None, from_to_cols_rows=None, hist_adjuster_alg_type=None, threads_num=1, blend_type='MULTI_BAND_SEAM'):
#         Job.__init__(self)
#         self.already_done = False
#         self.tiles_fname = '"{0}"'.format(tiles_fname)
#         self.output_fname = '"{0}"'.format(output_fname)
#         self.output_type = '--output_type "{0}"'.format(output_type)
#         self.scale = '--scale {0}'.format(scale)
#         self.from_x = '--from_x {0}'.format(from_x)
#         self.from_y = '--from_y {0}'.format(from_y)
#         self.to_x = '--to_x {0}'.format(to_x)
#         self.to_y = '--to_y {0}'.format(to_y)
#         if invert_image:
#             self.invert_image = '--invert_image'
#         else:
#             self.invert_image = ''
#         self.tile_size = ''
#         if tile_size is not None:
#             self.tile_size = '--tile_size {0}'.format(tile_size)
#         self.from_to_cols_rows = ''
#         if from_to_cols_rows is not None:
#             self.from_to_cols_rows = '--from_to_cols_rows {0}'.format(','.join([str(i) for i in from_to_cols_rows]))
# #         self.hist_adjuster_fname = ''
# #         if hist_adjuster_fname is not None:
# #             self.hist_adjuster_fname = '--hist_adjuster {0}'.format(hist_adjuster_fname)
#         self.hist_adjuster_alg_type = ''
#         if hist_adjuster_alg_type is not None:
#             self.hist_adjuster_alg_type = '--hist_adjuster_alg_type {0}'.format(hist_adjuster_alg_type)
#         self.empty_placeholder = '--empty_placeholder'
#         self.blend_type = '--blend_type {}'.format(blend_type)
#         self.dependencies = [ ]
#         #self.threads = threads_num
#         #self.threads_str = "-t {0}".format(threads_num)
#         self.memory = 4500
#         ##self.memory = 3500
#         #self.memory = 3000
#         #self.memory = 1500
#         #self.memory = 9000
#         #self.memory = 7000
#         #self.memory = 3000
#         #self.memory = 2700
#         #self.memory = 18000
#         #self.memory = 22000
#         #self.memory = 8000
#         #self.memory = 2000
#         self.time = 1400
#         self.output = output_fname
#         #self.max_sequential_jobs = 14
#         self.max_sequential_jobs = 1
#         #self.already_done = os.path.exists(self.output_file)
# 
#     def command(self):
#         return ['python -u',
#                 os.path.join(os.environ['RENDERER'], 'scripts', 'wrappers', 'render.py'),
#                 #self.output_type, self.scale, self.from_x, self.from_y, self.to_x, self.to_y, self.invert_image, self.tile_size, self.from_to_cols_rows, self.hist_adjuster_fname, self.hist_adjuster_alg_type, self.empty_placeholder, self.tiles_fname, self.output_fname]
#                 self.output_type, self.scale, self.from_x, self.from_y, self.to_x, self.to_y, self.invert_image, self.tile_size, self.from_to_cols_rows, self.hist_adjuster_alg_type, self.empty_placeholder, self.blend_type, self.tiles_fname, self.output_fname]
# 



def read_layer_from_filename(fname):
    num = int(os.path.basename(fname)[:4])
    return num

def binary_search_layer(files, ts_to_layer, target_layer_num):
    first = 0
    last = len(files) - 1
    found = False

    while first <= last and not found:
        midpoint = (first + last)//2
        mid_file = files[midpoint]
        if mid_file not in ts_to_layer:
            ts_to_layer[mid_file] = read_layer_from_filename(mid_file)
        mid_layer = ts_to_layer[mid_file]
        if mid_layer == target_layer_num:
            found = True
            return midpoint
        else:
            if target_layer_num < mid_layer:
                last = midpoint - 1
            else:
                first = midpoint + 1

    # The target layer was not found, returning the one before
    return first



def filter_files_layers(orig_files, from_layer, to_layer):
    filtered_files = sorted(orig_files)
    ts_to_layer = {}
    if from_layer != -1:
        # binary search for the initial layer
        ret_val = binary_search_layer(filtered_files, ts_to_layer, from_layer)
        # if ret_val is -1, need to use the first layer from the filtered_files
        if ret_val >= 0:
            
            if ts_to_layer[filtered_files[ret_val]] >= from_layer:
                filtered_files = filtered_files[ret_val:]
            else:
                filtered_files = filtered_files[ret_val+1:]

    if to_layer != -1:
        # binary search for the last layer
        ret_val = binary_search_layer(filtered_files, ts_to_layer, to_layer)
        # if ret_val is bigger than length of the files, need to use all the filtered files
        if ret_val < len(filtered_files):
            
            if ts_to_layer[filtered_files[ret_val]] <= to_layer:
                filtered_files = filtered_files[:ret_val]
            else:
                filtered_files = filtered_files[:ret_val+1]

    print("filtered_files: {}".format(filtered_files))
    return filtered_files



def find_relevant_tiles(in_fname, tile_size, from_x, from_y, to_x, to_y):
    relevant_tiles = set()

    # load the tilespec, and each of the tiles bboxes into an rtree
    with open(in_fname, 'r') as f:
        tilespecs = json.load(f)

    tiles_rtree = RTree()
    for ts in tilespecs:
        bbox = ts["bbox"]
        # pyrtree uses the (x_min, y_min, x_max, y_max) notation
        tiles_rtree.insert(ts, Rect(int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3])))

    # iterate through all the possible output tiles and include the ones that are overlapping with tiles from the tilespec
    for y_idx, cur_y in enumerate(range(from_y, to_y, tile_size)):
        for x_idx, cur_x in enumerate(range(from_x, to_x, tile_size)):
            rect_res = tiles_rtree.query_rect( Rect(cur_x, cur_y, cur_x + tile_size, cur_y + tile_size) )
            for rtree_node in rect_res:
                if not rtree_node.is_leaf():
                    continue
                # found a tilespec that is in the output tile
                relevant_tiles.add((y_idx + 1, x_idx + 1))

    return relevant_tiles


def map_hist_adjuster_files(filtered_files, hist_adjuster_dir):
    if hist_adjuster_dir is None:
        return None

    ret = []

    hist_adjuster_files = glob.glob(os.path.join(hist_adjuster_dir, '*.pkl'))
    hist_adjuster_files_map = { os.path.basename(fname):fname for fname in hist_adjuster_files }

    for ts_fname in filtered_files:
        matching_base_fname = os.path.basename(ts_fname).replace('.json', '.pkl') # change suffix to .pkl
        if matching_base_fname in hist_adjuster_files_map.keys():
            ret.append(hist_adjuster_files_map[matching_base_fname])
        else:
            matching_base_fname = '_'.join(matching_base_fname.split('_')[1:]) # remove the layer number
            if matching_base_fname in hist_adjuster_files_map.keys():
                ret.append(hist_adjuster_files_map[matching_base_fname])
            else:
                ret.append(None)
        print("Histogram adjuster file {} (for {})".format(ret[-1], os.path.basename(ts_fname)))
    
    return ret

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    #common.fs_create_dir(path)

###############################
# Driver
###############################
if __name__ == '__main__':



    # Command line parser
    parser = argparse.ArgumentParser(description='Renders a given set of images using the SLURM cluster commands.')
    parser.add_argument('tiles_dir', metavar='tiles_dir', type=str, 
                        help='a directory that contains a tile_spec files in json format')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='the directory where the rendered output files will be stored (default: ./output)',
                        default='./output')
    parser.add_argument('--scale', type=float,
                        help='set the scale of the rendered images (default: full image)',
                        default=1.0)
    parser.add_argument('--from_layer', type=int,
                        help='the layer to start from (inclusive, default: the first layer in the data)',
                        default=-1)
    parser.add_argument('--to_layer', type=int,
                        help='the last layer to render (inclusive, default: the last layer in the data)',
                        default=-1)
    parser.add_argument('--hop', type=int,
                        help='the number of sections to skip between the range [from_layer, to_layer) (default: 1 - no skipped sections)',
                        default=1)
    parser.add_argument('--from_x', type=int,
                        help='the left coordinate (default: 0)',
                        default=0)
    parser.add_argument('--from_y', type=int,
                        help='the top coordinate (default: 0)',
                        default=0)
    parser.add_argument('--to_x', type=int,
                        help='the right coordinate (default: full image)',
                        default=-1)
    parser.add_argument('--to_y', type=int,
                        help='the bottom coordinate (default: full image)',
                        default=-1)
    parser.add_argument('--tile_size', type=int,
                        help='the size (square side) of each tile (default: 0 - whole image)',
                        default=0)
    parser.add_argument('--output_type', type=str,
                        help='The output type format',
                        default='png')
    parser.add_argument('-i', '--invert_image', action='store_true',
                        help='store an inverted image')
    parser.add_argument('--relevant_tiles_only', action='store_true',
                        help='if set, and a tile_size is specified, only stores tiles that are overlapping with the tilespec of each section')
    parser.add_argument('--bbox', type=str,
                        help='A pre-computed bbox if the entire dataset (optional, if not supplied will be calculated. Format: x_min,x_max,y_min,y_max)',
                        default=None)
#     parser.add_argument('--hist_adjuster_dir', type=str,
#                         help='A location of a directory with pkl files (each file name is the suffix of a tilespec json fname) that contain the histogram adjuster objects (default: None)',
#                         default=None)
    parser.add_argument('--hist_adjuster_alg_type', type=str,
                        help='the type of algorithm to use for a general per-tile histogram normalization. Supported types: CLAHE, GB11CLAHE (gaussian blur (11,11) and then clahe) (default: None)',
                        default=None)
    parser.add_argument('--per_job_cols_rows', type=str,
                        help='Only used when tile_size is set. The maximal number of columns and rows for a single job in the cluster, of the from "cols_num,rows_num".\
If not set, each job will render a single tile (default: None)',
                        default=None)
    #parser.add_argument('-s', '--skip_layers', type=str, 
    #                    help='the range of layers (sections) that will not be processed e.g., "2,3,9-11,18" (default: no skipped sections)',
    #                    default=None)
    parser.add_argument('--blend_type', type=str,
                        help='The type of blending to use. Values = {} (default: MULTI_BAND_SEAM)'.format(BlendType.__members__.keys()),
                        default='MULTI_BAND_SEAM')
    parser.add_argument('-p', '--processes_num', type=int,
                        help='the number of processes to use (default: 1)',
                        default=1)
 

    args = parser.parse_args() 

    #assert 'RENDERER' in os.environ
    #assert 'VIRTUAL_ENV' in os.environ


    if args.relevant_tiles_only:
        assert args.tile_size > 0
        assert args.per_job_cols_rows is None, "Not supported at the moment"

    if args.per_job_cols_rows is not None:
        assert args.tile_size > 0

    create_dir(args.output_dir)


    #all_files = glob.glob(os.path.join(args.tiles_dir, '*.json'))
    all_files = common.get_ts_files(args.tiles_dir)
    if len(all_files) == 0:
        print("No json files to render, quitting")
        sys.exit(1)

    args.blend_type = BlendType[args.blend_type]

    # Create the pool of processes
    pool = mp.Pool(processes=args.processes_num)

    print("Computing entire 3d volume bounding box (including filtered due to layer# restrictions)")
    # Compute the width and height of the entire 3d volume
    # so all images will have the same dimensions (needed for many image viewing applications, e.g., Fiji)
    if args.bbox is None:
        entire_image_bbox = common.read_bboxes_grep_pool(all_files, pool)
#         entire_image_bbox = BoundingBox.read_bbox_grep(all_files[0])
#         for in_fname in all_files[1:]:
#             entire_image_bbox.extend(BoundingBox.read_bbox_grep(in_fname))
    else:
        entire_image_bbox = [float(f) for f in args.bbox.split(',')]
    print("Final bbox for the 3d image: {}".format(entire_image_bbox))

    # Set the boundaries according to the entire_image_bbox
    if args.from_x == 0:
        args.from_x = int(math.floor(entire_image_bbox[0]))
    if args.from_y == 0:
        args.from_y = int(math.floor(entire_image_bbox[2]))
    if args.to_x == -1:
        args.to_x = int(math.ceil(entire_image_bbox[1]))
    if args.to_y == -1:
        args.to_y = int(math.ceil(entire_image_bbox[3]))

    # Set the max_col and max_row (in case of rendering tiles)
    max_col = 0
    max_row = 0
    actual_tile_size = 0 # pre-scaled version of the tile size
    if not args.tile_size == 0:
        actual_tile_size = int(math.ceil(args.tile_size / args.scale))
        max_col = int(math.ceil((args.to_x - args.from_x) / float(actual_tile_size)))
        max_row = int(math.ceil((args.to_y - args.from_y) / float(actual_tile_size)))

    all_running_jobs = []


    filtered_files = filter_files_layers(all_files, args.from_layer, args.to_layer)

    filtered_files = [filtered_files[i] for i in range(0, len(filtered_files), args.hop)]

    #hist_adjuster_list = map_hist_adjuster_files(filtered_files, args.hist_adjuster_dir)

    # Render each section to fit the out_shape
    rendering_jobs = []
    for in_fname_idx, in_fname in enumerate(filtered_files):
        print("Processing section {}".format(in_fname))
        if args.tile_size == 0:
            # Single image (no tiles)
            out_fname_prefix = os.path.splitext(os.path.join(args.output_dir, os.path.basename(in_fname)))[0]
            out_fname = "{}.{}".format(out_fname_prefix, args.output_type)
            out_fname_empty = "{}_empty".format(out_fname)

            if not os.path.exists(out_fname) and not os.path.exists(out_fname_empty):
                print("Rendering section {}".format(in_fname))
#                 hist_adjuster_fname = None
#                 if hist_adjuster_list is not None:
#                     hist_adjuster_fname = hist_adjuster_list[in_fname_idx]
                #job_render = RenderSection(in_fname, out_fname, args.output_type, args.scale, args.from_x, args.from_y, args.to_x, args.to_y, args.invert_image, hist_adjuster_fname=hist_adjuster_fname, hist_adjuster_alg_type=args.hist_adjuster_alg_type, threads_num=1)
                job_render = pool.apply_async(render_core.render_tilespec, (in_fname, out_fname, args.scale, args.output_type, [args.from_x, args.to_x, args.from_y, args.to_y], args.tile_size, args.invert_image), dict(hist_adjuster_alg_type=args.hist_adjuster_alg_type, blend_type=args.blend_type))
                rendering_jobs.append(job_render)
        else:
            if args.per_job_cols_rows is not None:
                # Use multple cols and rows per job
                block_cols_step, block_rows_step = [int(i) for i in args.per_job_cols_rows.split(',')]

                tiles_out_dir = os.path.splitext(os.path.join(args.output_dir, os.path.basename(in_fname)))[0]
                create_dir(tiles_out_dir)
                out_fname_prefix = os.path.splitext(os.path.join(tiles_out_dir, os.path.basename(in_fname)))[0]

                # Iterate over each block of rows and columns and save the tiles
                for cur_row in range(0, max_row, block_rows_step):
                    from_row = cur_row
                    to_row = min(cur_row + block_rows_step, max_row)
                    #from_y = args.from_y + cur_row * actual_tile_size
                    #next_row = min(cur_row + block_row_step, max_row)
                    #to_y = min(args.from_y + next_row * actual_tile_size, args.to_y)
                    for cur_col in range(0, max_col, block_cols_step):

                        from_col = cur_col
                        to_col = min(cur_col + block_cols_step, max_col)

                        out_fname = "{}_tr{}-tc{}.{}".format(out_fname_prefix, cur_row + 1, cur_col + 1, args.output_type)
                        out_fname_empty = "{}_empty".format(out_fname)
                        last_tile_fname = "{}_tr{}-tc{}.{}".format(out_fname_prefix, to_row, to_col, args.output_type) # 1-based numbering, and the tow_row/to_col are always the end of the block
                        last_tile_fname_empty = "{}_empty".format(last_tile_fname)

                        if not os.path.exists(last_tile_fname) and not os.path.exists(last_tile_fname_empty):
                            print("Rendering section {}, from tr{}-tc{} to tr{}-tc{}".format(in_fname, cur_row + 1, cur_col + 1, to_row, to_col))
#                             hist_adjuster_fname = None
#                             if hist_adjuster_list is not None:
#                                 hist_adjuster_fname = hist_adjuster_list[in_fname_idx]

                            # Render the tiles
                            #job_render = RenderSection(in_fname, out_fname_prefix, args.output_type, args.scale, args.from_x, args.from_y, args.to_x, args.to_y, args.invert_image, tile_size=args.tile_size, from_to_cols_rows=(from_col, from_row, to_col, to_row), hist_adjuster_fname=hist_adjuster_fname, hist_adjuster_alg_type=args.hist_adjuster_alg_type, threads_num=1)
                            job_render = pool.apply_async(render_core.render_tilespec, (in_fname, out_fname_prefix, args.scale, args.output_type, [args.from_x, args.to_x, args.from_y, args.to_y], args.tile_size, args.invert_image), dict(hist_adjuster_alg_type=args.hist_adjuster_alg_type, from_to_cols_rows=(from_col, from_row, to_col, to_row), blend_type=args.blend_type))
                            rendering_jobs.append(job_render)


            else:
                # A single tile per job, so just let each job crop its target area
                if args.relevant_tiles_only:
                    relevant_tiles_set = find_relevant_tiles(in_fname, args.tile_size, args.from_x, args.from_y, args.to_x, args.to_y)
                # Multiple tiles should be at the output (a single directory per section)
                tiles_out_dir = os.path.splitext(os.path.join(args.output_dir, os.path.basename(in_fname)))[0]
                create_dir(tiles_out_dir)
                out_fname_prefix = os.path.splitext(os.path.join(tiles_out_dir, os.path.basename(in_fname)))[0]

                # Iterate over each row and column and save the tile
                for cur_row in range(max_row):
                    from_y = args.from_y + cur_row * actual_tile_size
                    to_y = min(args.from_y + (cur_row + 1) * actual_tile_size, args.to_y)
                    for cur_col in range(max_col):
                        if args.relevant_tiles_only:
                            if (cur_row + 1, cur_col + 1) not in relevant_tiles_set:
                                print("Not rendering section {}, tr{}-tc{}, becuase output tile doesn't overlap with tilespec tiles".format(in_fname, cur_row + 1, cur_col + 1))
                                continue

                        tile_start_time = time.time()
                        out_fname = "{}_tr{}-tc{}.{}".format(out_fname_prefix, cur_row + 1, cur_col + 1, args.output_type)
                        out_fname_empty = "{}_empty".format(out_fname)

                        if not os.path.exists(out_fname) and not os.path.exists(out_fname_empty):
                            print("Rendering section {}, tr{}-tc{}".format(in_fname, cur_row + 1, cur_col + 1))
                            from_x = args.from_x + cur_col * actual_tile_size
                            to_x = min(args.from_x + (cur_col + 1) * actual_tile_size, args.to_x)

#                             hist_adjuster_fname = None
#                             if hist_adjuster_list is not None:
#                                 hist_adjuster_fname = hist_adjuster_list[in_fname_idx]

                            # Render the tile
                            #job_render = RenderSection(in_fname, out_fname, args.output_type, args.scale, from_x, from_y, to_x, to_y, args.invert_image, hist_adjuster_fname=hist_adjuster_fname, hist_adjuster_alg_type=args.hist_adjuster_alg_type, threads_num=1)
                            job_render = pool.apply_async(render_core.render_tilespec, (in_fname, out_fname, args.scale, args.output_type, [from_x, to_x, from_y, to_y], args.tile_size, args.invert_image), dict(hist_adjuster_alg_type=args.hist_adjuster_alg_type, blend_type=args.blend_type))
                            rendering_jobs.append(job_render)


    for job_render in rendering_jobs:
        job_render.get()

    print("All jobs finished, shutting down...")

    pool.close()
    pool.join()

