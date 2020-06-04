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
import rh_img_access_layer







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
    # if not os.path.exists(path):
    #    os.makedirs(path)
    common.fs_create_dir(path)

def get_mfov_tile_bbox(ts_fname, mfov_tile, halo_size):
    # load the tilespec, and find the right mfov tile
    with rh_img_access_layer.FSAccess(ts_fname, False) as data:
        tilespecs = json.load(data)

    for tile_ts in tilespecs:
        if tile_ts["mfov"] == mfov_tile[0] and tile_ts["tile_index"] == mfov_tile[1]:
            bbox = tile_ts["bbox"]
            from_x = bbox[0] - halo_size
            from_y = bbox[2] - halo_size
            to_x = bbox[1] + halo_size
            to_y = bbox[3] + halo_size
            return from_x, from_y, to_x, to_y
    raise Exception("Couldn't find mfov_tile: {} in section: {}".format(mfov_tile, ts_fname))


###############################
# Driver
###############################
if __name__ == '__main__':



    # Command line parser
    parser = argparse.ArgumentParser(description='Renders a given tile in a given section.')
    parser.add_argument('ts_fname', metavar='ts_fname', type=str, 
                        help='the tilespec file name to render')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='the directory where the rendered output files will be stored (default: ./output)',
                        default='./output')
    parser.add_argument('--scale', type=float,
                        help='set the scale of the rendered images (default: full image)',
                        default=1.0)
    parser.add_argument('--mfov_tile', type=str,
                        help='the mfov,tile indexes to render, e.g.: "30,61" for tile 61 in mfov 30 (default: None)',
                        default=None)
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



    if args.per_job_cols_rows is not None:
        assert args.tile_size > 0

    if args.mfov_tile is not None:
        assert(args.from_x == 0 and args.from_y == 0 and args.to_x == -1 and args.to_y == -1)

    create_dir(args.output_dir)


    #all_files = glob.glob(os.path.join(args.tiles_dir, '*.json'))
    ts_fname = args.ts_fname

    args.blend_type = BlendType[args.blend_type]

    # Create the pool of processes
    pool = mp.Pool(processes=args.processes_num)

    # Compute the width and height of the entire 3d volume
    # so all images will have the same dimensions (needed for many image viewing applications, e.g., Fiji)
    if args.bbox is None:
        entire_image_bbox = common.read_bboxes_grep_pool([args.ts_fname], pool)
#         entire_image_bbox = BoundingBox.read_bbox_grep(all_files[0])
#         for in_fname in all_files[1:]:
#             entire_image_bbox.extend(BoundingBox.read_bbox_grep(in_fname))
    else:
        entire_image_bbox = [float(f) for f in args.bbox.split(',')]
    print("Final bbox for the 2d image: {}".format(entire_image_bbox))

    # Set the boundaries according to the entire_image_bbox
    if args.mfov_tile is not None:
        mfov_tile = tuple([int(x) for x in args.mfov_tile.split(',')])
        assert(len(mfov_tile) == 2)
        args.from_x, args.from_y, args.to_x, args.to_y = get_mfov_tile_bbox(args.ts_fname, mfov_tile, 1500)
        print("mfov_tile bbox for the 2d image: {}".format(args.from_x, args.from_y, args.to_x, args.to_y))
    else:
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



    filtered_files = [args.ts_fname]

    #hist_adjuster_list = map_hist_adjuster_files(filtered_files, args.hist_adjuster_dir)

    # Render each section to fit the out_shape
    rendering_jobs = []
    for in_fname_idx, in_fname in enumerate(filtered_files):
        print("Processing section {}".format(in_fname))
        if args.tile_size == 0:
            # Single image (no tiles)
            out_fname_prefix = os.path.splitext(os.path.join(args.output_dir, os.path.basename(in_fname)))[0]
            if args.mfov_tile is not None:
                out_fname_prefix = "{}_m{}_t{}".format(out_fname_prefix, mfov_tile[0], mfov_tile[1])
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
                if args.mfov_tile is not None:
                    out_fname_prefix = "{}_m{}_t{}".format(out_fname_prefix, mfov_tile[0], mfov_tile[1])

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
#                 if args.relevant_tiles_only:
#                     relevant_tiles_set = find_relevant_tiles(in_fname, args.tile_size, args.from_x, args.from_y, args.to_x, args.to_y)
                # Multiple tiles should be at the output (a single directory per section)
                tiles_out_dir = os.path.splitext(os.path.join(args.output_dir, os.path.basename(in_fname)))[0]
                create_dir(tiles_out_dir)
                out_fname_prefix = os.path.splitext(os.path.join(tiles_out_dir, os.path.basename(in_fname)))[0]
                if args.mfov_tile is not None:
                    out_fname_prefix = "{}_m{}_t{}".format(out_fname_prefix, mfov_tile[0], mfov_tile[1])

                # Iterate over each row and column and save the tile
                for cur_row in range(max_row):
                    from_y = args.from_y + cur_row * actual_tile_size
                    to_y = min(args.from_y + (cur_row + 1) * actual_tile_size, args.to_y)
                    for cur_col in range(max_col):
#                         if args.relevant_tiles_only:
#                             if (cur_row + 1, cur_col + 1) not in relevant_tiles_set:
#                                 print("Not rendering section {}, tr{}-tc{}, becuase output tile doesn't overlap with tilespec tiles".format(in_fname, cur_row + 1, cur_col + 1))
#                                 continue

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

