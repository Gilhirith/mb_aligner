import sys
import os
import glob
import argparse
#from ..common.bounding_box import BoundingBox
import subprocess
import json
from rh_logger.api import logger
import rh_logger
import logging
import multiprocessing as mp
import numpy as np

def add_transformation(in_file, out_file, transform, deltas):
    # load the current json file
    try:
        with open(in_file, 'r') as f:
            data = json.load(f)
    except:
        logger.report_event("Error when reading {} - Exiting".format(in_file), log_level=logging.ERROR)
        raise

    if deltas[0] != 0.0 and deltas[1] != 0.0:
        for tile in data:
            # Update the transformation
            if "transforms" not in tile.keys():
                tile["transforms"] = []
            tile["transforms"].append(transform)

            # Update the bbox
            if "bbox" in tile.keys():
                bbox = tile["bbox"]
                bbox_new = [bbox[0] - deltas[0], bbox[1] - deltas[0], bbox[2] - deltas[1], bbox[3] - deltas[1]]
                tile["bbox"] = bbox_new

    with open(out_file, 'w') as f:
        json.dump(data, f, indent=4)
 
def read_minxy_grep(tiles_spec_fname):
    cmd = "grep -A 5 \"bbox\" {}".format(tiles_spec_fname)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Output will be of the following format:
    """
        "bbox": [
            13262.416015625,
            16390.580476722345,
            10470.920522462373,
            13195.109375
        ],
--
        "bbox": [
            16273.984836771779,
            19402.12109375,
            10470.3515625,
            13194.508027928736
        ],
--
...
    """

    # Parse all bounding boxes in the given json file
    lines = p.stdout.readlines()
    min_x = np.min([float(line.decode('utf-8').strip(' ,\n')) for line in lines[1::7]])
    #max_x = np.max([float(line.decode('utf-8').strip(' ,\n')) for line in lines[2::7]])
    min_y = np.min([float(line.decode('utf-8').strip(' ,\n')) for line in lines[3::7]])
    #max_y = np.max([float(line.decode('utf-8').strip(' ,\n')) for line in lines[4::7]])
    #return np.array([min_x, max_x, min_y, max_y])
    return [min_x, min_y]

#     ret_val = None
#     cur_bbox_lines = []
#     for line in iter(p.stdout.readline, ''):
#         if line.startswith("--"):
#             cur_bbox = BoundingBox.fromStr(BoundingBox.parse_bbox_lines(cur_bbox_lines))
#             if ret_val is None:
#                 ret_val = cur_bbox
#             else:
#                 ret_val.extend(cur_bbox)
#             cur_bbox_lines = []
#         else:
#             cur_bbox_lines.append(line.strip(' \n'))
#     if len(cur_bbox_lines) > 0:
#         cur_bbox = BoundingBox.fromStr(BoundingBox.parse_bbox_lines(cur_bbox_lines))
#         if ret_val is None:
#             ret_val = cur_bbox
#         else:
#             ret_val.extend(cur_bbox)
#     return ret_val

def normalize_coordinates(tile_fnames_or_dir, output_dir, pool):
    # Get all the files that need to be normalized
    all_files = []

    logger.report_event("Reading {}".format(tile_fnames_or_dir), log_level=logging.INFO)
    for file_or_dir in tile_fnames_or_dir:
        if not os.path.exists(file_or_dir):
            logger.report_event("{0} does not exist (file/directory), skipping".format(file_or_dir), log_level=logging.WARN)
            continue

        if os.path.isdir(file_or_dir):
            actual_dir_files = glob.glob(os.path.join(file_or_dir, '*.json'))
            all_files.extend(actual_dir_files)
        else:
            all_files.append(file_or_dir)

    if len(all_files) == 0:
        logger.report_event("No files for normalization found.", log_level=logging.ERROR)
        raise Exception("No files for normalization found.")

    logger.report_event("Normalizing coordinates of {} files".format(len(all_files)), log_level=logging.INFO)

    # Retrieve the bounding box of these files
    entire_image_bbox = None
    
    # merge the bounding boxes to a single bbox
    if len(all_files) > 0:
        sections_min_xys = np.array(pool.map(read_minxy_grep, all_files))
#         entire_image_bbox = np.array([
#                                 np.min(sections_bboxes[:, 0]), np.max(sections_bboxes[:, 1]),
#                                 np.min(sections_bboxes[:, 2]), np.max(sections_bboxes[:, 3])
#                             ])
    

    # Set the translation transformation
    #deltaX = entire_image_bbox[0]
    #deltaY = entire_image_bbox[2]
    deltaX = np.min(sections_min_xys[:, 0])
    deltaY = np.min(sections_min_xys[:, 1])

    logger.report_event("Entire 3D image min x,y: {}, {}".format(deltaX, deltaY), log_level=logging.INFO)

    # TODO - use models to create the transformation
    transform = {
            "className" : "mpicbg.trakem2.transform.TranslationModel2D",
            "dataString" : "{0} {1}".format(-deltaX, -deltaY)
        }

    # Add the transformation to each tile in each tilespec
#     out_files = [os.path.join(output_dir, os.path.basename(in_file)) for in_file in all_files]
#     pool.map(lambda in_f, out_f: add_transformation(in_f, out_f, transform, [deltaX, deltaY]), zip(all_files, out_files))
    pool_results = []
    for in_file in all_files:
        out_file = os.path.join(output_dir, os.path.basename(in_file))
        res = pool.apply_async(add_transformation, (in_file, out_file, transform, [deltaX, deltaY]))
        pool_results.append(res)

    for res in pool_results:
        res.get()





if __name__ == '__main__':
    # Command line parser
    parser = argparse.ArgumentParser(description='Given a list of tilespec file names, normalizes all the tilepecs to a single coordinate system starting from (0,0).')
    parser.add_argument('tile_files_or_dirs', metavar='tile_files_or_dirs', type=str, nargs='+',
                        help='a list of json files that need to be normalized or a directories of json files')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='an output directory (default: ./after_norm)',
                        default='./after_norm')
    parser.add_argument('-p', '--processes_num', type=int,
                        help='number of processes (default: 1)',
                        default=1)

    args = parser.parse_args()

    logger.start_process('normalize_coordinates', 'normalize_coordinates.py', [args.tile_files_or_dirs, args.output_dir])
    pool = mp.Pool(processes=args.processes_num)

    normalize_coordinates(args.tile_files_or_dirs, args.output_dir, pool)

    pool.close()
    pool.join()

    logger.end_process('normalize_coordinates', rh_logger.ExitCode(0))

