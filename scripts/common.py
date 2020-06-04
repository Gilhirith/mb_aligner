import subprocess
import numpy as np
import os
import glob
import re
from rh_logger.api import logger
import logging
import rh_logger
import fs.path
import fs
from urllib.parse import urlparse

def read_bboxes_grep(ts_fname):
    def parse_bbox_lines(bbox_lines):
        str = ''.join(bbox_lines)
        str = str[str.find('[') + 1:str.find(']')]
        bbox = [float(x) for x in str.split(',')]
        return bbox

    if "gs://" in ts_fname:
        cmd = "gsutil cat {} | grep -A 5 \"bbox\"".format(ts_fname)
    else:
        ts_fname = ts_fname.replace("osfs://", "")
        ts_fname = ts_fname.replace("file://", "")
        cmd = "grep -A 5 \"bbox\" {}".format(ts_fname)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Parse all bounding boxes in the given json file
    per_tile_bboxes = []
    cur_bbox_lines = []
    for line in iter(p.stdout.readline, ''):
        line = line.decode("utf-8")
        if line == '':
            break
        if line.startswith("--"):
            cur_bbox = parse_bbox_lines(cur_bbox_lines)
            per_tile_bboxes.append(cur_bbox)
            cur_bbox_lines = []
        else:
            cur_bbox_lines.append(line)
    if len(cur_bbox_lines) > 0:
        cur_bbox = parse_bbox_lines(cur_bbox_lines)
        per_tile_bboxes.append(cur_bbox)

    if len(per_tile_bboxes) == 0:
        return None
    per_tile_bboxes = np.array(per_tile_bboxes)
    entire_image_bbox = [np.min(per_tile_bboxes[:, 0]), np.max(per_tile_bboxes[:, 1]),
                         np.min(per_tile_bboxes[:, 2]), np.max(per_tile_bboxes[:, 3])]
    return entire_image_bbox

def read_bboxes_grep_pool(all_files, pool):

    all_bboxes = pool.map(read_bboxes_grep, all_files)
    all_bboxes = np.array(all_bboxes)
    entire_image_bbox = [np.min(all_bboxes[:, 0]), np.max(all_bboxes[:, 1]), np.min(all_bboxes[:, 2]), np.max(all_bboxes[:, 3])]

    return entire_image_bbox


def parse_workflow_folder(cur_fs, workflow_folder):
    '''
    Parses a single folder which has a coordinates text file for the section or multiple coordinates text files for its mfovs.
    The section coordinates filename will be of the format: full_image_coordinates.txt (or full_image_coordinates_corrected.txt),
    and the per-mfov image coordinates file will be: image_coordinates.txt
    Each workflow folder will have the following format:
    [N]_S[S]R1
    Where [N] is a 3-digit number (irrelevant), and [S] is the section number.
    Returns a map between a section number and the coordinates txt filename (or filenames in case no full section coordinates file was found).
    '''
    result = {}
    all_sec_folders = sorted(cur_fs.glob(workflow_folder + '/*_S*R1/'))
    for sec_folder_glob in all_sec_folders:
        sec_folder = sec_folder_glob.path
        m = re.match('.*/([0-9]{3})_S([0-9]+)R1/$', sec_folder)
        if m is not None:
            # make sure the txt file is there
            image_coordinates_files = None
            if cur_fs.exists(sec_folder + 'full_image_coordinates_corrected.txt'):
                image_coordinates_files = sec_folder + 'full_image_coordinates_corrected.txt'
            elif cur_fs.exists(sec_folder + 'full_image_coordinates.txt'):
                image_coordinates_files = sec_folder + 'full_image_coordinates.txt'
            else:
                # look for all mfov folders
                mfov_folders = sorted(cur_fs.glob(sec_folder + "0*/"))
                if len(mfov_folders) == 0:
                    logger.report_event("Could not detect coordinate/mfov files for sec: {} - skipping".format(sec_folder), log_level=logging.WARN)
                    continue
                all_mfov_folders_have_coordinates = True
                for mfov_folder_glob in mfov_folders:
                    mfov_folder = mfov_folder_glob.path
                    if not cur_fs.exists(mfov_folder + "image_coordinates.txt"):
                        logger.report_event("Could not detect mfov coordinates file for mfov: {} - skipping".format(mfov_folder), log_level=logging.WARN)
                        all_mfov_folders_have_coordinates = False
                if not all_mfov_folders_have_coordinates:
                    continue
                # Take the mfovs folders into account
                image_coordinates_files = [mfov_folder.path + "image_coordinates.txt" for mfov_folder in mfov_folders]
            sec_num = int(m.group(2))
            result[sec_num] = image_coordinates_files
    return result


def parse_workflows_folder(cur_fs, workflows_folder):
    '''
    Parses a folder which has at least one workflow folder.
    Each workflow folder will have the pattern [name]_[date]_[time],
    where [name] can be anything, [date] will be of the format YYYMMDD,
    and [time] will be HH-MM-SS.
    '''
    sub_folders = cur_fs.glob("*/")
    all_workflow_folders = []
    dir_to_time = {}
    for folder_glob in sub_folders:
        if folder_glob.info.is_dir:
            folder = folder_glob.path
            m = re.match('.*_([0-9]{8})_([0-9]{2})-([0-9]{2})-([0-9]{2})/$', folder)
            if m is not None:
                dir_to_time[folder] = "{}_{}-{}-{}".format(m.group(1), m.group(2), m.group(3), m.group(4))
                all_workflow_folders.append(folder)

    full_result = {}
    for sub_folder in sorted(all_workflow_folders, key=lambda folder: dir_to_time[folder]):
        if cur_fs.isdir(sub_folder):
            logger.report_event("Parsing sections from subfolder: {}".format(sub_folder), log_level=logging.INFO)
            full_result.update(parse_workflow_folder(cur_fs, sub_folder))

    # For debug
    # first_keys = sorted(list(full_result.keys()))[:20]
    # full_result = {k:full_result[k] for k in first_keys}
    return full_result

def get_fs_parsed_url(url):
    parsed_url = urlparse(url)
    url_prefix = "{}://{}".format(parsed_url.scheme, parsed_url.netloc)
    return url_prefix, parsed_url.path

def fs_create_dir(output_dir):
    fs_loc, fs_path = get_fs_parsed_url(output_dir)
    with fs.open_fs(fs_loc) as out_fs:
        if not out_fs.exists(fs_path):
            out_fs.makedirs(fs_path)

def get_ts_files(ts_folder):

    with fs.open_fs(ts_folder) as cur_fs:
        all_ts_fnames = []
        all_ts_fnames_glob = cur_fs.glob("*.json")
        for ts_fname_glob in all_ts_fnames_glob:
            ts_fname = ts_fname_glob.path
            all_ts_fnames.append(cur_fs.geturl(ts_fname))
        return all_ts_fnames


