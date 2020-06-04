import argparse
import sys
from rh_logger.api import logger
import logging
import rh_logger
import os
import re
import glob
import common
import pickle
from mb_aligner.dal.section import Section
import fs
import multiprocessing as mp


def sec_dir_to_wafer_section(sec_dir, args_wafer_num=None):
    wafer_folder = sec_dir.split(os.sep)[-4]
    section_folder = sec_dir.split(os.sep)[-2]

    if args_wafer_num is None:
        m = re.match('.*[W|w]([0-9]+).*', wafer_folder)
        if m is None:
            raise Exception("Couldn't find wafer number from section directory {} (wafer dir is: {})".format(sec_dir, wafer_folder))
        wafer_num = int(m.group(1))
    else:
        wafer_num = args_wafer_num

    m = re.match('.*_S([0-9]+)R1+.*', section_folder)
    if m is None:
        raise Exception("Couldn't find section number from section directory {} (section dir is: {})".format(sec_dir, section_folder))
    sec_num = int(m.group(1))

    return wafer_num, sec_num


def get_layer_num(sec_num, initial_layer_num, reverse, max_sec_num):
    if reverse:
        layer_num = max_sec_num - sec_num + initial_layer_num
    else:
        layer_num = sec_num + initial_layer_num - 1
    return layer_num


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Given a wafer's folder, searches for the recent sections, and creates a per-section tilespec file.")
    parser.add_argument("--wafer_folder", metavar="wafer_folder", required=True,
                        help="a folder of a single wafer containing workflow folders")
    parser.add_argument("-o", "--output_dir", metavar="output_dir",
                        help="The output directory name, where each section folder will have a json tilespec there")
    parser.add_argument("-i", "--initial_layer_num", metavar="initial_layer_num", type=int,
                        help="The layer# of the first section in the list. (default: 1)",
                        default=1)
    parser.add_argument("-f", "--filtered_mfovs_pkl", metavar="filtered_mfovs_pkl", type=str,
                        help="The name of the pkl file that has a per section mfovs list (default: None)",
                        default=None)
    parser.add_argument("-w", "--wafer_num", metavar="wafer_num", type=int,
                        help="Manually set the wafer number for the output files (default: parse from wafer folder)",
                        default=None)
    parser.add_argument('--reverse', action='store_true',
                        help='reverse the section numbering (reversed filename lexicographical order)')
    parser.add_argument("-p", "--processes_num", metavar="processes_num", type=int,
                        help="The unmber of processes to use (default: 1)",
                        default=1)
    
    return parser.parse_args(args)

def create_and_save_single_section(sec_relevant_mfovs, sections_map_sec_num, layer_num, wafer_folder, out_ts_fname):

    cur_fs = fs.open_fs(wafer_folder)
    if isinstance(sections_map_sec_num, list):
        # TODO - not implemented yet
        section = Section.create_from_mfovs_image_coordinates(sections_map_sec_num, layer_num, cur_fs=cur_fs, relevant_mfovs=sec_relevant_mfovs)
    else:
        section = Section.create_from_full_image_coordinates(sections_map_sec_num, layer_num, cur_fs=cur_fs, relevant_mfovs=sec_relevant_mfovs)
    section.save_as_json(out_ts_fname)


def parse_filtered_mfovs(filtered_mfovs_pkl):
    with open(filtered_mfovs_pkl, 'rb') as in_f:
        data = pickle.load(in_f)
    filtered_mfovs_map = {}
    # map the filtered_mfovs_map and the sorted_sec_keys
    for k, v in data.items():
        wafer_num = int(k.split('_')[0][1:])
        section_num = int(k.split('_')[1][1:4])
#         v[0] = v[0].replace('\\', '/')
#         v[1] = set(int(mfov_num) for mfov_num in v[1])
#         filtered_mfovs_map[wafer_num, section_num] = v
        filtered_mfovs_map[wafer_num, section_num] = set(int(mfov_num) for mfov_num in v[1])
 
    return filtered_mfovs_map

def create_tilespecs(args):

    cur_fs = fs.open_fs(args.wafer_folder)
    # parse the workflows directory
    sections_map = common.parse_workflows_folder(cur_fs, args.wafer_folder)

    logger.report_event("Finished parsing sections", log_level=logging.INFO)

    sorted_sec_keys = sorted(list(sections_map.keys()))
    if min(sorted_sec_keys) != 1:
        logger.report_event("Minimal section # found: {}".format(min(sorted_sec_keys)), log_level=logging.WARN)
    
    logger.report_event("Found {} sections in {}".format(len(sections_map), args.wafer_folder), log_level=logging.INFO)
    max_sec_num = max(sorted_sec_keys)
    if len(sorted_sec_keys) != max_sec_num:
        logger.report_event("There are {} sections, but maximal section # found: {}".format(len(sections_map), max(sorted_sec_keys)), log_level=logging.WARN)
        missing_sections = [i for i in range(1, max(sorted_sec_keys)) if i not in sections_map]
        logger.report_event("Missing sections: {}".format(missing_sections), log_level=logging.WARN)
    
    # if there's a filtered mfovs file, parse it
    filtered_mfovs_map = None
    if args.filtered_mfovs_pkl is not None:
        logger.report_event("Filtering sections mfovs", log_level=logging.INFO)
        filtered_mfovs_map = parse_filtered_mfovs(args.filtered_mfovs_pkl)

    logger.report_event("Outputing sections to tilespecs directory: {}".format(args.output_dir), log_level=logging.INFO)

    common.fs_create_dir(args.output_dir)

    pool = mp.Pool(processes=args.processes_num)

    pool_results = []

    for sec_num in sorted_sec_keys:
        # extract wafer and section# from directory name
        if isinstance(sections_map[sec_num], list):
            wafer_num, sec_num = sec_dir_to_wafer_section(os.path.dirname(sections_map[sec_num][0]), args.wafer_num)
        else:
            wafer_num, sec_num = sec_dir_to_wafer_section(sections_map[sec_num], args.wafer_num)
        out_ts_fname = os.path.join(args.output_dir, 'W{}_Sec{}_montaged.json'.format(str(wafer_num).zfill(2), str(sec_num).zfill(3)))
        if os.path.exists(out_ts_fname):
            logger.report_event("Already found tilespec: {}, skipping".format(os.path.basename(out_ts_fname)), log_level=logging.INFO)
            continue
        layer_num = get_layer_num(sec_num, args.initial_layer_num, args.reverse, max_sec_num)


        sec_relevant_mfovs = None
        if filtered_mfovs_map is not None:
            if (wafer_num, sec_num) not in filtered_mfovs_map:
                logger.report_event("WARNING: cannot find filtered data for (wafer, sec): {}, skipping".format((wafer_num, sec_num)), log_level=logging.INFO)
                continue
            sec_relevant_mfovs = filtered_mfovs_map[wafer_num, sec_num]


        res = pool.apply_async(create_and_save_single_section, (sec_relevant_mfovs, sections_map[sec_num], layer_num, args.wafer_folder, out_ts_fname))
        pool_results.append(res)

    for res in pool_results:
        res.get()

if __name__ == '__main__':
    args = parse_args()

    logger.start_process('main', 'create_tilespecs.py', [args])
    create_tilespecs(args)
    logger.end_process('main ending', rh_logger.ExitCode(0))


