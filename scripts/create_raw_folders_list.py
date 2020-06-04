import argparse
import sys
from rh_logger.api import logger
import logging
import rh_logger
import os
import re
import glob
import common




def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Given a wafer's folder, searches for the recent sections, and creates an ordered folders sections text file.")
    parser.add_argument("--wafer_folder", metavar="wafer_folder", required=True,
                        help="a folder of a single wafer containing workflow folders")
    parser.add_argument("-o", "--output_file", metavar="output_file",
                        help="The output text file name, where each section folder will be in a separate line")
    
    return parser.parse_args(args)


def create_raw_folders_list(args):

    # parse the workflows directory
    sections_map = common.parse_workflows_folder(args.wafer_folder)

    logger.report_event("Finished parsing sections", log_level=logging.INFO)

    sorted_sec_keys = sorted(list(sections_map.keys()))
    if min(sorted_sec_keys) != 1:
        logger.report_event("Minimal section # found: {}".format(min(sorted_sec_keys)), log_level=logging.WARN)
    
    logger.report_event("Found {} sections in {}".format(len(sections_map), args.wafer_folder), log_level=logging.INFO)
    if len(sorted_sec_keys) != max(sorted_sec_keys):
        logger.report_event("There are {} sections, but maximal section # found: {}".format(len(sections_map), max(sorted_sec_keys)), log_level=logging.WARN)
        missing_sections = [i for i in range(1, max(sorted_sec_keys)) if i not in sections_map]
        logger.report_event("Missing sections: {}".format(missing_sections), log_level=logging.WARN)
    
    logger.report_event("Outputing sections to list to: {}".format(args.output_file), log_level=logging.INFO)

    with open(args.output_file, 'wt') as out_f:
        for sec_num in sorted_sec_keys:
            if isinstance(sections_map[sec_num], list):
                folder_name = os.sep.join(sections_map[sec_num][0].split(os.sep)[:-2]) # strip the mfov path, and the coordinates file name
            else:
                folder_name = os.sep.join(sections_map[sec_num].split(os.sep)[:-1]) # strip the coordinates file name
            out_f.write(folder_name)
            out_f.write('\n')


if __name__ == '__main__':
    args = parse_args()

    logger.start_process('main', 'create_raw_folders_list.py', [args])
    create_raw_folders_list(args)
    logger.end_process('main ending', rh_logger.ExitCode(0))


