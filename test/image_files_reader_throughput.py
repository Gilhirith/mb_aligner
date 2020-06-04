import os
import sys
import multiprocessing as mp
import time


def pre_processes_section_dir(section_dir):
    """
    Creates a dictionary for all the mfovs in the section directory mapping to their image files
    """
    result = {}
    with os.scandir(section_dir) as it:
        for entry in it:
            if entry.name.startswith("00") and entry.is_dir():
                mfov_files = os.listdir(os.path.join(section_dir, entry.name))
                mfov_images = [fname for fname in mfov_files if fname.endswith(".bmp")]
                result[os.path.join(section_dir, entry.name)] = mfov_images
    return result

def pre_processes_dir(in_wafer_workflow_dir):
    """
    Creates a dictionary for all the mfovs in the workflow directory
    """
    result = {}
    with os.scandir(in_wafer_workflow_dir) as it:
        for entry in it:
            if "_S" in entry.name and entry.is_dir():
                section_result = pre_processes_section_dir(os.path.join(in_wafer_workflow_dir, entry.name))
                result.update(section_result)
    return result
    

def read_mfov_files(mfov_dir, mfov_files):
    data_len = 0
    for mfov_file in mfov_files:
        mfov_fname = os.path.join(mfov_dir, mfov_file)
        with open(mfov_fname, "rb") as in_f:
            data = in_f.read()
        data_len += len(data)
    return data_len

def read_all_files(mfovs_map, processes_num):
    """
    Receives a file structure map, and reads all the image files
    """
    pool = mp.Pool(processes=processes_num)

    pool_results = []
    for mfov_dir, mfov_files in mfovs_map.items():
        res = pool.apply_async(read_mfov_files, (mfov_dir, mfov_files))
        pool_results.append(res)

    pool.close()
    pool.join()

    total_bytes = 0
    for res in pool_results:
        total_bytes += res.get()
    return total_bytes
    

def read_image_files_throughput(in_wafer_dir, processes_num):
    print("Building files structure")
    st_time = time.time()
    mfovs_map = pre_processes_dir(in_wafer_dir)
    print("Took {} seconds".format(time.time() - st_time))

    print("Found {} mfov directories. Reading...".format(len(mfovs_map)))
    st_time = time.time()
    total_bytes = read_all_files(mfovs_map, processes_num)
    end_time = time.time()
    total_gbs = float(total_bytes)/(1024**3)
    print("Reading {} GBs took {} seconds ({} GBs/sec)".format(total_gbs, end_time - st_time, total_gbs / (end_time - st_time)))



if __name__ == '__main__':

    #in_wafer_workflow_dir = '/n/lichtmangpfs01/Instrument_drop/Nag/P14_cerebellum/reel1_wafer03_holder03/reel1_wafer03_holder03_20190101_13-27-47'
    #in_wafer_workflow_dir = '/n/lichtmangpfs01/Instrument_drop/Nag/P14_cerebellum/reel1_wafer03_holder03/reel1_wafer03_holder03_20190102_09-11-33'
    in_wafer_workflow_dir = sys.argv[1]
    processes_num = 16
    if len(sys.argv) > 2:
        processes_num = int(sys.argv[2])

    read_image_files_throughput(in_wafer_workflow_dir, processes_num)

