import sys
import os
import glob
import yaml
import cv2
import numpy as np
from rh_logger.api import logger
import logging
import rh_logger
import time
from mb_aligner.dal.section import Section
from mb_aligner.common.detector import FeaturesDetector
from mb_aligner.common.matcher import FeaturesMatcher
from mb_aligner.factories.processes_factory import ProcessesFactory
#from mb_aligner.stitching.optimize_2d_mfovs import OptimizerRigid2D
import multiprocessing as mp
import threading
#import queue
from collections import defaultdict
import tinyr
import importlib
import mb_aligner.common.utils
import pickle

#from ..pipeline.task_runner import TaskRunner
#import queue

class DetectorWorker(object):
    def __init__(self, processes_factory, input_queue, all_result_queues, detector_top_mask):
        self._detector = processes_factory.create_2d_detector()
        self._input_queue = input_queue
        self._all_result_queues = all_result_queues
        self._detector_top_mask = detector_top_mask

    def run(self):
        # Read a job from input queue (blocking)
        #print('Detector running')

        while True:
            #print("Detector queue size:", self._input_queue.qsize())
            job = self._input_queue.get()
            if job is None:
                break
            # job = (matcher's result queue idx, tile id, tile, start_point, crop_bbox)
            #out_queue, tile_fname, start_point, crop_bbox = job
            out_queue_idx, tile_id, tile, start_point, crop_bbox = job
            out_queue = self._all_result_queues[out_queue_idx]
            # process the job
            #print("Received job:", job)
            #print("Reading file:", tile.img_fname)
            #img = cv2.imread(tile_fname, 0)
            img = tile.image

            if self._detector_top_mask is not None:
                img = img[self._detector_top_mask:, :]
                start_point = np.array(start_point)
                start_point[1] += self._detector_top_mask
            if crop_bbox is not None:
                # Find the overlap between the given bbox and the tile actual bounding box,
                # and crop that overlap area
                # normalize the given bbox to the tile coordinate system, and then make sure that bounding box is in valid with the image size
                local_crop_bbox = [crop_bbox[0] - start_point[0], crop_bbox[1] - start_point[0], crop_bbox[2] - start_point[1], crop_bbox[3] - start_point[1]]
                local_crop_bbox = [
                                max(int(local_crop_bbox[0]), 0),
                                min(int(local_crop_bbox[1]), img.shape[1]),
                                max(int(local_crop_bbox[2]), 0),
                                min(int(local_crop_bbox[3]), img.shape[0])
                                  ]
                img = img[local_crop_bbox[2]:local_crop_bbox[3], local_crop_bbox[0]:local_crop_bbox[1]]
                #print("local_crop_bbox: {}".format(local_crop_bbox))
            scale_percent = 50 # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)    
            #print("detecting features file:", job[1])
            kps, descs = self._detector.detect(img)
            # Create an array of the points of the kps
            kps_pts = np.empty((len(kps), 2), dtype=np.float64)
            for kp_i, kp in enumerate(kps):
                kps_pts[kp_i][:] = kp.pt
#             # Change the features key points to the world coordinates
#             delta = np.array(start_point)
#             if crop_bbox is not None:
#                 delta[0] += local_crop_bbox[0]
#                 delta[1] += local_crop_bbox[2]
#             kps_pts += delta
            if crop_bbox is not None:
                kps_pts[:, 0] += local_crop_bbox[0]
                kps_pts[:, 1] += local_crop_bbox[2]

            if self._detector_top_mask is not None:
                kps_pts[:, 1] += self._detector_top_mask
                #start_point[1] -= self._detector_top_mask # no need because we copied start_point

            # result = (tile fname, local area, features pts, features descs)
            # Put result in matcher's result queue
            #print("submitting result for:", tile.img_fname)
            #print("Result[:30]:\n{}".format(kps_pts[:30]))
            out_queue.put((tile_id, kps_pts, descs))


class MatcherWorker(object):
    def __init__(self, processes_factory, input_queue, output_queue, matcher_queue, detectors_in_queue, matcher_thread_idx):
        self._matcher = processes_factory.create_2d_matcher()
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._matcher_queue = matcher_queue
        self._detectors_in_queue = detectors_in_queue
        self._matcher_thread_idx = matcher_thread_idx

    def run(self):
        # Read a job from input queue (blocking)
        #print('Detector running')

        while True:
            #print("Matcher queue size:", self._input_queue.qsize())
            job = self._input_queue.get()
            if job is None:
                break
            # job = (match_idx, tile1, tile2)
            match_idx, tile1, tile2 = job
            bbox1 = tile1.bbox
            bbox2 = tile2.bbox
#             # job = (match_idx, img_shape, all_tiles_list, tile_idx1, tile_idx2)
#             match_idx, img_shape, all_tiles_list, tile_idx1, tile_idx2 = job
#             tile_fname1, start_point1 = all_tiles_list[tile_idx1]
#             tile_fname2, start_point2 = all_tiles_list[tile_idx2]
            # process the job
            #print("Received match job:", match_idx)
            # Find shared bounding box
            extend_delta = 180 # TODO - should be a parameter
            #extend_delta = 100 # TODO - should be a parameter
            #extend_delta = 5 # TODO - should be a parameter
            intersection = [max(bbox1[0], bbox2[0]) - extend_delta,
                            min(bbox1[1], bbox2[1]) + extend_delta,
                            max(bbox1[2], bbox2[2]) - extend_delta,
                            min(bbox1[3], bbox2[3]) + extend_delta]

            #print("bbox1: {}".format(bbox1))
            #print("bbox2: {}".format(bbox2))
            #print("intersection: {}".format(intersection))
#             intersection = [max(start_point1[0], start_point2[0]) - extend_delta,
#                             min(start_point1[0] + img_shape[1], start_point2[0] + img_shape[1]) + extend_delta,
#                             max(start_point1[1], start_point2[1]) - extend_delta,
#                             min(start_point1[1] + img_shape[0], start_point2[1] + img_shape[0]) + extend_delta]

            # Send two detector jobs
            #print('Submitting job1 to detectors_in_queue')
            #job1 = (self._matcher_queue, tile1_id, tile1, (bbox1[0], bbox1[2]), intersection)
            tile1_id = (tile1.layer, tile1.mfov_index, tile1.tile_index)
            job1 = (self._matcher_thread_idx, tile1_id, tile1, (bbox1[0], bbox1[2]), intersection)
            self._detectors_in_queue.put(job1)
            #print('Submitting job2 to detectors_in_queue')
            #job2 = (self._matcher_queue, tile2_id, tile2, (bbox2[0], bbox2[2]), intersection)
            tile2_id = (tile2.layer, tile2.mfov_index, tile2.tile_index)
            job2 = (self._matcher_thread_idx, tile2_id, tile2, (bbox2[0], bbox2[2]), intersection)
            self._detectors_in_queue.put(job2)

            # fetch the results
            res_a = self._matcher_queue.get()
            res_b = self._matcher_queue.get()
            # res_a = (tile_A, kps_pts_A, descs_A))
            if res_a[0] == tile1_id:
                _, kps_pts1, descs1 = res_a
                _, kps_pts2, descs2 = res_b
            else:
                assert(res_a[0] == tile2_id)
                _, kps_pts1, descs1 = res_b
                _, kps_pts2, descs2 = res_a

            # perform the actual matching
            transform_model, filtered_matches = self._matcher.match_and_filter(kps_pts1, descs1, kps_pts2, descs2)
            #print("Transform_model:\n{}".format(None if transform_model is None else transform_model.get_matrix()))
            #print("filtered_matches ({}, {}):\n{}".format(tile1.mfov_index, tile2.mfov_index, filtered_matches))

#             # TODO - add fake matches in case none were found
#             if filtered_matches is None and tile1.mfov_index == tile2.mfov_index:
            #if filtered_matches is None:
            #    logger.report_event("Adding fake matches between: {} and {}".format((tile1.mfov_index, tile1.tile_index), (tile2.mfov_index, tile2.tile_index)), log_level=logging.INFO)
            #    intersection = [max(bbox1[0], bbox2[0]),
            #                    min(bbox1[1], bbox2[1]),
            #                    max(bbox1[2], bbox2[2]),
            #                    min(bbox1[3], bbox2[3])]
            #    intersection_center = np.array([intersection[0] + intersection[1], intersection[2] + intersection[3]]) * 0.5
            #    fake_match_points_global = np.array([
            #            [intersection_center[0] + intersection[0] - 2, intersection_center[1] + intersection[2] - 2],
            #            [intersection_center[0] + intersection[1] + 4, intersection_center[1] + intersection[2] - 4],
            #            [intersection_center[0] + intersection[0] + 2, intersection_center[1] + intersection[3] - 2],
            #            [intersection_center[0] + intersection[1] - 4, intersection_center[1] + intersection[3] - 6]
            #        ]) * 0.5
            #    filtered_matches = np.array([
            #            fake_match_points_global - np.array([bbox1[0], bbox1[2]]),
            #            fake_match_points_global - np.array([bbox2[0], bbox2[2]])
            #        ])
            #    #print("filtered_matches: {}".format(filtered_matches))
            #print("filtered_matches.shape: {}".format(filtered_matches.shape))

            self._output_queue.put((match_idx, filtered_matches))

            # return the filtered matches (the points for each tile in the global coordinate system)
            #assert(transform_model is not None)
            #transform_matrix = transform_model.get_matrix()
            #logger.report_event("Imgs {} -> {}, found the following transformations\n{}\nAnd the average displacement: {} px".format(i, j, transform_matrix, np.mean(Stitcher._compute_l2_distance(transform_model.apply(filtered_matches[1]), filtered_matches[0]))), log_level=logging.INFO)



class ThreadWrapper(object):
    def __init__(self, ctor, *args, **kwargs):
        self._process = threading.Thread(target=ThreadWrapper.init_and_run, args=(ctor, args), kwargs=kwargs)
        self._process.start()

    @staticmethod
    def init_and_run(ctor, args, **kwargs):
        #print("ctor:", ctor)
        #print("args:", args[0])
        worker = ctor(*args[0], **kwargs)
        worker.run()

    def join(self, timeout=None):
        return self._process.join(timeout)
    # TODO - add a process kill method

class ProcessWrapper(object):
    def __init__(self, ctor, *args, **kwargs):
        self._process = mp.Process(target=ProcessWrapper.init_and_run, args=(ctor, args), kwargs=kwargs)
        self._process.start()

    @staticmethod
    def init_and_run(ctor, args, **kwargs):
        #print("ctor:", ctor)
        #print("args:", args[0])
        worker = ctor(*args[0], **kwargs)
        worker.run()

    def join(self, timeout=None):
        return self._process.join(timeout)
    # TODO - add a process kill method



# class ProcessWrapper(object):
#     def __init__(self, ctor, *args, **kwargs):
#         self._process = mp.Process(target=ProcessWrapper.init_and_run, args=(ctor, args), kwargs=kwargs)
#         self._process.start()
# 
#     @staticmethod
#     def init_and_run(ctor, args, **kwargs):
#         print("ctor:", ctor)
#         print("args:", args[0])
#         worker = ctor(*args[0], **kwargs)
#         worker.run()
# 
#     def join(self, timeout=None):
#         return self._process.join(timeout)
#     # TODO - add a process kill method



class Stitcher(object):

    def __init__(self, conf):
        self._conf = conf


        missing_matches_policy_type = conf.get('missing_matches_policy_type', None)
        if missing_matches_policy_type is None:
            self._missing_matches_policy = None
        else:
            missing_matches_policy_params = conf.get('missing_matches_policy_params', {})
            self._missing_matches_policy = mb_aligner.common.utils.load_plugin(missing_matches_policy_type)(**missing_matches_policy_params)

        optimizer_type = conf.get('optimizer_type')
        optimizer_params = conf.get('optimizer_params', {})
        self._optimizer = mb_aligner.common.utils.load_plugin(optimizer_type)(**optimizer_params)

        self._processes_factory = ProcessesFactory(self._conf)
        # Initialize the detectors, matchers and optimizer objects
        #reader_params = conf.get('reader_params', {})
        detector_params = conf.get('detector_params', {})
        matcher_params = conf.get('matcher_params', {})
        detector_top_mask = conf.get('detector_top_mask', None)
        #reader_threads = reader_params.get('threads', 1)
        detector_threads = conf.get('detector_threads', 1)
        matcher_threads = conf.get('matcher_threads', 1)
        #assert(reader_threads > 0)
        assert(detector_threads > 0)
        assert(matcher_threads > 0)

        self._intermediate_directory = conf.get('intermediate_dir', None)
        if self._intermediate_directory is not None:
            if not os.path.exists(self._intermediate_directory):
                os.makedirs(self._intermediate_directory)


        # The design is as follows:
        # - There will be N1 detectors and N2 matchers (each with its own thread/process - TBD)
        # - All matchers have a single queue of tasks which they consume. It will be populated before the optimizer is called
        # - Each matcher needs to detect features in overlapping areas of 2 tiles. To that end, the matcher adds 2 tasks (for each tile and area)
        #   to the single input queue that's shared between all detectors. The detector performs its operation, and returns the result directly to the matcher.
        #   Once the features of the two tiles were computed, the matcher does the matching, and returns the result to a single result queue.
        # TODO - it might be better to use other techniques for sharing data between processes (e.g., shared memory, moving to threads-only, etc.)

        # Set up the detectors and matchers input queue
        self._detectors_in_queue = mp.Queue(maxsize=detector_params.get("queue_max_size", 0))
        self._matchers_in_queue = mp.Queue(maxsize=matcher_params.get("queue_max_size", 0))
        self._detectors_result_queues = [mp.Queue(maxsize=matcher_params.get("queue_max_size", 0)) for i in range(matcher_threads)] # each matcher will receive the detectors results in its own queue
        self._matchers_out_queue = mp.Queue(maxsize=matcher_params.get("queue_max_size", 0)) # Used by the manager to collect the matchers results

        # Set up the pool of detectors
        #self._detectors = [ThreadWrapper(DetectorWorker, (self._processes_factory, self._detectors_in_queue, self._detectors_result_queues)) for i in range(detector_threads)]
        #self._matchers = [ThreadWrapper(MatcherWorker, (self._processes_factory, self._matchers_in_queue, self._matchers_out_queue, self._detectors_result_queues[i], self._detectors_in_queue, i)) for i in range(matcher_threads)]
        self._detectors = [ProcessWrapper(DetectorWorker, (self._processes_factory, self._detectors_in_queue, self._detectors_result_queues, detector_top_mask)) for i in range(detector_threads)]
        self._matchers = [ProcessWrapper(MatcherWorker, (self._processes_factory, self._matchers_in_queue, self._matchers_out_queue, self._detectors_result_queues[i], self._detectors_in_queue, i)) for i in range(matcher_threads)]
        
#         # Set up the pools of dethreads (each thread will have its own queue to pass jobs around)
#         # Set up the queues
#         #readers_in_queue = queue.Queue(maxsize=reader_params.get("queue_max_size", 0)]))
#         detectors_in_queue = queue.Queue(maxsize=detector_params.get("queue_max_size", 0)]))
#         matchers_in_queue = queue.Queue(maxsize=matcher_params.get("queue_max_size", 0)]))
# 
#         # Set up the threads
# #         self._readers = [TaskRunner(name="reader_%d" % (_ + 1),
# #                                     input_queue=readers_in_queue,
# #                                     output_queue=detectors_in_queue)
# #                            for _ in range(reader_threads)]
#  
#         self._detectors = [TaskRunner(name="detector_%d" % (_ + 1),
#                                       input_queue=detectors_in_queue,
#                                       output_queue=matchers_in_queue,
#                                       init_fn=lambda: FeaturesDetector(conf['detector_type'], **detector_params))
#                            for _ in range(detector_threads)]
# 
#         self._matchers = [TaskRunner(name="matcher_%d" % (_ + 1),
#                                       input_queue=matchers_in_queue,
#                                       init_fn=lambda: FeaturesMatcher(FeaturesDetector(conf['detector_type'], **detector_params), **matcher_params))
#                            for _ in range(detector_threads)]
# 
#         #self._detector = FeaturesDetector(conf['detector_type'], **detector_params)
#         #self._matcher = FeaturesMatcher(self._detector, **matcher_params)


    def __del__(self):
        self.close()

    def close(self):

        print("Closing all matchers")
        for i in range(len(self._matchers)):
            self._matchers_in_queue.put(None)
        for m in self._matchers:
            m.join()
        print("Closing all detectors")
        for i in range(len(self._detectors)):
            self._detectors_in_queue.put(None)
        for d in self._detectors:
            d.join()

    @staticmethod
    def read_imgs(folder):
        img_fnames = sorted(glob.glob(os.path.join(folder, '*')))[:10]
        print("Loading {} images from {}.".format(len(img_fnames), folder))
        imgs = [cv2.imread(img_fname, 0) for img_fname in img_fnames]
        return img_fnames, imgs


    @staticmethod
    def load_conf_from_file(conf_fname):
        '''
        Loads a given configuration file from a yaml file
        '''
        print("Using config file: {}.".format(conf_fname))
        if conf_fname is None:
            return {}
        with open(conf_fname, 'r') as stream:
            conf = yaml.load(stream)
            conf = conf['stitching']
        
        logger.report_event("loaded configuration: {}".format(conf), log_level=logging.INFO)
        return conf


    @staticmethod
    def _compute_l2_distance(pts1, pts2):
        delta = pts1 - pts2
        s = np.sum(delta**2, axis=1)
        return np.sqrt(s)

    @staticmethod
    def _compute_features(detector, img):
        result = detector.detect(img)
        return result

    def _compute_tile_features(self, tile, bboxes=None):
        img = tile.image
        if bbox is not None:
            # Find the overlap between the given bbox and the tile actual bounding box,
            # and crop that overlap area
            t_bbox = tile.bbox
            # normalize the given bbox to the tile coordinate system, and then make sure that bounding box is in valid with the image size
            crop_bbox = [bbox[0] - t_bbox[0], bbox[1] - t_bbox[0], bbox[2] - t_bbox[2], bbox[3] - t_bbox[2]]
            crop_bbox = [
                            max(int(crop_bbox[0]), 0),
                            min(int(crop_bbox[1]), tile.width),
                            max(int(crop_bbox[2]), 0),
                            min(int(crop_bbox[3]), tile.height)
                        ]
            img = img[crop_bbox[2]:crop_bbox[3], crop_bbox[0]:crop_bbox[1]]
            
        result = self._detector.detect(img)

        # Change the features key points to the world coordinates
        delta_x = tile.bbox[0]
        delta_y = tile.bbox[2]
        if bbox is not None:
            delta_x += crop_bbox[0]
            delta_y += crop_bbox[2]
        for kp in result[0]:
            cur_point = list(kp.pt)
            cur_point[0] += delta_x
            cur_point[1] += delta_y
            kp.pt = tuple(cur_point)

        return result


    @staticmethod
    def _match_features(features_result1, features_result2, i, j):
        transform_model, filtered_matches = self._matcher.match_and_filter(*features_result1, *features_result2)
        assert(transform_model is not None)
        transform_matrix = transform_model.get_matrix()
        logger.report_event("Imgs {} -> {}, found the following transformations\n{}\nAnd the average displacement: {} px".format(i, j, transform_matrix, np.mean(Stitcher._compute_l2_distance(transform_model.apply(filtered_matches[1]), filtered_matches[0]))), log_level=logging.INFO)
        return transform_matrix
 

    @staticmethod
    def _create_section_rtree(section):
        '''
        Receives a section, and returns an rtree of all the section's tiles bounding boxes
        '''
        tiles_rtree = tinyr.RTree(interleaved=False, max_cap=5, min_cap=2)
        # Insert all tiles bounding boxes to an rtree
        for t in section.tiles():
            bbox = t.bbox
            # using the (x_min, x_max, y_min, y_max) notation
            tiles_rtree.insert(t, bbox)

        return tiles_rtree



    @staticmethod
    def _find_overlapping_tiles_gen(section):
        '''
        Receives a section, and yields triplets of (tile1, tile2, overlap_bbox ())
        '''
        tiles_rtree = Stitcher._create_section_rtree(section)
        # Iterate over the section tiles, and for each tile find all of its overlapping tiles
        for t in section.tiles():
            bbox = t.bbox
            rect_res = tiles_rtree.search(bbox)
            for overlap_t in rect_res:
                # We want to create a directed comparison (each tile with tiles that come after it in a lexicographical order)
                if overlap_t.mfov_index > t.mfov_index or (overlap_t.mfov_index == t.mfov_index and overlap_t.tile_index > t.tile_index):
                    yield t, overlap_t
#                     # Compute overlap area
#                     overlap_bbox = overlap_t.bbox
#                     intersection = [max(bbox[0], overlap_bbox[0]),
#                                     min(bbox[1], overlap_bbox[1]),
#                                     max(bbox[2], overlap_bbox[2]),
#                                     min(bbox[3], overlap_bbox[3])]
# 
#                     yield t, overlap_t, intersection
            

    def _compute_match_features(self, section):
        logger.report_event("Starting feature computation and matching...", log_level=logging.INFO)
        st_time = time.time()
        match_jobs = []
        extend_delta = 50 # TODO - should be a parameter
        for tile1, tile2 in Stitcher._find_overlapping_tiles_gen(section):
            # Add a matching job
            # job = (match_idx, tile1, tile2)
            job = (len(match_jobs), tile1, tile2)
            match_jobs.append((tile1, tile2))
            #print('Submitting matching job')
            self._matchers_in_queue.put(job)


        # fetch all the results from the matchers
        match_results_map = {}
        logger.report_event("Collecting matches results", log_level=logging.INFO)
        res_match_jobs_counter = 0
        total_matches_num = 0
        while res_match_jobs_counter < len(match_jobs):
            match_idx, filtered_matches = self._matchers_out_queue.get()
            tile1, tile2 = match_jobs[match_idx]
            tile1_unique_idx = (tile1.layer, tile1.mfov_index, tile1.tile_index)
            tile2_unique_idx = (tile2.layer, tile2.mfov_index, tile2.tile_index)
            if filtered_matches is None or len(filtered_matches[0]) <= 3: # TODO - make a parameter
                if self._missing_matches_policy is None:
                    logger.report_event("Removing no matches for pair: {} -> {}".format(tile1_unique_idx, tile2_unique_idx), log_level=logging.INFO)
                else:
                    self._missing_matches_policy.add_missing_match({(tile1_unique_idx, tile2_unique_idx): (tile1, tile2)})
            else:
                match_results_map[(tile1_unique_idx, tile2_unique_idx)] = filtered_matches
                total_matches_num += len(filtered_matches[0])
            res_match_jobs_counter += 1
        
        missing_matches_map = []
        if self._missing_matches_policy is not None:
            logger.report_event("Handling missing matches", log_level=logging.INFO)
            missing_matches_map = self._missing_matches_policy.fix_missing_matches(match_results_map)
            match_results_map.update(missing_matches_map)
            self._missing_matches_policy.reset()
            

        # verify that we have more than 2 matches:
        if total_matches_num <= 2:
            logger.report_event("Couldn't find enough matches for section {}, skipping optimization".format(section.canonical_section_name_no_layer), log_level=logging.ERROR)
            return None
                
        if self._intermediate_directory is not None:
            intermediate_fname = os.path.join(self._intermediate_directory, '{}.pkl'.format(section.canonical_section_name_no_layer))
            logger.report_event("Saving intermediate result to: {}".format(intermediate_fname), log_level=logging.INFO)
            with open(intermediate_fname, 'wb') as out_f:
                pickle.dump([match_results_map, missing_matches_map], out_f, protocol=pickle.HIGHEST_PROTOCOL)


        return match_results_map, missing_matches_map


    def stitch_section(self, section, match_results_map=None):
        '''
        Receives a single section (assumes no transformations), stitches all its tiles, and updaates the section tiles' transformations.
        '''

        logger.report_event("stitch_section starting.", log_level=logging.INFO)
        # Compute features

        if match_results_map is None:
            match_results_map, missing_matches_map = self._compute_match_features(section)
            if match_results_map is None:
                return

        logger.report_event("Starting optimization", log_level=logging.INFO)
        # Generate a map between tile and its original estimated location
        orig_locations = {}
        for tile in section.tiles():
            tile_unique_idx = (tile.layer, tile.mfov_index, tile.tile_index)
            orig_locations[tile_unique_idx] = [tile.bbox[0], tile.bbox[2]]

        optimized_transforms_map = self._optimizer.optimize(orig_locations, match_results_map)

        #if self._filter_inter_mfov_matches:
        #    # find a "seam" of tiles between 

        logger.report_event("Done optimizing, updating tiles transforms", log_level=logging.INFO)

        for tile in section.tiles():
            tile_unique_idx = (tile.layer, tile.mfov_index, tile.tile_index)
            if tile_unique_idx not in optimized_transforms_map:
                # TODO - should remove the tile
                logger.report_event("Could not find a transformation for tile {} in the optimization result, skipping tile".format(tile_unique_idx), log_level=logging.WARNING)
            else:
                tile.set_transform(optimized_transforms_map[tile_unique_idx])
                #print("Mfov {}, transform:\n{}".format(tile.mfov_index, tile.transforms[0].get_matrix()))




    @staticmethod
    def align_img_files(imgs_dir, conf, processes_num):
        # Read the files
        _, imgs = StackAligner.read_imgs(imgs_dir)

        aligner = StackAligner(conf, processes_num)
        return aligner.align_imgs(imgs)


def test_detector(section_dir, conf_fname, workers_num, files_num):
    conf = Stitcher.load_conf_from_file(conf_fname)
    img_fnames = glob.glob(os.path.join(section_dir, '000*', '*.bmp'))[:files_num]

    processes_factory = ProcessesFactory(conf)
    detector_params = conf.get('detector_params', {})
    matcher_params = conf.get('matcher_params', {})
    detector_in_queue = queue.Queue(maxsize=detector_params.get("queue_max_size", 0))
    detector_result_queue = queue.Queue(maxsize=matcher_params.get("queue_max_size", 0))

    # Set up the pool of detectors
    #detector_worker = ThreadWrapper(DetectorWorker, (processes_factory, detector_in_queue))
    detector_workers = [ThreadWrapper(DetectorWorker, (processes_factory, detector_in_queue, [detector_result_queue])) for i in range(workers_num)]

    for img_fname in img_fnames:
        # job = (matcher's result queue idx, tile fname, local area)
        job1 = (0, img_fname, np.array([150, 100]), None)
        #print('Submitting job to detector_in_queue')
        detector_in_queue.put(job1)

    for i in range(len(img_fnames)):
        #print('Fetching results from detector_result_queue')
        res = detector_result_queue.get()
        #print("Detector result:", res[0], len(res[2][0]))
    
    print("Closing all detectors")
    for i in range(workers_num):
        detector_in_queue.put(None)
    for i in range(workers_num):
        detector_workers[i].join()


if __name__ == '__main__':
    section_dir = '/n/home10/adisuis/Harvard/git/rh_aligner/tests/ECS_test9_cropped/images/010_S10R1/full_image_coordinates.txt'
    section_num = 10
    #section_dir = '/n/lichtmanfs2/100um_Sept2017/EM/w01h03/100umsept2017_20170912_17-52-13/181_S181R1/full_image_coordinates.txt'
    #section_num = 181
    #section_dir = '/n/lichtmanfs2/Alex/EM/ROI2_w04/W04_H04_ROI2_20180109_16-51-34/003_S3R1/full_image_coordinates.txt'
    #section_num = 3
    #section_dir = '/n/lichtmanfs2/Alex/EM/ROI2_w08/W08_H01_ROI2_20171227_00-07-19/010_S10R1/full_image_coordinates.txt'
    #section_num = 8_10
    conf_fname = '../../conf/conf_example.yaml'
    processes_num = 8
    out_fname = './output_stitched_sec{}_test_optimizer.json'.format(section_num)
# 
#     section = Section.create_from_full_image_coordinates(section_dir, section_num)
#     conf = Stitcher.load_conf_from_file(conf_fname)
#     stitcher = Stitcher(conf, processes_num)
#     stitcher.stitch_section(section) # will stitch and update the section
# 
#     # Save the transforms to file
#     import json
#     print('Writing output to: {}'.format(out_fname))
#     section.save_as_json(out_fname)
# #     img_fnames, imgs = StackAligner.read_imgs(imgs_dir)
# #     for img_fname, img, transform in zip(img_fnames, imgs, transforms):
# #         # assumption: the output image shape will be the same as the input image
# #         out_fname = os.path.join(out_path, os.path.basename(img_fname))
# #         img_transformed = cv2.warpAffine(img, transform[:2,:], (img.shape[1], img.shape[0]), flags=cv2.INTER_AREA)
# #         cv2.imwrite(out_fname, img_transformed)

# Testing
#    test_detector('/n/home10/adisuis/Harvard/git/rh_aligner/tests/ECS_test9_cropped/images/010_S10R1', conf_fname, 8, 500)

    logger.start_process('main', 'stitcher.py', [section_dir, conf_fname])
    section = Section.create_from_full_image_coordinates(section_dir, section_num)
    conf = Stitcher.load_conf_from_file(conf_fname)
    stitcher = Stitcher(conf)
    stitcher.stitch_section(section) # will stitch and update the section tiles' transformations

    # output the section
    out_tilespec = section.tilespec
    import json
    with open(out_fname, 'wt') as out_f:
        json.dump(out_tilespec, out_f, sort_keys=True, indent=4)

    del stitcher

    logger.end_process('main ending', rh_logger.ExitCode(0))

