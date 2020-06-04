import numpy as np
import copy
import rh_renderer.models as models
from scipy.misc import comb
from rh_logger.api import logger
import logging
import math

import pyximport
pyximport.install()
from mb_aligner.common import ransac_cython

def array_to_string(arr):
    return arr.tostring()
    #return '_'.join(map(str, arr))

def tri_area(p1, p2, p3):
    scalar = p1.ndim == 1
    p1 = np.atleast_2d(p1)
    p2 = np.atleast_2d(p2)
    p3 = np.atleast_2d(p3)
        
    area = (p1[:, 0]*(p2[:, 1] - p3[:, 1]) +
            p2[:, 0]*(p3[:, 1] - p1[:, 1]) + 
            p3[:, 0]*(p1[:, 1] - p2[:, 1])) / 2.0
    # area might be negative
    return area[0] if scalar else area

def choose_forward(n, k, n_draws):
    '''Choose k without replacement from among N

    :param n: number of samples to choose from
    :param k: number of samples to choose
    :param n_draws: number of tuples to return
    
    returns an n_draws by k array of k-tuples
    '''
    if n == 0:
        return np.zeros((0, k), int)
    max_combinations = comb(n, k)
    if max_combinations / 3 < n_draws:
        return choose_forward_dense(n, k, n_draws)
    else:
        return choose_forward_sparse(n, k, n_draws)

def enumerate_choices(n, k):
    '''Enumerate all the ways to choose k from n

    returns choices sorted lexigraphically, e.g.
    0, 1
    0, 2
    1, 2
    '''
    if k == 1:
        return np.arange(n).reshape(n, 1)
    #
    # Enumerate ways to choose k-1 from n-1 (there are no ways
    # to choose the last, so n-1)
    last = enumerate_choices(n-1, k-1)
    #
    # number of possible choices for each of the previous
    # is from among the remaining.
    #
    n_choices = n - 1 - last[:, -1]
    index = np.hstack([[0], np.cumsum(n_choices)])
    #
    # allocate memory for the result
    #
    result = np.zeros((index[-1], k), int)
    #
    # Create a back pointer into "last" for each element of the new array
    #
    back_ptr = np.zeros(result.shape[0], int)
    back_ptr[index[:-1]] = 1
    back_ptr = np.cumsum(back_ptr) - 1
    #
    # Broadcast the elements of the old array into the new one
    # using the back pointer
    result[:, :-1] = last[back_ptr, :]
    #
    # pull a cumsum trick: fill the last column with all "1" except
    # for the first element which is - its place in the array
    #
    result[1:, -1] = 1
    #
    # Then we subtract the number of entries to get back to zero
    result[index[1:-1], -1] = -n_choices[:-1]+1
    #
    # The last result has to start at the next to last + 1
    # 0, 1 <-
    # 0, 2
    # 1, 2 <-
    result[:, -1] = np.cumsum(result[:, -1]) + result[:, -2] + 1
    return result
    
def choose_forward_dense(n, k, n_draws):
    '''Choose k without replacement from among N where n_draws ~ # of combos

    :param n: number of samples to choose from
    :param k: number of samples to choose
    :param n_draws: number of tuples to return
    
    returns an n_draws by k array of k-tuples
    '''
    all_possible = enumerate_choices(n, k)
    choices = np.random.choice(np.arange(all_possible.shape[0]), n_draws, 
                               replace=False)
    return all_possible[choices]

def choose_forward_sparse(n, k, n_draws):
    '''Choose k without replacement from among N where n_draws << combos
    
    :param n: number of samples to choose from
    :param k: number of samples to choose
    :param n_draws: number of tuples to return
    
    returns an n_draws by k array of k-tuples
    '''
    #
    # We assume that there is very little chance of collisions
    # and we choose a few more than asked
    #
    extra = int(np.sqrt(n_draws)) + 1
    n1_draws = n_draws + extra
    choices = np.random.randint(0, n, (n1_draws, k))
    while True:
        #
        # We sort in the k direction to get indices from low to high per draw
        #
        choices.sort(axis=1)
        #
        # We then argsort and duplicates should be adjacent in argsortland
        #
        order = np.lexsort([choices[:, k_] for k_ in range(k-1, -1, -1)])
        to_remove = np.where(
            np.all(choices[order[:-1]] == choices[order[1:]], axis=1))
        result = np.delete(choices, order[to_remove], axis=0)
        if len(result) >= n_draws:
            return result
        #
        # Add some more choices if we didn't get enough
        #
        choices = np.vstack((result, np.random.randint(0, n, (extra, k))))
    
def check_model_stretch(model_matrix, max_stretch=0.25):
    # Use the eigen values to validate the stretch
    assert(max_stretch >= 0.0 and max_stretch <= 1.0)
    eig_vals, _ = np.linalg.eig(model_matrix)
    # Note that this also takes flipping as an incorrect transformation
    valid_eig_vals = [eig_val for eig_val in eig_vals if eig_val >= 1.0 - max_stretch and eig_val <= 1.0 + max_stretch]
    return len(valid_eig_vals) == 2

def filter_triangles(m0, m1, choices, 
                     max_stretch=0.25, 
                     max_area=.3):
    '''Filter a set of match choices
    
    :param m0: set of points in one domain
    :param m1: set of matching points to m0 in another domain
    :param choices: an N x 3 array of triangles
    :param max_stretch: filter out a choice if it shrinks by 1-max_stretch
        or stretches by 1+max_stretch
    :param max_area: filter out a choice if the area of m1's triangle is
        less than 1-max_area or more than 1+max_area

    If a triangle in m0 has a different absolute area than in m1, exclude it
    If the eigenvalues of the affine transform array indicate a shrink of
    a factor of 1-max_stretch or a stretch of a factor of 1+max_stretch,
    exclude
    '''
    pt1a, pt2a, pt3a = [m0[choices][:, _, :] for _ in range(3)]
    pt1b, pt2b, pt3b = [m1[choices][:, _, :] for _ in range(3)]
    areas_a = tri_area(pt1a, pt2a, pt3a)
    areas_b = tri_area(pt1b, pt2b, pt3b)
    area_ratio = areas_a / (areas_b + np.finfo(areas_b.dtype).eps)
    mask = (area_ratio <= 1+max_area) & (area_ratio >= 1-max_area)
    return choices[mask]



def ransac(sample_matches, test_matches, target_model_type, iterations, epsilon, min_inlier_ratio, min_num_inlier, det_delta=0.55, max_stretch=None, max_rot_deg=None, tri_angles_comparator=None, max_distance=None):
    # model = Model.create_model(target_model_type)
    assert(len(sample_matches[0]) == len(sample_matches[1]))

    proposed_model = models.Transforms.create(target_model_type)
       
    max_rot_deg_cos = None
    if max_rot_deg is not None:
        max_rot_deg_cos = math.cos(max_rot_deg * math.pi / 180.0)
        #print("max_rot_deg: {},  max_rot_deg_cos: {}, {}".format(max_rot_deg, max_rot_deg_cos, max_rot_deg * math.pi / 180.0))

    if proposed_model.MIN_MATCHES_NUM > sample_matches[0].shape[0]:
        logger.report_event("RANSAC cannot find a good model because the number of initial matches ({}) is too small.".format(sample_matches[0].shape[0]), log_level=logging.WARN)
        return None, None, None

    if target_model_type == 0:
        # Translation model, use cython to improve speed
        res_status, res_model_params, res_inliers_mask = ransac_cython.ransac_translation(
                #np.array([pts1.T, pts2.T]), np.array([pts1.T, pts2.T]),
                [sample_matches[0].astype(np.float32), sample_matches[1].astype(np.float32)],
                [test_matches[0].astype(np.float32), test_matches[1].astype(np.float32)],
                int(iterations),
                float(epsilon),
                float(min_inlier_ratio),
                int(min_num_inlier)
            )

        if res_status != 0: # failure
            return None, None, None
            

        best_model = models.TranslationModel((res_model_params[0], res_model_params[1]))
        best_model_mean_dists = 0

        return res_inliers_mask, best_model, best_model_mean_dists
 
    elif target_model_type == 1:
        # Rigid model, use cython to improve speed
        if max_rot_deg_cos is None:
            max_rot_deg_cos = 0
        res_status, res_model_params, res_inliers_mask = ransac_cython.ransac_rigid(
                #np.array([pts1.T, pts2.T]), np.array([pts1.T, pts2.T]),
                [sample_matches[0].astype(np.float32), sample_matches[1].astype(np.float32)],
                [test_matches[0].astype(np.float32), test_matches[1].astype(np.float32)],
                int(iterations),
                float(epsilon),
                float(min_inlier_ratio),
                int(min_num_inlier),
                float(max_rot_deg_cos)
            )

        if res_status != 0: # failure
            return None, None, None
            

        best_model = models.RigidModel(res_model_params[0], (res_model_params[1], res_model_params[2]))
        best_model_mean_dists = 0

        return res_inliers_mask, best_model, best_model_mean_dists
 
    best_model = None
    best_model_score = 0 # The higher the better
    best_inlier_mask = None
    best_model_mean_dists = 0

    # Avoiding repeated indices permutations using a dictionary
    # Limit the number of possible matches that we can search for using n choose k
    max_combinations = int(comb(len(sample_matches[0]), proposed_model.MIN_MATCHES_NUM))
    max_iterations = min(iterations, max_combinations)
    choices = choose_forward(len(sample_matches[0]),
                             proposed_model.MIN_MATCHES_NUM,
                             max_iterations)
    if proposed_model.MIN_MATCHES_NUM == 3:
        choices = filter_triangles(sample_matches[0], sample_matches[1], choices)
    if max_distance is not None:
        # estimate the center of each section by looking at the "central" feature location
        sec1_min_xy = np.min(test_matches[0], axis=0)
        sec1_max_xy = np.max(test_matches[0], axis=0)
        sec2_min_xy = np.min(test_matches[1], axis=0)
        sec2_max_xy = np.max(test_matches[1], axis=0)
        sec1_center = (sec1_min_xy + sec1_max_xy) / 2.0
        sec2_center = (sec2_min_xy + sec2_max_xy) / 2.0
        max_distance2 = max_distance**2
    for min_matches_idxs in choices:
        # Try to fit them to the model
        if proposed_model.fit(sample_matches[0][min_matches_idxs], sample_matches[1][min_matches_idxs]) == False:
            continue
        model_matrix = proposed_model.get_matrix()[:2, :2]
        if abs(model_matrix[0, 0]) < 0.01:
            continue
        if max_rot_deg_cos is not None and proposed_model.MIN_MATCHES_NUM == 2:
            if model_matrix[0, 0] < max_rot_deg_cos:
                continue
        if proposed_model.MIN_MATCHES_NUM == 3:
            # check the stretch of the new transformation
            if max_stretch is not None and not check_model_stretch(model_matrix, max_stretch):
                continue
            # if the proposed model distorts the image too much, skip the model
            if det_delta is not None:
                det = np.linalg.det(model_matrix)
                if det < 1.0 - det_delta or det > 1.0 + det_delta:
                    continue

            # Compare the triangle angles
            if tri_angles_comparator is not None:
                if tri_angles_comparator(proposed_model) is False:
                    continue
        # Check what's the "translational" distance of the given transformation
        if max_distance is not None:
            sec1_center_transformed = proposed_model.apply(sec1_center)[0]
            centers_dist2 = np.sum((sec2_center - sec1_center_transformed)**2)
            if centers_dist2 > max_distance2:
                continue
        # print "proposed_model", proposed_model.to_str()
        # Verify the new model 
        proposed_model_score, inlier_mask, proposed_model_mean = proposed_model.score(test_matches[0], test_matches[1], epsilon, min_inlier_ratio, min_num_inlier)
        # print "proposed_model_score", proposed_model_score
        if proposed_model_score > best_model_score:
            best_model = copy.deepcopy(proposed_model)
            best_model_score = proposed_model_score
            best_inlier_mask = inlier_mask
            best_model_mean_dists = proposed_model_mean
    '''
    if best_model is None:
        print("Cannot find a good model during ransac. best_model_score {}".format(best_model_score))
    else:
        print("RANSAC result: best_model_score", best_model_score, "best_model:", best_model.to_str(), "best_model_mean_dists:", best_model_mean_dists)
    '''
    return best_inlier_mask, best_model, best_model_mean_dists


def filter_after_ransac(candidates, model, max_trust, min_num_inliers):
    """
    Estimate the AbstractModel and filter potential outliers by robust iterative regression.
    This method performs well on data sets with low amount of outliers (or after RANSAC).
    """
    # copy the model
    new_model = copy.deepcopy(model)
    dists = []

    # iteratively find a new model, by fitting the candidates, and removing those that are far than max_trust*median-distance
    # until the set of remaining candidates does not change its size

    # for the initial iteration, we set a value that is higher the given candidates size
    prev_iteration_num_inliers = candidates.shape[1] + 1

    # keep a copy of the candidates that will be changed due to fitting and error 
    inliers = copy.copy(candidates[0])

    # keep track of the candidates using a mask
    candidates_mask = np.ones((candidates.shape[1]), dtype=np.bool)

    while prev_iteration_num_inliers > np.sum(candidates_mask):
        prev_iteration_num_inliers = np.sum(candidates_mask)
        # Get the inliers and their corresponding matches
        inliers = candidates[0][candidates_mask]
        to_image_candidates = candidates[1][candidates_mask]

        # try to fit the model
        if new_model.fit(inliers, to_image_candidates) == False:
            break

        # get the meidan error (after transforming the points)
        pts_after_transform = new_model.apply(inliers)
        dists = np.sqrt(np.sum((pts_after_transform - to_image_candidates) ** 2, axis=1))
        median = np.median(dists)
        # print "dists mean", np.mean(dists)
        # print "median", median
        # print dists <= (median * max_trust)
        inliers_mask = dists <= (median * max_trust)
        candidates_mask[candidates_mask == True] = inliers_mask


    if np.sum(candidates_mask) < min_num_inliers:
        return None, None, -1

    return new_model, candidates_mask, np.mean(dists)


def filter_matches(sample_matches, test_matches, target_model_type, iterations, epsilon, min_inlier_ratio, min_num_inlier, max_trust, det_delta=0.35, max_stretch=None, max_rot_deg=None, robust_filter=True, tri_angles_comparator=None, max_distance=None):
    """Perform a RANSAC filtering given all the matches"""
    new_model = None
    filtered_matches = None
    meandists = -1

    # Apply RANSAC
    # print "Filtering {} matches".format(matches.shape[1])
    logger.report_event("pre-ransac matches count: sample={}, test={}".format(len(sample_matches[0]), len(test_matches[0])), log_level=logging.DEBUG)
    inliers_mask, model, _ = ransac(sample_matches, test_matches, target_model_type, iterations, epsilon, min_inlier_ratio, min_num_inlier, det_delta, max_stretch, max_rot_deg, tri_angles_comparator=tri_angles_comparator, max_distance=max_distance)
    if inliers_mask is None:
        logger.report_event("post-ransac matches count: 0", log_level=logging.DEBUG)
    else:
        logger.report_event("post-ransac matches count: {}".format(np.sum(inliers_mask)), log_level=logging.DEBUG)

    if not robust_filter:
        inliers = np.array([test_matches[0][inliers_mask], test_matches[1][inliers_mask]])
        return model, inliers, inliers_mask

    # Apply further filtering
    if inliers_mask is not None:
        inliers = np.array([test_matches[0][inliers_mask], test_matches[1][inliers_mask]])
        # print "Found {} good matches out of {} matches after RANSAC".format(inliers.shape[1], matches.shape[1])
        new_model, filtered_inliers_mask, meandists = filter_after_ransac(inliers, model, max_trust, min_num_inlier)
        filtered_matches = np.array([inliers[0][filtered_inliers_mask], inliers[1][filtered_inliers_mask]])
    '''
    if new_model is None:
        print "No model found after RANSAC"
    else:
        # _, filtered_matches_mask, mean_val = new_model.score(matches[0], matches[1], epsilon, min_inlier_ratio, min_num_inlier)
        # filtered_matches = np.array([matches[0][filtered_matches], matches[1][filtered_matches]])
        print "Model found after robust regression: {}, applies to {} out of {} matches.".format(new_model.to_str(), filtered_matches.shape[1], matches.shape[1])
    '''
    if filtered_matches is None:
        logger.report_event("post-ransac-filter matches count: 0", log_level=logging.DEBUG)
        return None, None, None
    else:
        logger.report_event("post-ransac-filter matches count: {}".format(filtered_matches.shape[1]), log_level=logging.DEBUG)
        inliers_mask[inliers_mask == True][filtered_inliers_mask == False] = False # clear everrything that's not set by filtered_inliers_mask
        return new_model, filtered_matches, inliers_mask


def decompose_affine_matrix(m):
    """
    Decomposing the above using: https://math.stackexchange.com/questions/78137/decomposition-of-a-nonsquare-affine-matrix (first answer)
    the decomposition of the 2*2 matrix (w/o translation) is as follows:
    
    [[ a, b],   =    [[p, 0],  *  [[1, 0],  *  [[cosT, sinT], 
     [ d, e]]         [0, r]]      [q, 1]]      [-sinT, cosT]]
    
    2*2 matrix  =    scale     *   shear    *   rotation
    """
    m = m[:2, :2]
    p = math.sqrt(m[0, 0]**2 + m[0, 1]**2)
    r = (m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])/p
    q = (m[0, 0] * m[1, 0] + m[0, 1] * m[1, 1])/(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])
    theta = math.atan2(m[1, 0], m[0, 0])

    rot_mat = np.array([[math.cos(theta), -math.sin(theta)],
                        [math.sin(theta), math.cos(theta)]])
    shear_mat = np.array([[1, 0],
                          [q, 1]])
    scale_mat = np.array([[p, 0],
                          [0, r]])
    return scale_mat, shear_mat, rot_mat, np.dot(scale_mat, np.dot(shear_mat, rot_mat))
