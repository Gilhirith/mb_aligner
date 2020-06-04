import numpy as np
import time
import ransac as orig_ransac

#import pstats, cProfile
#import line_profiler


import pyximport
pyximport.install()
import ransac_cython

def test1():
    print("**** Testing normal distribution with 500 points:")
    N = 500
    min_val = -2000
    max_val = 3000
    np.random.seed(7)
    pts1 = np.random.random_sample((N, 2)).astype(np.float32) * (max_val - min_val) + min_val

    # randomize theta, t_x, t_y
    theta = np.float32(np.random.random_sample() * (0.1 - (-0.1)) + (-0.1))
    t_x, t_y = np.random.random_sample(2).astype(np.float32) * (max_val - min_val) + min_val

    print("Actual transformation: Theta {}, Tx {}, Ty {}".format(theta, t_x, t_y))

    # apply the transformation 
    trans_mat = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ], dtype=np.float32)
    pts2 = np.dot(trans_mat, pts1.T).T + np.array([t_x, t_y])

    # Add some noise
    pts2 += np.random.normal(scale=3, size=(N, 2))

    iterations = 100
    epsilon = 5.0
    min_inlier_ratio = 0.05
    min_num_inlier = 0.1 * N
    max_rot_deg = 0.1
#     out = ransac_cython.ransac(
#             [pts1, pts2], [pts1, pts2],
#             iterations,
#             epsilon,
#             min_inlier_ratio,
#             min_num_inlier,
#             0, # det_delta
#             0, # max_stretch
#             max_rot_deg,
#             0, # tri_angles_comparator
#         )
    st_time = time.time()
    out = ransac_cython.ransac_rigid(
            #np.array([pts1.T, pts2.T]), np.array([pts1.T, pts2.T]),
            [pts1, pts2], [pts1, pts2],
            iterations,
            epsilon,
            min_inlier_ratio,
            min_num_inlier,
            max_rot_deg
        )
    end_time = time.time()


    print("Output is: {}, time: {} seconds".format(out[1], end_time - st_time))

def test2():
    print("**** Testing 50% outliers with 500 points:")
    N = 500
    min_val = -2000
    max_val = 3000
    np.random.seed(7)
    pts1 = np.random.random_sample((N, 2)).astype(np.float32) * (max_val - min_val) + min_val

    # randomize theta, t_x, t_y
    theta = np.float32(np.random.random_sample() * (0.1 - (-0.1)) + (-0.1))
    t_x, t_y = np.random.random_sample(2).astype(np.float32) * (max_val - min_val) + min_val

    print("Actual transformation: Theta {}, Tx {}, Ty {}".format(theta, t_x, t_y))

    # apply the transformation 
    trans_mat = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ], dtype=np.float32)
    pts2 = np.dot(trans_mat, pts1.T).T + np.array([t_x, t_y])

    # Add some noise
    outlier_mask = np.random.choice(N, int(N/2), replace=False)
    pts2[outlier_mask] += np.random.normal(scale=30, size=(len(outlier_mask), 2)) + 10

    iterations = 100
    epsilon = 5.0
    min_inlier_ratio = 0.05
    min_num_inlier = 0.1 * N
    max_rot_deg = 0.1
    st_time = time.time()
#     out = ransac_cython.ransac(
#             [pts1, pts2], [pts1, pts2],
#             iterations,
#             epsilon,
#             min_inlier_ratio,
#             min_num_inlier,
#             0, # det_delta
#             0, # max_stretch
#             max_rot_deg,
#             0, # tri_angles_comparator
#         )
    out = ransac_cython.ransac_rigid(
            #np.array([pts1.T, pts2.T]), np.array([pts1.T, pts2.T]),
            [pts1, pts2], [pts1, pts2],
            iterations,
            epsilon,
            min_inlier_ratio,
            min_num_inlier,
            max_rot_deg
        )
    end_time = time.time()


    print("Output is: {}, time: {} seconds".format(out[1], end_time - st_time))


def test3():
    print("**** Testing number of matches is much smaller than iterations:")
    N = 100
    min_val = -2000
    max_val = 3000
    np.random.seed(7)
    pts1 = np.random.random_sample((N, 2)).astype(np.float32) * (max_val - min_val) + min_val

    # randomize theta, t_x, t_y
    theta = np.float32(np.random.random_sample() * (0.1 - (-0.1)) + (-0.1))
    t_x, t_y = np.random.random_sample(2).astype(np.float32) * (max_val - min_val) + min_val

    print("Actual transformation: Theta {}, Tx {}, Ty {}".format(theta, t_x, t_y))

    # apply the transformation 
    trans_mat = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ], dtype=np.float32)
    pts2 = np.dot(trans_mat, pts1.T).T + np.array([t_x, t_y])

    # Add some noise
    pts2 += np.random.normal(scale=3, size=(N, 2))

    iterations = 100000
    epsilon = 5.0
    min_inlier_ratio = 0.05
    min_num_inlier = 0.1 * N
    max_rot_deg = 0.1
    st_time = time.time()
#     out = ransac_cython.ransac(
#             [pts1, pts2], [pts1, pts2],
#             iterations,
#             epsilon,
#             min_inlier_ratio,
#             min_num_inlier,
#             0, # det_delta
#             0, # max_stretch
#             max_rot_deg,
#             0, # tri_angles_comparator
#         )
    out = ransac_cython.ransac_rigid(
            #np.array([pts1.T, pts2.T]), np.array([pts1.T, pts2.T]),
            [pts1, pts2], [pts1, pts2],
            iterations,
            epsilon,
            min_inlier_ratio,
            min_num_inlier,
            max_rot_deg
        )
    end_time = time.time()


    print("Output is: {}, time: {} seconds".format(out[1], end_time - st_time))


def test4():
    print("**** Testing no-noise with 500 points:")
    N = 500
    min_val = -2000
    max_val = 3000
    np.random.seed(7)
    pts1 = np.random.random_sample((N, 2)).astype(np.float32) * (max_val - min_val) + min_val

    # randomize theta, t_x, t_y
    theta = np.float32(np.random.random_sample() * (0.1 - (-0.1)) + (-0.1))
    t_x, t_y = np.random.random_sample(2).astype(np.float32) * (max_val - min_val) + min_val

    print("Actual transformation: Theta {}, Tx {}, Ty {}".format(theta, t_x, t_y))

    # apply the transformation 
    trans_mat = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ], dtype=np.float32)
    pts2 = np.dot(trans_mat, pts1.T).T + np.array([t_x, t_y])

    iterations = 100
    epsilon = 5.0
    min_inlier_ratio = 0.05
    min_num_inlier = 0.1 * N
    max_rot_deg = 0.1
    st_time = time.time()
#     out = ransac_cython.ransac(
#             [pts1, pts2], [pts1, pts2],
#             iterations,
#             epsilon,
#             min_inlier_ratio,
#             min_num_inlier,
#             0, # det_delta
#             0, # max_stretch
#             max_rot_deg,
#             0, # tri_angles_comparator
#         )
    out = ransac_cython.ransac_rigid(
            #np.array([pts1.T, pts2.T]), np.array([pts1.T, pts2.T]),
            [pts1, pts2], [pts1, pts2],
            iterations,
            epsilon,
            min_inlier_ratio,
            min_num_inlier,
            max_rot_deg
        )
    end_time = time.time()


    print("Output is: {}, time: {} seconds".format(out[1], end_time - st_time))



def test5():
    print("**** Testing normal distribution with 50000 points:")
    N = 50000
    min_val = -2000
    max_val = 3000
    np.random.seed(7)
    pts1 = np.random.random_sample((N, 2)).astype(np.float32) * (max_val - min_val) + min_val

    # randomize theta, t_x, t_y
    theta = np.float32(np.random.random_sample() * (0.1 - (-0.1)) + (-0.1))
    t_x, t_y = np.random.random_sample(2).astype(np.float32) * (max_val - min_val) + min_val

    print("Actual transformation: Theta {}, Tx {}, Ty {}".format(theta, t_x, t_y))

    # apply the transformation 
    trans_mat = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ], dtype=np.float32)
    pts2 = np.dot(trans_mat, pts1.T).T + np.array([t_x, t_y])

    # Add some noise
    pts2 += np.random.normal(scale=3, size=(N, 2))

    iterations = 100
    epsilon = 5.0
    min_inlier_ratio = 0.05
    min_num_inlier = 0.1 * N
    max_rot_deg = 0.1


    out = None
    st_time = time.time()

#     func = ransac_cython.ransac_rigid
#     profile = line_profiler.LineProfiler(func)
#     profile.runcall(func, np.array([pts1.T, pts2.T]), np.array([pts1.T, pts2.T]),
#              iterations,
#              epsilon,
#              min_inlier_ratio,
#              min_num_inlier,
#              max_rot_deg)

#     cProfile.runctx("ransac_cython.ransac_rigid(\
#         np.array([pts1.T, pts2.T]), np.array([pts1.T, pts2.T]),\
#         iterations,\
#         epsilon,\
#         min_inlier_ratio,\
#         min_num_inlier,\
#         max_rot_deg\
#         )", globals(), locals(), "Profile.prof")

    out = ransac_cython.ransac_rigid(
            #np.array([pts1.T, pts2.T]), np.array([pts1.T, pts2.T]),
            [pts1, pts2], [pts1, pts2],
            iterations,
            epsilon,
            min_inlier_ratio,
            min_num_inlier,
            max_rot_deg
        )
    end_time = time.time()


    print("Output is: {}, time: {} seconds".format(out[1], end_time - st_time))

#     assert_stats(profile, func.__name__)

#     s = pstats.Stats("Profile.prof")
#     s.strip_dirs().sort_stats("time").print_stats()

    print("Running original ransac")
    target_model_type = 1
    st_time = time.time()
    out = orig_ransac.ransac(np.array([pts1, pts2]), np.array([pts1, pts2]),
        target_model_type,
        iterations,
        epsilon,
        min_inlier_ratio,
        min_num_inlier,
        det_delta=None,
        max_stretch=None,
        max_rot_deg=None,
        tri_angles_comparator=None)
    end_time = time.time()

    print("Output (original ransac) is: {}, time: {} seconds".format(out[1].to_str(), end_time - st_time))


def test6():
    print("**** Testing normal distribution with 50000 points, 50000 iterations:")
    N = 50000
    min_val = -2000
    max_val = 3000
    np.random.seed(7)
    pts1 = np.random.random_sample((N, 2)).astype(np.float32) * (max_val - min_val) + min_val

    # randomize theta, t_x, t_y
    theta = np.float32(np.random.random_sample() * (0.1 - (-0.1)) + (-0.1))
    t_x, t_y = np.random.random_sample(2).astype(np.float32) * (max_val - min_val) + min_val

    print("Actual transformation: Theta {}, Tx {}, Ty {}".format(theta, t_x, t_y))

    # apply the transformation 
    trans_mat = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ], dtype=np.float32)
    pts2 = np.dot(trans_mat, pts1.T).T + np.array([t_x, t_y])

    # Add some noise
    pts2 += np.random.normal(scale=3, size=(N, 2))

    iterations = 50000
    epsilon = 5.0
    min_inlier_ratio = 0.05
    min_num_inlier = 0.1 * N
    max_rot_deg = 0.1


    out = None
    st_time = time.time()

#     func = ransac_cython.ransac_rigid
#     profile = line_profiler.LineProfiler(func)
#     profile.runcall(func, np.array([pts1.T, pts2.T]), np.array([pts1.T, pts2.T]),
#              iterations,
#              epsilon,
#              min_inlier_ratio,
#              min_num_inlier,
#              max_rot_deg)

#     cProfile.runctx("ransac_cython.ransac_rigid(\
#         np.array([pts1.T, pts2.T]), np.array([pts1.T, pts2.T]),\
#         iterations,\
#         epsilon,\
#         min_inlier_ratio,\
#         min_num_inlier,\
#         max_rot_deg\
#         )", globals(), locals(), "Profile.prof")

    out = ransac_cython.ransac_rigid(
            #np.array([pts1.T, pts2.T]), np.array([pts1.T, pts2.T]),
            [pts1, pts2], [pts1, pts2],
            iterations,
            epsilon,
            min_inlier_ratio,
            min_num_inlier,
            max_rot_deg
        )
    end_time = time.time()


    print("Output is: {}, time: {} seconds".format(out[1], end_time - st_time))

#     assert_stats(profile, func.__name__)

#     s = pstats.Stats("Profile.prof")
#     s.strip_dirs().sort_stats("time").print_stats()

    print("Running original ransac - Very slow, uncomment if needed")
#     target_model_type = 1
#     st_time = time.time()
#     out = orig_ransac.ransac(np.array([pts1, pts2]), np.array([pts1, pts2]),
#         target_model_type,
#         iterations,
#         epsilon,
#         min_inlier_ratio,
#         min_num_inlier,
#         det_delta=None,
#         max_stretch=None,
#         max_rot_deg=None,
#         tri_angles_comparator=None)
#     end_time = time.time()
# 
#     print("Output (original ransac) is: {}, time: {} seconds".format(out[1].to_str(), end_time - st_time))



def assert_stats(profile, name):
    profile.print_stats()
    stats = profile.get_stats()
    assert len(stats.timings) > 0, "No profile stats."
    for key, timings in stats.timings.items():
        if key[-1] == name:
            assert len(timings) > 0
            break
    else:
        raise ValueError("No stats for %s." % name)

if __name__ == '__main__':
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()

