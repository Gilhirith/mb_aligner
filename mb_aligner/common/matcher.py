import numpy as np
from . import ransac
from rh_renderer import models

class FeaturesMatcher(object):

    def __init__(self, matcher_init_fn, **kwargs):
        self._matcher = matcher_init_fn()

        self._params = {}
        # get default values if no value is present in kwargs
        #self._params["num_filtered_percent"] = kwargs.get("num_filtered_percent", 0.25)
        #self._params["filter_rate_cutoff"] = kwargs.get("filter_rate_cutoff", 0.25)
        self._params["ROD_cutoff"] = kwargs.get("ROD_cutoff", 0.92)
        self._params["min_features_num"] = kwargs.get("min_features_num", 40)

        # Parameters for the RANSAC
        self._params["model_index"] = kwargs.get("model_index", 3)
        self._params["iterations"] = kwargs.get("iterations", 5000)
        self._params["max_epsilon"] = kwargs.get("max_epsilon", 30.0)
        self._params["min_inlier_ratio"] = kwargs.get("min_inlier_ratio", 0.01)
        self._params["min_num_inlier"] = kwargs.get("min_num_inliers", 7)
        self._params["max_trust"] = kwargs.get("max_trust", 3)
        self._params["det_delta"] = kwargs.get("det_delta", None)
        self._params["max_stretch"] = kwargs.get("max_stretch", None)
        self._params["avoid_robust_filter"] = kwargs.get("avoid_robust_filter", False)
        self._params["max_rot_deg"] = kwargs.get("max_rot_deg", None)

        self._params["use_regularizer"] = True if "use_regularizer" in kwargs.keys() else False
        self._params["regularizer_lambda"] = kwargs.get("regularizer_lambda", 0.1)
        self._params["regularizer_model_index"] = kwargs.get("regularizer_model_index", 1)

        self._params["best_k_matches"] = kwargs.get("best_k_matches", 0) # 0 = all of the matches

        self._params["max_distance"] = kwargs.get("max_distance", None)

    def match(self, features_kps1, features_descs1, features_kps2, features_descs2):
        if features_descs1 is None or len(features_descs1) < self._params["min_features_num"] or features_descs2 is None or len(features_descs2) < self._params["min_features_num"]:
            return None
        matches = self._matcher.knnMatch(features_descs1, features_descs2, k=2)

        good_matches = []
        for m, n in matches:
            #if (n.distance == 0 and m.distance == 0) or (m.distance / n.distance < actual_params["ROD_cutoff"]):
            if m.distance < self._params["ROD_cutoff"] * n.distance:
                good_matches.append(m)

#         match_points = (
#             np.array([features_kps1[m.queryIdx].pt for m in good_matches]),
#             np.array([features_kps2[m.trainIdx].pt for m in good_matches]),
#             np.array([m.distance for m in good_matches])
#         )
        match_points = (
            features_kps1[[m.queryIdx for m in good_matches]],
            features_kps2[[m.trainIdx for m in good_matches]],
            np.array([m.distance for m in good_matches])
        )


        return match_points

    def match_and_filter(self, features_kps1, features_descs1, features_kps2, features_descs2):
        match_points = self.match(features_kps1, features_descs1, features_kps2, features_descs2)

        if match_points is None:
            return None, None

        model, filtered_matches, mask = ransac.filter_matches(match_points, match_points, self._params['model_index'],
                    self._params['iterations'], self._params['max_epsilon'], self._params['min_inlier_ratio'],
                    self._params['min_num_inlier'], self._params['max_trust'], self._params['det_delta'], self._params['max_stretch'],
                    self._params['max_rot_deg'],
                    robust_filter=not self._params['avoid_robust_filter'], max_distance=self._params['max_distance'])

        if model is None:
            return None, None

        if self._params["use_regularizer"]:
            regularizer_model, _, _ = ransac.filter_matches(match_points, match_points, self._params['regularizer_model_index'],
                        self._params['iterations'], self._params['max_epsilon'], self._params['min_inlier_ratio'],
                        self._params['min_num_inlier'], self._params['max_trust'], self._params['det_delta'], self._params['max_stretch'],
                        self._params['max_rot_deg'],
                        robust_filter=not self._params['avoid_robust_filter'], max_distance=self._params['max_distance'])

            if regularizer_model is None:
                return None, None

            result = model.get_matrix() * (1 - self._params["regularizer_lambda"]) + regularizer_model.get_matrix() * self._params["regularizer_lambda"]
            model = models.AffineModel(result)

        if self._params['best_k_matches'] > 0 and self._params['best_k_matches'] < len(filtered_matches[0]):
            # Only keep the best K matches out of the filtered matches
            best_k_matches_idxs = np.argpartition(match_points[2][mask], -self._params['best_k_matches'])[-self._params['best_k_matches']:]
            filtered_matches = np.array([match_points[0][mask][best_k_matches_idxs], match_points[1][mask][best_k_matches_idxs]])

        return model, filtered_matches

