from mb_aligner.common.detector import FeaturesDetector
from mb_aligner.common.matcher import FeaturesMatcher
from mb_aligner.stitching.optimizer import Optimizer2D

class ProcessesFactory(object):
    '''
    A factory for the processes (algorithms) of the stitching and alignment pipeline.
    '''

    def __init__(self, conf):
        self._conf = conf

    def create_2d_detector(self):
        detector_params = self._conf.get('detector_params', {})
        return FeaturesDetector(self._conf['detector_type'], **detector_params)

    def create_2d_matcher(self):
        matcher_params = self._conf.get('matcher_params', {})
        # The matcher type is actually determined by the detector type (the detector's output)
        matcher_init_fn = FeaturesDetector.get_matcher_init_fn(self._conf['detector_type'])
        return FeaturesMatcher(matcher_init_fn, **matcher_params)

    def create_2d_optimizer(self):
        optimizer_params = self._conf.get('optimizer_params', {})
        return Optimizer2D(self._conf['optimizer_type'], **optimizer_params)

