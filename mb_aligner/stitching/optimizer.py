from enum import Enum
from mb_aligner.stitching.optimize_translation_damping_2d_tiles import TranslationDamping2DOptimizer
from mb_aligner.stitching.optimize_rigid_2d_tiles import Rigid2DOptimizer

class Optimizer2D(object):

    class Type(Enum):
        RIGID = 1
        TRANSLATION_DAMPING = 2

    def __init__(self, optimizer_type_name, **kwargs):
        optimizer_type = Optimizer2D.Type[optimizer_type_name]
        if optimizer_type == Optimizer2D.Type.RIGID:
            self._optimizer2d = Rigid2DOptimizer(**kwargs)
        elif optimizer_type == Optimizer2D.Type.TRANSLATION_DAMPING:
            self._optimizer2d = TranslationDamping2DOptimizer(**kwargs)
        else:
            raise("Unknown feature detector algorithm given")


    def optimize(self, *args, **kwargs):
        return self._optimizer2d.optimize(*args, **kwargs)

