# Taken from https://github.com/menpo/menpo/blob/master/menpo/transform/thinplatesplines.py

import numpy as np
#from .base import Transform, Alignment, Invertible
from .rbf import R2LogR2RBF
import scipy.linalg as la
import logging


# Note we inherit from Alignment first to get it's n_dims behavior
#class ThinPlateSplines(Alignment, Transform, Invertible):
class ThinPlateSplines(object):
    SVD_RANDOM_POINTS_THRESHOLD = 2000

    r"""
    The thin plate splines (TPS) alignment between 2D `source` and `target`
    landmarks.

    ``kernel`` can be used to specify an alternative kernel function. If
    ``None`` is supplied, the :class:`R2LogR2RBF` kernel will be used.

    Parameters
    ----------
    source : ``(N, 2)`` `ndarray`
        The source points to apply the tps from
    target : ``(N, 2)`` `ndarray`
        The target points to apply the tps to
    kernel : :class:`menpo.transform.rbf.RadialBasisFunction`, optional
        The kernel to apply.
    min_singular_val : `float`, optional
        If the target has points that are nearly coincident, the coefficients
        matrix is rank deficient, and therefore not invertible. Therefore, we
        only take the inverse on the full-rank matrix and drop any singular
        values that are less than this value (close to zero).

    Raises
    ------
    ValueError
        TPS is only with on 2-dimensional data
    """
    def __init__(self, source_points, target_points, kernel=None, min_singular_val=1e-4):
        logging.basicConfig(level=logging.DEBUG)
        self.source_points = source_points
        self.target_points = target_points
        self.n_points, self.n_dims = source_points.shape
        #Alignment.__init__(self, source, target)
        if self.n_dims != 2:
            raise ValueError('TPS can only be used on 2D data.')
        if kernel is None:
            kernel = R2LogR2RBF(source_points)
        self.min_singular_val = min_singular_val
        self.kernel = kernel
        logging.log(logging.DEBUG, "TPS c'tor 1 populate done")
        # k[i, j] is the rbf weighting between source i and j
        # (of course, k is thus symmetrical and it's diagonal nil)
        self.k = self.kernel.apply(self.source_points)
        logging.log(logging.DEBUG, "TPS c'tor 2 applied kernel on source points done")
        # p is a homogeneous version of the source points
        self.p = np.concatenate(
            [np.ones([self.n_points, 1]), self.source_points], axis=1)
        o = np.zeros([3, 3])
        top_l = np.concatenate([self.k, self.p], axis=1)
        bot_l = np.concatenate([self.p.T, o], axis=1)
        self.l = np.concatenate([top_l, bot_l], axis=0)
        self.v, self.y, self.coefficients = None, None, None
        logging.log(logging.DEBUG, "TPS c'tor 3 basic concatenation done")
        self._build_coefficients()

    def _build_coefficients(self):
        logging.log(logging.DEBUG, "TPS build_coefficents 1 starting")
        self.v = self.target_points.T.copy()
        logging.log(logging.DEBUG, "TPS build_coefficents 2 target_points copy done")
        self.y = np.hstack([self.v, np.zeros([2, 3])])
        logging.log(logging.DEBUG, "TPS build_coefficents 3 hstack done, self.l.shape: {}".format(self.l.shape))

        # If two points are coincident, or very close to being so, then the
        # matrix is rank deficient and thus not-invertible. Therefore,
        # only take the inverse on the full-rank set of indices.
        if self.n_points <= ThinPlateSplines.SVD_RANDOM_POINTS_THRESHOLD:
            _u, _s, _v = np.linalg.svd(self.l)
        else:
            logging.log(logging.DEBUG, "TPS too many points, using randomized SVD")
            # Randomized svd uses much less memory and is much faster than the numpy/scipy versions
            from sklearn.utils.extmath import randomized_svd
            _u, _s, _v = randomized_svd(self.l, 4, ThinPlateSplines.SVD_RANDOM_POINTS_THRESHOLD)
            
        logging.log(logging.DEBUG, "TPS build_coefficents 4 svd done")
        keep = _s.shape[0] - sum(_s < self.min_singular_val)
        logging.log(logging.DEBUG, "TPS build_coefficents 5 finding keep done. keep={}, _u.shape: {}, _s.shape: {}, _v.shape: {}".format(keep, _u[:, :keep].shape, _s[:keep, None].shape, _v[:keep, :].shape))
        _s_over = 1.0 / _s[:keep, None]
        logging.log(logging.DEBUG, "TPS build_coefficents 5.1 compute 1.0 / _s done")
        _s_over_mul_v = _s_over * _v[:keep, :]
        logging.log(logging.DEBUG, "TPS build_coefficents 5.2 compute _s_over * _v[:keep, :] done")
        #inv_l = _u[:, :keep].dot(_s_over_mul_v)
        inv_l = la.blas.dgemm(1.0, _u[:, :keep], _s_over_mul_v)
        logging.log(logging.DEBUG, "TPS build_coefficents 5.3 compute _u[:, :keep].dot(_s_over_mul_v) [== inv_l] done")
        #inv_l = _u[:, :keep].dot(1.0 / _s[:keep, None] * _v[:keep, :])
        logging.log(logging.DEBUG, "TPS build_coefficents 6 compute inv_l done, inv_l.shape: {}, self.y.T.shape: {}".format(inv_l.shape, self.y.T.shape))
        #print(inv_l)
        self.coefficients = inv_l.dot(self.y.T)
        logging.log(logging.DEBUG, "TPS build_coefficents 7 compute coefficients done, self.coefficients.shape: {}".format(self.coefficients.shape))

    def _sync_state_from_target(self):
        # now the target is updated, we only have to rebuild the
        # coefficients.
        self._build_coefficients()

    def _apply(self, points, **kwargs):
        r"""
        Performs a TPS transform on the given points.

        Parameters
        ----------
        points : ``(N, D)`` `ndarray`
            The points to transform.

        Returns
        -------
        f : ``(N, D)`` `ndarray`
            The transformed points

        Raises
        ------
        ValueError
            TPS can only be applied to 2D data.
        """
        if points.shape[1] != self.n_dims:
            raise ValueError('TPS can only be applied to 2D data.')
        x = points[..., 0][:, None]
        y = points[..., 1][:, None]
        # calculate the affine coefficients of the warp
        # (C = Constant component, then X, Y respectively)
        c_affine_c = self.coefficients[-3]
        c_affine_x = self.coefficients[-2]
        c_affine_y = self.coefficients[-1]
        # the affine warp component
        f_affine = c_affine_c + c_affine_x * x + c_affine_y * y
        # calculate a distance matrix (for L2 Norm) between every source
        # and the target
        kernel_dist = self.kernel.apply(points)
        # grab the affine free components of the warp
        c_affine_free = self.coefficients[:-3]
        # build the affine free warp component
        f_affine_free = kernel_dist.dot(c_affine_free)
        return f_affine + f_affine_free

    def apply(self, x, batch_size=None, **kwargs):
        r"""
        Applies this transform to ``x``.
        If ``x`` is :map:`Transformable`, ``x`` will be handed this transform
        object to transform itself non-destructively (a transformed copy of the
        object will be returned).
        If not, ``x`` is assumed to be an `ndarray`. The transformation will be
        non-destructive, returning the transformed version.
        Any ``kwargs`` will be passed to the specific transform :meth:`_apply`
        method.
        Parameters
        ----------
        x : :map:`Transformable` or ``(n_points, n_dims)`` `ndarray`
            The array or object to be transformed.
        batch_size : `int`, optional
            If not ``None``, this determines how many items from the numpy
            array will be passed through the transform at a time. This is
            useful for operations that require large intermediate matrices
            to be computed.
        kwargs : `dict`
            Passed through to :meth:`_apply`.
        Returns
        -------
        transformed : ``type(x)``
            The transformed object or array
        """

        def transform(x_):
            """
            Local closure which calls the :meth:`_apply` method with the
            `kwargs` attached.
            """
            return self._apply_batched(x_, batch_size, **kwargs)

        try:
            return x._transform(transform)
        except AttributeError:
            return self._apply_batched(x, batch_size, **kwargs)

    def _apply_batched(self, x, batch_size, **kwargs):
        if batch_size is None:
            return self._apply(x, **kwargs)
        else:
            outputs = []
            n_points = x.shape[0]
            for lo_ind in range(0, n_points, batch_size):
                hi_ind = lo_ind + batch_size
                outputs.append(self._apply(x[lo_ind:hi_ind], **kwargs))
            return np.vstack(outputs)


    @property
    def has_true_inverse(self):
        r"""
        :type: ``False``
        """
        return False

    def pseudoinverse(self):
        r"""
        The pseudoinverse of the transform - that is, the transform that
        results from swapping `source` and `target`, or more formally, negating
        the transforms parameters. If the transform has a true inverse this
        is returned instead.

        :type: ``type(self)``
        """
        return ThinPlateSplines(self.target, self.source, kernel=self.kernel)
