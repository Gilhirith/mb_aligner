import numpy as np
from scipy.spatial.distance import cdist
import math
#from .base import Transform


#class RadialBasisFunction(Transform):
class RadialBasisFunction(object):
    r"""
    Radial Basis Functions are a class of transform that is used by
    :map:`ThinPlateSplines`. They have to be able to take their own radial
    derivative for :map:`ThinPlateSplines` to be able to take its own total
    derivative.

    Parameters
    ----------
    c : ``(n_centres, n_dims)`` `ndarray`
        The set of centers that make the basis. Usually represents a set of
        source landmarks.
    """
    def __init__(self, c):
        self.c = c

    @property
    def n_centres(self):
        r"""
        The number of centres.

        :type: `int`
        """
        return self.c.shape[0]

    @property
    def n_dims(self):
        r"""
        The RBF can only be applied on points with the same dimensionality as
        the centres.

        :type: `int`
        """
        return self.c.shape[1]

    @property
    def n_dims_output(self):
        r"""
        The result of the transform has a dimension (weight) for every centre.

        :type: `int`
        """
        return self.n_centres


class R2LogR2RBF(RadialBasisFunction):
    r"""
    The :math:`r^2 \log{r^2}` basis function.

    The derivative of this function is :math:`2 r (\log{r^2} + 1)`.

    .. note::

        :math:`r = \lVert x - c \rVert`

    Parameters
    ----------
    c : ``(n_centres, n_dims)`` `ndarray`
        The set of centers that make the basis. Usually represents a set of
        source landmarks.
    """
    def __init__(self, c):
        super(R2LogR2RBF, self).__init__(c)

    def _apply(self, x, **kwargs):
        """
        Apply the basis function.

        .. note::

            :math:`r^2 \log{r^2} === r^2 2 \log{r}`

        Parameters
        ----------
        x : ``(n_points, n_dims)`` `ndarray`
            Set of points to apply the basis to.

        Returns
        -------
        u : ``(n_points, n_centres)`` `ndarray`
            The basis function applied to each distance,
            :math:`\lVert x - c \rVert`.
        """
        euclidean_distance = cdist(x, self.c)
        mask = euclidean_distance == 0
        with np.errstate(divide='ignore', invalid='ignore'):
#             u = (euclidean_distance ** 2 *
#                  (2 * np.log(euclidean_distance)))
            u = (euclidean_distance ** 2 *
                 (2.0 / np.log2(math.e) * np.log2(euclidean_distance)))
        # reset singularities to 0
        u[mask] = 0
        return u

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


class R2LogRRBF(RadialBasisFunction):
    r"""
    Calculates the :math:`r^2 \log{r}` basis function.

    The derivative of this function is :math:`r (1 + 2 \log{r})`.

    .. note::

        :math:`r = \lVert x - c \rVert`

    Parameters
    ----------
    c : ``(n_centres, n_dims)`` `ndarray`
        The set of centers that make the basis. Usually represents a set of
        source landmarks.
    """
    def __init__(self, c):
        super(R2LogRRBF, self).__init__(c)

    def _apply(self, points, **kwargs):
        """
        Apply the basis function :math:`r^2 \log{r}`.

        Parameters
        ----------
        points : ``(n_points, n_dims)`` `ndarray`
            Set of points to apply the basis to.

        Returns
        -------
        u : ``(n_points, n_centres)`` `ndarray`
            The basis function applied to each distance,
            :math:`\lVert points - c \rVert`.
        """
        euclidean_distance = cdist(points, self.c)
        mask = euclidean_distance == 0
        with np.errstate(divide='ignore', invalid='ignore'):
            u = euclidean_distance ** 2 * np.log(euclidean_distance)
        # reset singularities to 0
        u[mask] = 0
        return u

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

