import mxnet as mx

from gluonts.distribution.distribution import getF
from gluonts.core.component import validated
from gluonts.mx.distribution.bijection import BijectionBlock


class ShufflePermutation(BijectionBlock):
    @validated()
    def __init__(self, event_shape,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(event_shape) == 1
        self._event_shape = event_shape
        # print('TO DO set RNG for Perm')
        indices = mx.nd.arange(event_shape[0])
        self.indices = mx.nd.random.shuffle(indices)
        self.inv_indices = mx.nd.zeros_like(indices)
        for i in range(len(self.indices)):
            self.inv_indices[self.indices[i]] = i

    @property
    def event_shape(self) -> int:
        return self._event_shape

    @property
    def event_dim(self) -> int:
        return len(self.event_shape)

    def _permutate(self, x, inv):
        if inv:
            indices = self.indices
        else:
            indices = self.inv_indices
        indices = indices.expand_dims(0)
        reversed_axis = list(reversed(range(x.ndim)))
        return getF(x).gather_nd(x.transpose(reversed_axis), indices)\
            .transpose(reversed_axis)

    def f_inv(self, y):
        return self._permutate(y, inv=True)

    def f(self, x):
        return self._permutate(x, inv=False)

    def log_abs_det_jac(self, x=None, y=None):
        return 0