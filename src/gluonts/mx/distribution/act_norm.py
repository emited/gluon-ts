import mxnet as mx
from gluonts.support.util import _broadcast_param
from gluonts.mx.distribution.distribution import getF
from gluonts.mx.distribution.affine_coupling import clip_preserve_gradients
from gluonts.mx.distribution.bijection import Bijection, BijectionBlock


class ConditionedActNorm1d(Bijection):
    @property
    def event_shape(self):
        return self._event_shape

    @property
    def event_dim(self) -> int:
        return len(self.event_shape)

    def f(self, x):
        F = getF(x)
        return x * F.exp(self.log_scale) + self.mean

    def f_inv(self, x):
        F = getF(x)
        return (x - self.mean) * F.exp(-self.log_scale)

    def log_abs_det_jac(self, x, y):
        if x is not None:
            batch_shape = x.shape[:-1]
        elif y is not None:
            batch_shape = y.shape[:-1]
        else:
            assert False
        jac = self.log_scale.sum()
        jac = _broadcast_param(jac,
                               axes=range(len(batch_shape)),
                               sizes=batch_shape
                               ).reshape(batch_shape)
        return jac


class ActNorm1d(BijectionBlock, ConditionedActNorm1d):
    def __init__(self, event_shape, dtype='float32',
                 log_scale_min_clip=-float('inf'), log_scale_max_clip=float('inf'),
                 mean_min_clip=-float('inf'), mean_max_clip=float('inf'),
                 log_scale_initializer='zeros', mean_initializer='zeros',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._event_shape = event_shape
        self.dtype = dtype
        self.log_scale_initializer = log_scale_initializer
        self.mean_initializer = mean_initializer
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip
        self.mean_min_clip = mean_min_clip
        self.mean_max_clip = mean_max_clip
        self._mean = self.params.get('mean',
                                     shape=self.event_shape,
                                     init=self.mean_initializer,
                                     dtype=self.dtype,
                                     )
        self._log_scale = self.params.get('log_scale',
                                          shape=self.event_shape,
                                          init=self.log_scale_initializer,
                                          dtype=self.dtype
                                          )

    @property
    def mean(self):
        return clip_preserve_gradients(self._mean.data(),
                                       self.mean_min_clip,
                                       self.mean_max_clip
                                       )

    @property
    def log_scale(self):
        return clip_preserve_gradients(self._log_scale.data(),
                                       self.log_scale_min_clip,
                                       self.log_scale_max_clip
                                       )


class ConditionedShift1d(Bijection):
    def __init__(self, mean):
        super().__init__()
        self.mean = mean
        self._event_shape = (mean.shape[-1],)

    @property
    def event_shape(self):
        return self._event_shape

    @property
    def event_dim(self):
        return len(self.event_shape)

    def f(self, x):
        return x + self.mean

    def f_inv(self, x):
        return x - self.mean

    def log_abs_det_jac(self, x, y):
        return mx.nd.zeros(x.shape[:-1])