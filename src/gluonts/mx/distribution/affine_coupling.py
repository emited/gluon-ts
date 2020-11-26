from gluonts.core.component import validated
from gluonts.distribution.distribution import getF
from gluonts.mx.distribution.bijection import BijectionBlock


def clip_preserve_gradients(x, min, max):
    # This helper function clamps gradients but still passes through the gradient in clamped regions
    return x + getF(x).stop_gradient((x.clip(min, max) - x))


class AffineCoupling(BijectionBlock):
    @validated()
    def __init__(self, event_shape, hypernet, split_unit=None,
                 log_scale_min_clip=-5., log_scale_max_clip=3.,
                 mean_min_clip=-float('inf'), mean_max_clip=float('inf'),
                 scale_by_dim=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert log_scale_min_clip < log_scale_max_clip
        assert len(event_shape) == 1
        self._event_shape = event_shape
        self.split_unit = event_shape[0] // 2 if split_unit is None else split_unit
        self.hypernet = hypernet

        self._cached_log_scale = None
        self._cached_x_y = (None, None)

        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip
        self.mean_min_clip = mean_min_clip
        self.mean_max_clip = mean_max_clip
        self.scale_by_dim = scale_by_dim

    @property
    def event_shape(self):
        return self._event_shape

    @property
    def event_dim(self) -> int:
        return len(self.event_shape)

    def _clip_log_scale(self, log_scale):
        return clip_preserve_gradients(
            log_scale,
            self.log_scale_min_clip,
            self.log_scale_max_clip
        )

    def _clip_mean(self, mean):
        return clip_preserve_gradients(
            mean,
            self.mean_min_clip,
            self.mean_max_clip
        )

    def _get_affine_params(self, x1):
        params = self.hypernet(x1)
        assert params.shape[-1] % 2 == 0

        mean = params.slice_axis(axis=-1, begin=0, end=params.shape[-1] // 2)
        log_scale = params.slice_axis(axis=-1, begin=params.shape[-1] // 2, end=None)

        log_scale = self._clip_log_scale(log_scale)
        mean = self._clip_mean(mean)
        self._cached_log_scale = log_scale

        return mean, log_scale

    def _split_input(self, x):
        x1 = x.slice_axis(axis=-1, begin=0, end=self.split_unit)
        x2 = x.slice_axis(axis=-1, begin=self.split_unit, end=None)
        return x1, x2

    def f(self, x):
        F = getF(x)

        x1, x2 = self._split_input(x)
        mean, log_scale = self._get_affine_params(x1)

        y1 = x1
        y2 = F.exp(log_scale) * x2 + mean

        y = F.concat(y1, y2, dim=x.ndim - 1)
        self._cached_x_y = (x, y)

        return y

    def f_inv(self, y):
        F = getF(y)

        y1, y2 = self._split_input(y)
        mean, log_scale = self._get_affine_params(y1)

        x1 = y1
        x2 = (y2 - mean) * F.exp(-log_scale)

        x = F.concat(x1, x2, dim=y.ndim - 1)
        self._cached_x_y = (x, y)

        return x

    def log_abs_det_jac(self, x, y):
        x_old, y_old = self._cached_x_y
        if self._cached_log_scale is not None and x is x_old and y is y_old:
            log_scale = self._cached_log_scale
        else:
            x1, _ = self._split_input(x)
            _, log_scale = self._get_affine_params(x1)

        return log_scale.sum(-1)