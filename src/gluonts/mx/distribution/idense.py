import numpy as np
import scipy
import scipy.linalg as slinalg  # as okl#import dtrtri
from functools import partial

import mxnet as mx
from mxnet.initializer import Initializer, Orthogonal
from mxnet.gluon import Parameter

from gluonts.core.component import validated
from gluonts.support.util import _broadcast_param
from gluonts.mx.distribution.distribution import getF
from gluonts.mx.distribution.bijection import BijectionBlock


def log_abs_det(A):
    F = getF(A)
    A_squared = F.linalg.gemm2(A, A, transpose_a=True)
    L = F.linalg.potrf(A_squared)
    return F.diag(L, axis1=-2, axis2=-1).abs().log().sum(-1)


class DeterministicInitializer(Initializer):
    def __init__(self, arr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arr = arr

    def _init_weight(self, name, arr):
        arr[:] = self.arr


class InvertibleDense(BijectionBlock):
    @validated()
    def __init__(self, event_shape, activation=None, add_identity=False, use_plu=False, use_bias=True, flatten=True,
                 dtype='float32', weight_initializer=Orthogonal(1), bias_initializer='zeros',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not (add_identity and use_plu)
        assert len(event_shape) == 1

        self._flatten = flatten
        self.add_identity = add_identity
        self.use_plu = use_plu
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self._has_been_init = False
        self.dtype = dtype
        self.use_bias = use_bias
        with self.name_scope():
            self._units = event_shape[0]
            self._event_shape = event_shape
            self.init_weight()

            if self.use_bias:
                self.bias = self.params.get('bias',
                                            shape=self.event_shape,
                                            init=self.bias_initializer,
                                            dtype=self.dtype,
                                            allow_deferred_init=True
                                            )
            else:
                self.bias = None

            if activation is not None:
                self.act = activation
            else:
                self.act = None

    def init_weight(self):
        w = Parameter(shape=(self._units, self._units), dtype=self.dtype, name='w')
        w.initialize(init=self.weight_initializer)
        np_w = w.data().asnumpy()

        dtype = 'float64'

        if not isinstance(self.weight_initializer, Orthogonal):
            # Random orthogonal matrix:
            print('doing qr decomp !')
            np_w = slinalg.qr(np_w)[0].astype('float32')
            if self.add_identity:
                np_w = np_w - np.eye(*np_w.shape)

        if self.use_plu:
            np_p, np_l, np_u = scipy.linalg.lu(np_w)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(abs(np_s))
            np_u = np.triu(np_u, k=1)
            np_l_mask = np.tril(np.ones(np_w.shape, dtype=dtype), -1)

            sign_s = mx.nd.array(np_sign_s, dtype=dtype)
            log_s = mx.nd.array(np_log_s, dtype=dtype)
            u = mx.nd.array(np_u, dtype=dtype)
            l = mx.nd.array(np_l, dtype=dtype)
            p = mx.nd.array(np_p, dtype=dtype)
            l_mask = mx.nd.array(np_l_mask, dtype=dtype)

            self._p = p
            self.sign_s = sign_s
            self.l_mask = l_mask
            self.l_unmasked = self.params.get(
                'l_unmasked',
                shape=(self._units, self._units),
                dtype=dtype,
                init=DeterministicInitializer(l)
            )
            self.u_unmasked = self.params.get(
                'u_unmasked',
                shape=(self._units, self._units),
                dtype=dtype,
                init=DeterministicInitializer(u)
            )
            self.log_s = self.params.get(
                'log_s',
                shape=(self._units,),
                dtype=dtype,
                init=DeterministicInitializer(log_s)
            )

        else:
            w = self.F.array(np_w, dtype=dtype)
            self._w = self.params.get(
                '_w',
                shape=np_w.shape,
                dtype=dtype,
                init=DeterministicInitializer(w)
            )

    @property
    def event_shape(self) -> int:
        return self._event_shape

    @property
    def event_dim(self) -> int:
        return len(self.event_shape)

    @property
    def F(self):
        if hasattr(self, '_F'):
            return self._F
        else:
            self._F = mx.nd
            return self._F

    @property
    def l(self):
        return ((self.l_unmasked.data() * self.l_mask).astype(self.dtype)
                + self.F.eye(self.l_unmasked.shape[0], dtype=self.dtype))

    @property
    def l_inv(self):
        l = (self.l_unmasked.data() * self.l_mask
             + self.F.eye(self.l_unmasked.shape[0], dtype=self.l_mask.dtype))
        l_inv, info = slinalg.lapack.clapack.dtrtri(l.asnumpy(), True)
        return self.F.array(l_inv, dtype=self.dtype)

    @property
    def u(self):
        return (self.u_unmasked.data() * self.l_mask.T
                + self.F.diag(self.sign_s * self.log_s.data().exp())).astype(self.dtype)

    @property
    def u_inv(self):
        u = (self.u_unmasked.data() * self.l_mask.T
             + self.F.diag(self.sign_s * self.log_s.data().exp()))
        u_inv, info = slinalg.lapack.clapack.dtrtri(u.asnumpy(), False)
        return self.F.array(u_inv, dtype=self.dtype)

    @property
    def p(self):
        return self._p.astype(self.dtype)

    @property
    def p_inv(self):
        # the inverse of a permutation matrix is its transpose
        return self._p.T.astype(self.dtype)

    def w(self, F):
        if self.use_plu:
            return F.gemm2(self.p, F.gemm2(self.l, self.u))
        else:
            if self.add_identity:
                return (self._w.data() + F.eye(self._w.shape[0], dtype=self._w.dtype)).astype(self.dtype)
            else:
                return self._w.data().astype(self.dtype)

    def w_inv(self, F):
        if self.use_plu:
            return F.gemm2(self.u_inv, F.gemm2(self.l_inv, self.p_inv)).astype(self.dtype)
        else:
            if self.add_identity:
                return F.linalg.inverse((self._w.data() + F.eye(self._w.shape[0], dtype=self._w.dtype))).astype(
                    self.dtype)
            else:
                return F.linalg.inverse(self._w.data()).astype(self.dtype)

    def w_dot(self, x, bias=None):
        F = getF(x)
        fc = partial(F.FullyConnected, flatten=self._flatten, no_bias=True, num_hidden=self._units)
        fc_bias = partial(F.FullyConnected, flatten=self._flatten, num_hidden=self._units)
        if self.use_plu:
            return fc_bias(fc(fc(x, self.u), self.l), self.p, bias=bias, no_bias=bias is None)
        else:
            return fc_bias(x, self.w(F), bias=bias, no_bias=bias is None)

    def w_inv_dot(self, y, bias=None):
        F = getF(y)
        fc = partial(F.FullyConnected, flatten=self._flatten, no_bias=True, num_hidden=self._units)
        if bias is not None:
            y = F.broadcast_minus(y, self.bias.data())
        if self.use_plu:
            return fc(fc(fc(y, self.p_inv), self.l_inv), self.u_inv)
        else:
            return fc(y, self.w_inv(F))

    def f_inv(self, y):
        x = y.astype(self.dtype)
        x = self.w_dot(x, bias=None if self.bias is None else self.bias.data())
        if self.act is not None:
            x = self.act.f(x)
        return x

    def f(self, x):
        y = x.astype(self.dtype)
        if self.act is not None:
            y = self.act.f_inv(y)
        y = self.w_inv_dot(y, bias=self.bias if self.bias is None else self.bias.data())
        return y

    def log_abs_det_jac(self, x=None, y=None):
        if x is not None:
            batch_shape = x.shape[:-1]
        elif y is not None:
            batch_shape = y.shape[:-1]
        else:
            batch_shape = ()

        if self.use_plu:
            ladj = -self.log_s.data().sum().astype(self.dtype)
        else:
            jac = self.w(getF(x))
            # entry 1 returns the log abs det
            # ladj = getF(x).linalg.slogdet()[1]
            ladj = -log_abs_det(jac)

        ladj = _broadcast_param(ladj,
                                axes=range(len(batch_shape)),
                                sizes=batch_shape
                                ).reshape(batch_shape)

        if self.act is not None:
            # in f we call self.act.f_inv
            ladj = ladj - self.act.log_abs_det_jac(x, y)

        return ladj