# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Standard library imports
from typing import List

# Third-party imports
import mxnet as mx
from mxnet.gluon import nn

# First-party imports
from gluonts.core.component import validated
from gluonts.model.common import Tensor
from gluonts.mx.block.batch_norm import BatchNorm
from gluonts.mx.activation import get_activation


class MLP(nn.HybridBlock):
    """
    Defines an MLP block.

    Parameters
    ----------
    layer_sizes
        number of hidden units per layer.

    flatten
        toggle whether to flatten the output tensor.

    activation
        activation function of the MLP, default is relu.
    """

    @validated()
    def __init__(
        self, layer_sizes: List[int], flatten: bool, activation="relu", **kwargs
    ) -> None:
        super().__init__()
        self.layer_sizes = layer_sizes
        with self.name_scope():
            self.layers = nn.HybridSequential()
            for layer, layer_dim in enumerate(layer_sizes):
                self.layers.add(
                    nn.Dense(layer_dim, flatten=flatten, activation=activation, **kwargs)
                )

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        x
            Input tensor

        Returns
        -------
        Tensor
            Output of the MLP given the input tensor.
        """
        return self.layers(x)

class BNMLP(nn.HybridBlock):
    @validated()
    def __init__(self, in_units, hidden_units, out_units,
                 num_layers=1, activation='relu', batch_norm=True, flatten=True,
                 weight_initializer=mx.init.Xavier(magnitude=0.02),
                 bias_initializer='zeros',
                 ):
        super().__init__()
        self.in_units = in_units
        self.hidden_units = hidden_units
        self.out_units = out_units
        self.num_layers = num_layers
        self.activation = activation
        self.batch_norm = batch_norm
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.flatten = flatten

        in_units = self.in_units
        with self.name_scope():
            self.mlp = nn.HybridSequential()
            for i in range(self.num_layers):
                lin = nn.Dense(self.hidden_units,
                               in_units=in_units,
                               activation=None,
                               weight_initializer=self.weight_initializer,
                               bias_initializer=self.bias_initializer,
                               flatten=self.flatten,
                               )
                act = get_activation(self.activation)()
                self.mlp.add(lin, act)
                if self.batch_norm:
                    print('TO DO: Make Batch Norm work in inference mode !!')
                    bn = BatchNorm()
                    self.mlp.add(bn)
                in_units = self.hidden_units
            last_lin = nn.Dense(self.out_units,
                                in_units=in_units,
                                activation=None,
                                weight_initializer=self.weight_initializer,
                                bias_initializer=self.bias_initializer,
                                flatten=self.flatten,
                                )
            self.mlp.add(last_lin)

    def _set_batch_norm_axis(self, x):
        for layer in self.mlp:
            if isinstance(layer, BatchNorm):
                layer.set_axis(x.ndim - 1)

    def hybrid_forward(self, F, x):
        self._set_batch_norm_axis(x)
        return self.mlp(x)
