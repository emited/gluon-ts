from gluonts.mx.block.mlp import BNMLP
from gluonts.mx.distribution.idense import InvertibleDense
from gluonts.mx.distribution.perm import ShufflePermutation
from gluonts.mx.distribution.affine_coupling import AffineCoupling
from gluonts.mx.distribution.act_norm import ActNorm1d
from gluonts.mx.distribution.bijection import (
    BijectionBlock,
    ComposedBijectionBlock,
)


default_params = {
    'num_blocks': 9,
    'block': {
        'num_steps': 1,
        'permutation': {'_name': 'shuffle'},
        'affine_coupling': {
            'scale_by_dim': True,
            'hypernet': {
                'flatten': False,
                'batch_norm': False,
                'num_layers': 2,
                'activation': 'relu',
                'hidden_units': 16,
                'weight_initializer': {
                    'name': 'xavier',
                    'magnitude': 0.02,
                }
            }
        }
    }
}


def init_permutation(_name, **kwargs):
    if _name == 'shuffle':
        return ShufflePermutation(**kwargs)
    if _name == 'idense':
        return InvertibleDense(**kwargs)
    raise NotImplementedError(_name)


def glow_block(num_steps, event_shape, affine_coupling, permutation, act_norm=False):
    assert len(event_shape) == 1

    layers = []
    for _ in range(num_steps):

        if act_norm:
            layers += [ActNorm1d(**act_norm, event_shape=event_shape)]

        hypernet = BNMLP(
            in_units=event_shape[0] // 2,
            out_units=(event_shape[0] - event_shape[0] // 2) * 2,
            **affine_coupling['hypernet']
        )

        layers += [AffineCoupling(
            hypernet=hypernet,
            split_unit=event_shape[0] // 2,
            event_shape=event_shape,
            **{k: v for k, v in affine_coupling.items()
               if k != 'hypernet'}
        )]

        layers += [init_permutation(event_shape=event_shape, **permutation)]

    return ComposedBijectionBlock(layers)


def glow(event_shape, block=default_params['block'], num_blocks=9):
    blocks = ComposedBijectionBlock([])
    for _ in range(num_blocks):
        blocks += glow_block(event_shape=event_shape, **block)
    return blocks