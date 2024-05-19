import chex
import equinox as eqx
import jax
import jax.random as jr, jax.numpy as jnp, jax.nn as jnn
import numpy as np
from typing import Callable, List


def trunc_init(weight: jax.Array, scale, key: chex.PRNGKey) -> jax.Array:
    out, in_ = weight.shape
    stddev = scale
    return stddev * jr.truncated_normal(key, shape=(out, in_), lower=-1, upper=1)


def init_linear_weight(model, init_fn, key):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [
        x.weight
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
        if is_linear(x)
    ]
    weights = get_weights(model)
    new_weights = [
        init_fn(weight, model.init_scale, subkey)
        for weight, subkey in zip(weights, jr.split(key, len(weights)))
    ]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model


class SplineLinear(eqx.nn.Linear):
    init_scale: float

    def __init__(
        self, in_features: int, out_features: int, init_scale: float = 0.1, *, key
    ) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, use_bias=False, key=key)


class RadialBasisFunction(eqx.Module):

    grid: chex.ArrayDevice
    denominator: float

    def __init__(
        self,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 5,
        denominator: float = 1,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = jnp.linspace(grid_min, grid_max, num_grids)
        self.grid = grid
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def __call__(self, x):
        return jnp.exp(-(((x[..., None] - self.grid) / self.denominator) ** 2))


class KANLayer(eqx.Module):
    layernorm: eqx.nn.LayerNorm
    rbf: RadialBasisFunction
    spline_linear: eqx.Module
    base_activation: Callable
    base_linear: eqx.nn.Linear
    use_base_update: bool

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 5,
        use_base_update: bool = True,
        base_activation=jnn.silu,
        spline_weight_init_scale: float = 0.1,
        *,
        key
    ) -> None:
        super().__init__()
        self.layernorm = eqx.nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        spline_linear = SplineLinear(
            input_dim * num_grids, output_dim, spline_weight_init_scale, key=key
        )
        self.spline_linear = init_linear_weight(spline_linear, trunc_init, key)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = eqx.nn.Linear(input_dim, output_dim, key=key)

    def __call__(self, x, time_benchmark=False):
        if not time_benchmark:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)

        ret = self.spline_linear(spline_basis.reshape(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret


class KAN(eqx.Module):
    layers: eqx.nn.Sequential

    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 5,
        use_base_update: bool = True,
        base_activation=jnn.silu,
        spline_weight_init_scale: float = 0.1,
        *,
        key
    ) -> None:
        super().__init__()
        self.layers = eqx.nn.Sequential(
            [
                KANLayer(
                    in_dim,
                    out_dim,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=num_grids,
                    use_base_update=use_base_update,
                    base_activation=base_activation,
                    spline_weight_init_scale=spline_weight_init_scale,
                    key=key,
                )
                for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
            ]
        )

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    key = jr.PRNGKey(0)
    in_features = 28 * 28
    out_features = 10
    f1 = eqx.filter_jit(KAN(layers_hidden=[in_features, 64, 32, out_features], key=key))
    x = np.arange(in_features, dtype=np.float32)
