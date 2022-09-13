import math
from functools import lru_cache
from typing import Union, List, Tuple

import torch as t
from torch import nn
from torch.nn.functional import pad
import numpy as np
import opt_einsum
from opt_einsum import contract_expression


class TTMContainer(nn.Module):
    """
    Creates object which stores sequence of TTM cores in a form of PyTorch ParameterList

    Attributes:
    ----------

    ranks: List[Tuple[int, int]]
        Dimensions of TTM cores
    cores: nn.ParameterList
        List of Torch.nn objects, corresponding to TT cores


    See Also:
    ----------
    ttm2.TTMLinear
    """

    def __init__(self, dims: List[Tuple[int, int]], rank_or_ranks: Union[int, List[int]]):
        super().__init__()

        ranks = rank_or_ranks if isinstance(rank_or_ranks, list) else [rank_or_ranks] * (len(dims) - 1)

        assert len(dims) == len(ranks) + 1
        self.ranks = ranks
        self.cores = nn.ParameterList(
            [
                nn.Parameter(t.randn(r1, dim1, dim2, r2) / np.sqrt(dim1 * dim2 * r1), requires_grad=True)
                for (dim1, dim2), r1, r2 in zip(dims, [1] + ranks, ranks + [1])
            ]
        )

    @property
    def n_dims(self):
        return len(self.cores)


def shrink(dims: List[Tuple[int, int]], rank: int):
    cur_dims = (1, 1)
    result_dims = []
    for d1, d2 in dims:
        target_size = rank
        cur_dims = cur_dims[0] * d1, cur_dims[1] * d2
        if cur_dims[0] * cur_dims[1] >= target_size:
            result_dims.append(cur_dims)
            cur_dims = (1, 1)

    if cur_dims[0] * cur_dims[1] != 1:
        result_dims.append(cur_dims)

    return result_dims


class TTM(nn.Module):
    """
    A wrapper over TTM container, that provides cretion TTM over given 2-dimention matrix.

    Attributes:
    ----------

    dim_in: int
        The first dimension of the initial matrix

    dim_out: int
        The second dimension of the initial matrix

    dims_in: tuple(int)
        First dimensions of the expected sequence of TTM cores

    dims_out: tuple(int)
        Seconf dimensions of the expected sequence of TTM cores

    ttm:
        TTMContainer object


    See Also:
    ----------
    ttm2.TTMLinear
    .TTMContainer
    """

    def __init__(self, dim_in, dim_out, rank: int):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dims_in = tuple(factorize(dim_in))
        self.dims_out = tuple(factorize(dim_out))
        self.dims_in += (1,) * (len(self.dims_out) - len(self.dims_in))
        self.dims_out += (1,) * (len(self.dims_in) - len(self.dims_out))
        self.dims = list(zip(self.dims_in, self.dims_out))
        self.dims = shrink(self.dims, rank)
        print(self.dims)
        self.dims_in = tuple(d for d, _ in self.dims)
        self.dims_out = tuple(d for _, d in self.dims)

        self.tt = TTMContainer(self.dims, rank)

        with t.no_grad():
            state = t.eye(1)
            for core_param in self.tt.cores[:-1]:
                core = cached_einsum('ij,jklm->iklm', state, core_param)
                shape = core.shape
                core = core.reshape(-1, shape[-1])
                q, r = t.linalg.qr(core)
                core_param.data.copy_(q.reshape(shape))
                state = r
            self.tt.cores[-1].data.copy_(cached_einsum('ij,jklm->iklm', state, self.tt.cores[-1]))

    def forward(self, x):
        ein_str_parts = []

        for i, cores in enumerate(self.tt.cores):
            r1 = opt_einsum.get_symbol(self.tt.n_dims * 2 + i)
            dim_in = opt_einsum.get_symbol(i * 2)
            dim_out = opt_einsum.get_symbol(i * 2 + 1)
            r2 = opt_einsum.get_symbol(self.tt.n_dims * 2 + i + 1)

            ein_str_parts.append(f'{r1}{dim_in}{dim_out}{r2}')

        batch_dim = self.tt.n_dims * 3 + 1

        in_dims = [batch_dim] + (2 * np.arange(self.tt.n_dims)).tolist()
        ein_str_parts.append(''.join(opt_einsum.get_symbol(dim) for dim in in_dims))

        out_dims = [batch_dim] + (2 * np.arange(self.tt.n_dims) + 1).tolist()
        ein_str_parts.append(''.join(opt_einsum.get_symbol(dim) for dim in out_dims))

        ein_str = f'{",".join(ein_str_parts[:-1])}->{ein_str_parts[-1]}'
        # print (ein_str)
        x_reshaped = x.reshape(x.shape[:1] + self.dims_in)
        return cached_einsum(ein_str, *self.tt.cores, x_reshaped).reshape(x_reshaped.shape[0], -1)

    def full_tensor(self) -> t.Tensor:
        """
        Matches the elements of the tensor train with the elements of the original matrix (which was decomposed into a TT)

        """
        ds = [(opt_einsum.get_symbol(i * 2 + 0), opt_einsum.get_symbol(i * 2 + 1)) for i in range(self.tt.n_dims)]
        rs = [opt_einsum.get_symbol(self.tt.n_dims * 2 + i) for i in range(self.tt.n_dims + 1)]

        left = ','.join(f'{r1}{d1}{d2}{r2}' for r1, (d1, d2), r2 in zip(rs[:-1], ds, rs[1:]))
        right = ''.join(d for ds in zip(*ds) for d in ds)

        return cached_einsum(f'{left}->{right}', *self.tt.cores).reshape(self.dim_in, self.dim_out)


class TTMByHands(TTM):
    def forward(self, x):
        x = x.reshape(x.shape[:1] + self.dims_in + (1,))
        batch_symbol = opt_einsum.get_symbol(self.tt.n_dims)
        prev_rank_symbol = opt_einsum.get_symbol(self.tt.n_dims + 1)
        next_rank_symbol = opt_einsum.get_symbol(self.tt.n_dims + 2)
        new_dim_symbol = opt_einsum.get_symbol(self.tt.n_dims + 3)

        x_dims = [batch_symbol] + [opt_einsum.get_symbol(i) for i in range(self.tt.n_dims)] + [prev_rank_symbol]

        x_expr = ''.join(x_dims)
        for i in range(self.tt.n_dims):
            core_expr = f'{prev_rank_symbol}{opt_einsum.get_symbol(i)}{new_dim_symbol}{next_rank_symbol}'
            result_expr = ''.join(x_dims[:i + 1] + [new_dim_symbol] + x_dims[i + 2:-1] + [next_rank_symbol])
            # print ("{x_expr},{core_expr},{result_expr}", x_expr,core_expr,result_expr)
            x_new = cached_einsum(f'{x_expr},{core_expr}->{result_expr}', x, self.tt.cores[i])

            pad_dims = (
                    [0, max(0, x_new.shape[-1] - x.shape[-1])] +
                    [0] * (2 * (x.ndim - i - 3)) +
                    [0, x_new.shape[i + 1] - x.shape[i + 1]] +
                    [0] * (2 * (i + 1))
            )
            x_padded = nn.functional.pad(x, pad_dims)
            if x_padded.shape[-1] > x_new.shape[-1]:
                x_padded = x_padded[..., x_new.shape[-1]]
            assert x_new.shape == x_padded.shape
            x = x_new + x_padded
        return x.reshape(x.shape[0], -1)


def factorize(n: int) -> List[int]:
    if n == 1:
        return [1]
    factors = []
    i = 2
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n //= i
        i += 1
    if n != 1:
        factors.append(n)
    return factors


def cached_einsum(expr: str, *args):
    return cached_einsum_expr(expr, *[arg.shape for arg in args])(*args)


@lru_cache(maxsize=None)
def cached_einsum_expr(expr: str, *shapes):
    return contract_expression(expr, *shapes)



class TTMLinear(nn.Module):
    """
    Creates TTM layer corresponding to Linear layer with weihgts matrix MXN

    Attributes:
    ----------

    d_in: int
        The first dimension (M) of the weights in the initial Linear layer

    d_out: int
        The second dimension (N) of the weights in the initial Linear layer

    d_in_ttm: int
        Nearest number to d_in that can be factored by a value not greater than treshhold

    d_out_ttm: int
        Nearest number to d_out that can be factored by a value not greater than treshhold

    ttm:
        object of TTM class

    See Also:
    ----------
    ttm2.TTM
    """

    def __init__(self, d_in: int, d_out: int, rank: int):
        super().__init__()
        """
        Pads initial layer shape up to best approximated shape and create a TTM object.

        """
        self.d_in = d_in
        self.d_in_ttm = best_approx(d_in)
        self.d_out = d_out
        self.d_out_ttm = best_approx(d_out)
        self.ttm = TTM(self.d_in_ttm, self.d_out_ttm, rank)

    def forward(self, x: t.tensor):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        x = pad(x, (0, 0, 0, self.d_in_ttm - self.d_in))
        x = self.ttm(x)
        x = x.reshape(shape[0], -1, self.d_out)
        return x[:, :self.d_out]

    def row_norms(self):
        return self.ttm.full_tensor()[:self.d_in, :self.d_out].norm(p=2, dim=0)


def best_approx(n: int, max_factor: int = 3):
    """
    Finds the best finds the best factorization with values not greater than max_factor

    Parameters:
    ----------
    n : int
        number to factorize
    max_factor: int
        upper bound on possible multipliers

    Returns:
    The most suitable number greater or equal to the n

    """
    n_factors = math.floor(math.log2(n))
    while True:
        factors = factorize(n)
        if len(factors) <= n_factors and all([f <= max_factor for f in factors]):
            return n
        n += 1