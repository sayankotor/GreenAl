from functools import lru_cache
from typing import Union

import torch as t
from torch import nn
import numpy as np
import opt_einsum
from opt_einsum import contract_expression

from typing import List, Union, Tuple


def factorize(n: int):
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


def mix(first: tuple, second: tuple):
    return sum(map(tuple, zip(first, second)), tuple()) + first[len(second):] + second[len(first):]


def unmix(seq, len1: int, len2: int):
    min_len = min(len1, len2)
    return (
        seq[:2 * min_len:2] + seq[2 * min_len:2 * min_len + len1 - len2],
        seq[1:2 * min_len:2] + seq[2 * min_len:2 * min_len + len2 - len1],
    )


class TTContainer(nn.Module):
    def __init__(self, dims: List[int], rank_or_ranks: Union[int, List[int]]):
        super().__init__()

        ranks = rank_or_ranks if isinstance(rank_or_ranks, list) else [rank_or_ranks] * (len(dims) - 1)

        assert len(dims) == len(ranks) + 1

        self.cores = nn.ParameterList(
            [
                nn.Parameter(t.randn(r1, dim, r2) / np.sqrt(dim * r1), requires_grad=True)
                for dim, r1, r2 in zip(dims, [1] + ranks, ranks + [1])
            ]
        )

    @property
    def n_dims(self):
        return len(self.cores)


class TTM(nn.Module):
    def __init__(self, dim_in, dim_out, rank: int):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dims_in = tuple(factorize(dim_in))
        self.dims_out = tuple(factorize(dim_out))
        self.dims = mix(self.dims_in, self.dims_out)

        self.tt = TTContainer(self.dims, rank)

    def get_in_out_dim_inds(self) -> Tuple[List[int], List[int]]:
        return unmix(list(range(len(self.dims_in) + len(self.dims_out))), len(self.dims_in), len(self.dims_out))

    def forward(self, x):
        ein_str_parts = []

        for i, cores in enumerate(self.tt.cores):
            r1 = opt_einsum.get_symbol(self.tt.n_dims + i)
            dim = opt_einsum.get_symbol(i)
            r2 = opt_einsum.get_symbol(self.tt.n_dims + i + 1)

            ein_str_parts.append(f'{r1}{dim}{r2}')

        in_inds, out_inds = self.get_in_out_dim_inds()

        batch_dim = self.tt.n_dims * 2 + 2
        x_dims = [batch_dim] + in_inds
        ein_str_parts.append(''.join(opt_einsum.get_symbol(dim) for dim in x_dims))

        ein_str_parts.append(''.join(opt_einsum.get_symbol(dim) for dim in out_inds))
        ein_str = f'{",".join(ein_str_parts[:-1])}->{opt_einsum.get_symbol(batch_dim)}{ein_str_parts[-1]}'
        x_reshaped = x.reshape(x.shape[:1] + self.dims_in) # [4096, 2, 2, 2, 2, 2, 2, 2, 2, 3]
        a = cached_einsum(ein_str, *self.tt.cores, x_reshaped) # [4096, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3]
        a = a.reshape(x_reshaped.shape[0], -1) #([4096, 3072]) might be ([4096, 768])
        
        return cached_einsum(ein_str, *self.tt.cores, x_reshaped).reshape(x_reshaped.shape[0], -1)

    def full_tensor(self) -> t.Tensor:
        ds = [opt_einsum.get_symbol(i) for i in range(len(self.dims))]
        rs = [opt_einsum.get_symbol(len(self.dims) + i) for i in range(len(self.dims) + 1)]

        left = ','.join(f'{r1}{d}{r2}' for r1, d, r2 in zip(rs, ds, rs[1:]))
        right = ''.join(sum(unmix(ds, len(self.dims_in), len(self.dims_out)), []))

        return cached_einsum(f'{left}->{right}', *self.tt.cores).reshape(self.dim_in, self.dim_out)


class TTMByHand(TTM):
    def forward(self, x):
        x = x.reshape(x.shape[:1] + (2,) * self.n_dims_in + (1,))
        for i in range(self.n_dims_in):
            core = cached_einsum('ris,sjt->rijt', self.tt.cores[i * 2], self.tt.cores[i * 2 + 1])
            x = t.tensordot(x, core, ([-1, 1], [0, 1]))
        for i in range(self.n_dims_in * 2, self.tt.n_dims):
            x = cached_einsum('b...r,ris->b...is', x, self.tt.cores[i])

        return x.reshape(x.shape[0], -1)


def cached_einsum(expr: str, *args):
    return cached_einsum_expr(expr, *[arg.shape for arg in args])(*args)


@lru_cache()
def cached_einsum_expr(expr: str, *shapes, maxsize=128):
    return contract_expression(expr, *shapes)
