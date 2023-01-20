import torch
from torch import nn
from linear import TTMLinear

import tensorly as tl
tl.set_backend('pytorch')

class TTDropout(nn.Module):
    def __init__(self, old_layer, proba, min_dim, rank):
        super().__init__()
        self.proba = proba
        self.min_dim = min_dim
        self.layer = old_layer
        self.rank = rank
        
    #def create_zero_mask(self):
    #def forward(self, inpt):
               
    def apply_tensor_dropout1(self, tt_tensor, training=True):
        print ("TTD dropout1", training)
        if (not self.proba) or ((not training)):
            return tt_tensor

        device = tt_tensor.ttm.tt.cores[0].device

        sampled_indices = []
        for i, rank in enumerate(tt_tensor.ttm.tt.ranks):
            if rank > self.min_dim:
                idx = tl.arange(rank, device=device, dtype=torch.int64)
                idx = idx[torch.bernoulli(torch.ones(rank, device=device)*(1 - self.proba),
                                          out=torch.empty(rank, device=device, dtype=torch.bool))]
                if len(idx) == 0:
                    idx = torch.randint(0, rank, size=(min_values, ), device=device, dtype=torch.int64)
            else:
                idx = tl.arange(rank, device=device, dtype=torch.int64).tolist()

            sampled_indices.append(idx)

        print (len(sampled_indices))
        lens = [len(elem) for elem in sampled_indices]
        print (lens)
        sampled_factors = []
        if training:
            scaling = 1/(1 - self.proba)
        else:
            scaling = 1
        for i, f in enumerate(tt_tensor.ttm.tt.cores):
            if i == 0:
                ax = len(tt_tensor.ttm.tt.cores[0].shape) - 1
                sampled_factors.append(torch.clone(torch.index_select(f, ax, sampled_indices[i])*scaling))
            elif i == (len(tt_tensor.ttm.tt.cores) - 1):
                ax = 0
                sampled_factors.append(torch.clone(torch.index_select(f, ax, sampled_indices[i - 1])*scaling))
            else:
                ax_0 = 0
                ax_end = len(tt_tensor.ttm.tt.cores[0].shape) - 1
                new_tensor = torch.index_select(f, ax_0, sampled_indices[i - 1])
                new_tensor = torch.index_select(new_tensor, ax_end, sampled_indices[i])*scaling
                sampled_factors.append(torch.clone(new_tensor))

        return nn.ParameterList(sampled_factors)
    
    def forward(self, inpt):
        if self.training:
            print ("self training")
            new_layer = TTMLinear(self.layer.d_in, self.layer.d_out, self.rank)
            for i in range(len(new_layer.ttm.tt.cores)):
                print (i)
                new_layer.ttm.tt.cores[i] = torch.clone(self.layer.ttm.tt.cores[i])
            new_layer.ttm.tt.cores = self.apply_tensor_dropout1(new_layer, training=True)
            new_layer(inpt)
        else:
            print ("else")
            return self.layer(inpt)
        