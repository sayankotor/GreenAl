from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
from src.layers2.linear import TTMLinear

# Initializing a GPT2 configuration
configuration = GPT2Config()

# Initializing a model from the configuration
model = GPT2LMHeadModel(configuration)

# Accessing the model configuration
configuration = model.config


configuration = GPT2Config()

class GPT2_TT_Model(GPT2LMHeadModel):
    def __init__(self, configuration, rank):
        super().__init__(configuration)
        for i in range(len(self.transformer.h)):
            # fc part
            old_layer = self.transformer.h[i].mlp.c_fc
            (in_, out_) = old_layer.weight.shape
            print (old_layer.weight.shape)
            layer = TTMLinear(d_in=in_, d_out=out_, rank=rank)
            #drop_layer = TTDropout(layer, proba = 0.7, min_dim = 2, rank=128)
            #layer = drop_layer
            self.transformer.h[i].mlp.c_fc = layer

            # projection
            old_layer = self.transformer.h[i].mlp.c_proj
            (in_, out_) = old_layer.weight.shape
            #print (old_layer.weight.shape)
            layer = TTMLinear(d_in=in_, d_out=out_, rank=rank)
            #drop_layer = TTDropout(layer, proba = 0.8, min_dim = 2, rank=128)
            #layer = drop_layer
            self.transformer.h[i].mlp.c_proj = layer
        