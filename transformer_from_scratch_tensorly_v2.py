from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
import torch
from transformers import GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling

from datasets import load_dataset
from transformers import TextDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import Trainer, TrainingArguments
from transformers import default_data_collator

from tltorch.factorized_layers import FactorizedLinear

class Object(object):
    pass

args = Object()
args.local_rank = -1
args.max_steps = -1
args.per_gpu_train_batch_size = 32
args.per_gpu_eval_batch_size = 32
args.n_gpu = 1
args.gradient_accumulation_steps = 2
args.num_train_epochs = 40
args.weight_decay = 0.0
args.learning_rate = 2.5e-5
args.adam_epsilon = 1e-8
args.warmup_steps = 4000
args.seed = 42
args.mlm = False
args.device = torch.device('cuda:1')
args.fp16 = False
args.max_grad_norm = 1.0
args.logging_steps = 500.0
args.save_steps = 50
args.evaluate_during_training = True
args.output_dir = '/notebook/greenAI/out_simple_transformer_tensorly'
args.eval_batch_size = 32
args.save_total_limit = 2
args.is_factorized = True

from src.layers2.linear import TTMLinear
from help_trainer import train

from typing import Union, List, Tuple
from tltorch.factorized_layers.factorized_linear import FactorizedLinear

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

def log2(n: int, assert_eq: bool):
    res = 0
    while 2 ** res < n:
        res += 1

    if assert_eq:
        assert 2 ** res == n

    return res

def best_approx(n: int, max_factor: int = 3):
    n_factors = log2(n, False)
    while True:
        factors = factorize(n)
        print ("factors", tuple(factors))
        return tuple(factors)
        if len(factors) <= n_factors and all([f <= max_factor for f in factors]):
            return n
        n += 1

def main():
    
    # Initializing a GPT2 configuration
    configuration = GPT2Config()

    # Initializing a model from the configuration
    model = GPT2LMHeadModel(configuration)
    if (args.is_factorized):
        for i in range(len(model.transformer.h)):
            # fc part
            old_layer = model.transformer.h[i].mlp.c_fc
            (in_, out_) = old_layer.weight.shape
            itf = (16,16,3)
            otf = (16,16,12)
            print (old_layer.weight.shape)
            layer = FactorizedLinear(in_tensorized_features=itf, out_tensorized_features=otf, rank=128, factorization = 'blocktt')
            print (layer.weight.shape)
            model.transformer.h[i].mlp.c_fc = layer

            # projection
            old_layer = model.transformer.h[i].mlp.c_proj
            (in_, out_) = old_layer.weight.shape
            otf = (16,16,3)
            itf = (16,16,12)
            print (old_layer.weight.shape)
            layer = FactorizedLinear(in_tensorized_features=itf, out_tensorized_features=otf, rank=128, factorization = 'blocktt')
            print (layer.weight.shape)
            model.transformer.h[i].mlp.c_proj = layer

    # Accessing the model configuration
    configuration = model.config
    
    device = torch.device("cuda:1")
    a = model.to(device)
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    dataset_train = load_dataset('wikitext', 'wikitext-103-v1', split='train')
    dataset_valid = load_dataset('wikitext', 'wikitext-103-v1', split='validation')
    
    dataset_train = TextDataset(tokenizer=tokenizer, 
                                file_path="/notebook/greenAI/wikitext-2/wiki.train.tokens", 
                                block_size=128)

    dataset_valid = TextDataset(tokenizer=tokenizer, 
                                file_path="/notebook/greenAI/wikitext-2/wiki.valid.tokens", 
                                block_size=128)

    sampler = RandomSampler(dataset_train)
    tdata_loader = DataLoader(dataset_train, 
                             batch_size=42, 
                             sampler=sampler)
    vdata_loader = DataLoader(dataset_valid, 
                             batch_size=42, 
                             sampler=sampler)
    
    train(args, dataset_train, dataset_valid, model, tokenizer)
    
if __name__ == "__main__":
    main()