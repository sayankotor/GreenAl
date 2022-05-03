from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
import torch
from transformers import GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling

from datasets import load_dataset
from transformers import TextDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import Trainer, TrainingArguments
from transformers import default_data_collator

import argparse
from src.classes.gpt2_tt import GPT2_TT_Model

class Object(object):
    pass

args = Object()
args.local_rank = -1
args.max_steps = -1
args.per_gpu_train_batch_size = 8
args.per_gpu_eval_batch_size = 8
args.n_gpu = 1
args.gradient_accumulation_steps = 8
args.num_train_epochs = 40
args.weight_decay = 0.0001
args.learning_rate = 6.25e-5
args.adam_epsilon = 1e-8
args.warmup_steps = 4000
args.seed = 42
args.mlm = False
args.device = torch.device('cuda:3')
args.fp16 = False
args.max_grad_norm = 1.0
args.logging_steps = 500.0
args.save_steps = 50
args.evaluate_during_training = True
args.output_dir = '/notebook/greenAI/out_tt_transformer'
args.eval_batch_size = 16
args.save_total_limit = 2

# args.is_factorized = True % unusable, factorization functionality goes to class definition

parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, required=True, default=200)

args_out = parser.parse_args()

args.rank = args_out.rank

from src.layers2.linear import TTMLinear
from help_trainer import train

def main():
    
    # Initializing a GPT2 configuration
    # Initializing a GPT2 configuration
    configuration = GPT2Config()
   
    # Initializing a model from the configuration
    if (args.rank > 0):
        model = GPT2_TT_Model(configuration, rank = args.rank)
    else:
        model = GPT2LMHeadModel(configuration)

    #if (args.is_factorized):
        #for i in range(len(model.transformer.h)):
            # fc part
            #old_layer = model.transformer.h[i].mlp.c_fc
            #(in_, out_) = old_layer.weight.shape
            #print (old_layer.weight.shape)
            #layer = TTMLinear(d_in=in_, d_out=out_, rank=64)
            #model.transformer.h[i].mlp.c_fc = layer

            # projection
            #old_layer = model.transformer.h[i].mlp.c_proj
            #(in_, out_) = old_layer.weight.shape
            #print (old_layer.weight.shape)
            #layer = TTMLinear(d_in=in_, d_out=out_, rank=64)
            #model.transformer.h[i].mlp.c_proj = layer

    # Accessing the model configuration
    configuration = model.config
    
    device = torch.device("cuda:3")
    a = model.to(device)
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    #dataset_train = load_dataset('wikitext', 'wikitext-103-v1', split='train')
    #dataset_valid = load_dataset('wikitext', 'wikitext-103-v1', split='validation')
    
    dataset_train = TextDataset(tokenizer=tokenizer, 
                                file_path="/notebook/greenAI/wikitext-103/wiki.train.tokens", 
                                block_size=512)


    dataset_valid = TextDataset(tokenizer=tokenizer, 
                                file_path="/notebook/greenAI/wikitext-103/wiki.valid.tokens", 
                                block_size=512)
    
    dataset_test = TextDataset(tokenizer=tokenizer, 
                                file_path="/notebook/greenAI/wikitext-103/wiki.test.tokens", block_size=512)
    print (len(dataset_train))
    
    train(args, dataset_train, dataset_valid, dataset_test, model, tokenizer)
    
if __name__ == "__main__":
    main()