from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
import torch
from transformers import GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling

from datasets import load_from_disk

import argparse
import os

import torch.multiprocessing as mp

from transformers import Trainer, TrainingArguments
from transformers import default_data_collator
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader

from datasets import load_dataset

import argparse
from src.classes.gpt2_tt import GPT2_TT_Model


import sys
sys.path.append("/notebook/greenAI/transformers/src")
from transformers.tokenization_utils import PreTrainedTokenizer
import torch
import torch.distributed as dist
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

#mapping datasets
from datasets import load_dataset
from transformers import TextDataset

# iterable datasets
from src.data_classes.iterable_dataset_mp import getListOfFiles, FileListDataset

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1, 2, 3"

from src.layers2.linear import TTMLinear
from help_trainer_last import train

#import sys
#sys.path.append("/home/jovyan/") 
#sys.path.append("/home/jovyan/transformers/src") 


from datasets.utils.logging import set_verbosity_error
set_verbosity_error()


def train_mp_wrapper(gpu, args):
    print ("wr", gpu, flush = True)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=gpu, world_size=args.n_gpu)
    print ("gpu", gpu, flush = True)
    worker_info = torch.utils.data.get_worker_info()
    print ("worker_info in wrapper", worker_info)
    # Initializing a GPT2 configuration
    # Initializing a GPT2 configuration
    configuration = GPT2MedConfig()
   
    # Initializing a model from the configuration
    if (args.rank > 0):
        model = GPT2_TT_Model(configuration, rank = args.rank)
    else:
        model = GPT2LMHeadModel(configuration)
    model.to(gpu)
    ddp_model = DDP(model, device_ids=[gpu], output_device = gpu)
    # Accessing the model configuration
    configuration = model.config
    torch.manual_seed(0)
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    device = torch.device(f'cuda:{gpu}')
    ddp_model.to(gpu)
    
    #dataset_train = TextDatasetHugginFace(dataset = '', block_size = 512, has_tokens = True, tokenizer = tokenizer)
    filelist = getListOfFiles('/notebook/greenAI/owt_files/openwebtext/texts')
    dataset_train = FileListDataset.from_filelist(filelist=filelist, tokenizer=tokenizer, seq_len=512, current_proc=gpu, n_proc=args.n_gpu)
    
    dataset_valid = TextDataset(tokenizer=tokenizer, 
                                file_path="/notebook/greenAI/wikitext-103/wiki.valid.tokens", 
                                block_size=512)
    
    dataset_test = TextDataset(tokenizer=tokenizer, 
                                file_path="/notebook/greenAI/wikitext-103/wiki.test.tokens", block_size=512)
    
    print ("wr5", gpu)
    
    train(args, dataset_train, dataset_valid, dataset_test, ddp_model, tokenizer, gpu)
    dist.destroy_process_group()

def main():
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True, default=200)

    args = parser.parse_args()

    #args.rank = args_out.rank
    
    args.local_rank = 0
    args.max_steps = -1
    args.per_gpu_train_batch_size = 4
    args.per_gpu_eval_batch_size = 4
    args.n_gpu = 3
    args.gradient_accumulation_steps = 32
    args.num_train_epochs = 40
    args.weight_decay = 0.0001
    args.learning_rate = 6.25e-5
    args.adam_epsilon = 1e-8
    args.warmup_steps = 8000
    args.seed = 42
    args.mlm = False
    args.device = torch.device('cuda')
    args.fp16 = False
    args.max_grad_norm = 1.0
    args.logging_steps = 200
    args.save_steps = 500
    args.evaluate_during_training = True
    args.output_dir = '/notebook/greenAI/out_tt_transformer'
    args.eval_batch_size = 16
    args.save_total_limit = 2
   
    mp.spawn(train_mp_wrapper, nprocs=args.n_gpu, args=(args,))
    
if __name__ == "__main__":
    main()