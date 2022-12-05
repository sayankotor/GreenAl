from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import sys
import os

sys.path.append('/notebook/greenAI/')

from src.classes.gpt_med_config import GPT2MedConfig
from src.classes.gpt2_tt import GPT2_TT_Model
configuration = GPT2MedConfig()


import argparse

def train_mp_wrapper(gpu, args):
    
    """Wraps the process of training distributed model on a single gpu.
       Registers the process, creates Data Parralell GPT model (regular or compressed), create test, valid datasets and train dataloders.
    """
    
    print ("wr", gpu, flush = True)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=gpu, world_size=1)
    print ("gpu", gpu, flush = True)
    # Initializing a GPT2 configuration
    configuration = GPT2MedConfig()
   
    model = GPT2_TT_Model(configuration, rank = args.rank)
    model.to(gpu)
    
    ddp_model = DDP(model, device_ids=[gpu], output_device = gpu)
    device = torch.device(f'cuda:{gpu}') 
    
    # loading from checkpoint
    dictt1 = torch.load("/notebook/greenAI/out_transformer_0_v4/checkpoint-6000/model_tt.pth", map_location=device)
    ddp_model.load_state_dict(dictt1)   
        
    torch.save(ddp_model.module.state_dict(), '/notebook/greenAI/out_transformer_0_v4/checkpoint-6000/model_tt.pth')

    dist.destroy_process_group()

def main():
    
    """
    The main function of an experiment module.
    Processes arguments from the command line and provides data-parallel training of GPT-based models on several GPUs.
    Every training process corresponds to a certain GPU, the process started is provided by the spawn method - creating a fresh python process with its separate own interpreter.
    The spawn point should be as close to the start of the main module process as possible. It is not recommended to add working functionality in the main process before the spawn point.
    
    The following experiment attributes are set:
    
        - training attributes:
           - max_steps - maximum number of steps in a training process
           - per_gpu_train_batch_size - batch size processed into single GPU card while training. In the corresponding experiment, settings is equivalent to train_batch_size
           - per_gpu_eval_batch_size - batch size processed into single GPU card while evaluating. In the corresponding experiment settings is equivalent to eval_batch_size
           - n_gpu - number of GPU cards involved in training. In the corresponding experiment, settings is equivalent to the number of the separate training process (i.e. "world size").
           - num_train_epochs - number of training epoch (40 -100 for relatively small datasets, 4-6 for a big one)
           - weight_decay - weight weights regularizer is added with
           - learning_rate - the size of step in gradient descent
           - adam_epsilon - epsilon parameter in case of default optimizer (AdamW)
           - warmup_steps - warmup_steps in case of scheduler
           - seed - random seed to further reproduce the experiments
           - device - cuda device
           - fp16 - is low precision is used while training (False for every experiment)
           - max_grad_norm - gradient norm
           - logging_steps - how often model will be evaluated and write results to log
           - save_steps - how often model will be saved
           - evaluate_during_training - whether the model will be evaluated during the training process (on a validation dataset) or only at the end of training process (on a test dataset)
           - output_dir - path to the directory where checkpoints (model weights, optimizer, and scheduler states)
           - eval_batch_size - batch size processed while evaluating
           - save_total_limit - number of checkpoints stored 
          
        - loading attributes:
           - from_chkpt - should the model be loaded from the checkpoint or not
           - chkpt_path - path to the checkpoint for loading (if from_chkpt is True)
           
        - model attributes:
          - rank - set therank of TT layer. If rank is 0, model is a regular GPT
              
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True, default=200)

    args = parser.parse_args()

    #args.rank = args_out.rank
    
    args.local_rank = 0
    args.max_steps = -1

    mp.spawn(train_mp_wrapper, nprocs=1, args=(args,))
    
if __name__ == "__main__":
    main()
    
    
#AdamW, β1=0.9, β2=0.95, eps=1e−8
#learning rate:
#peak=6e-5
#warmup over 183_105 samples (375M tokens)
#cosine decay for learning rate down to 10% of its value, over 410B tokens (after 410B tokens, training continues at 10% of the original learning rate, that is fixed --min-lr)
#clipping by global norm of 1 (as in GPT-3)
#weight decay of 0.1 (same as in GPT3 and 530B trainings)