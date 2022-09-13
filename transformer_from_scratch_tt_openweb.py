from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
import torch
from transformers import GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling

from datasets import load_dataset
from transformers import TextDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import Trainer, TrainingArguments
from transformers import default_data_collator

from datasets import load_dataset

import argparse
from src.classes.gpt2_tt import GPT2_TT_Model

from torch.utils.data import Dataset
import sys
sys.path.append("/notebook/transformers/src")
from transformers.tokenization_utils import PreTrainedTokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def flatten(t):
    return [item for sublist in t for item in sublist]

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


class TextDatasetHugginFace(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset: Dataset,
        block_size: int,
    ):
        self.examples = []
        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)
        print ("tokenization start")
        #tokenized_text = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(txt)) for txt in texts]
        dataset_train = dataset.map(tokenize_function, batched=True)
        tokenized_text = flatten(dataset_train['input_ids'])
        print (len(tokenized_text))
        print ("tokenization end")
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
            self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)
    
    

class Object(object):
    pass

args = Object()
args.local_rank = -1
args.max_steps = -1
args.per_gpu_train_batch_size = 8
args.per_gpu_eval_batch_size = 8
args.n_gpu = 1
args.gradient_accumulation_steps = 16
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

#import sys
#sys.path.append("/home/jovyan/") 
#sys.path.append("/home/jovyan/transformers/src") 

from transformers.configuration_utils import PretrainedConfig

class GPT2MedConfig(PretrainedConfig):
    """
    """

    model_type = "gpt2-medium"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=1024,
        n_layer=24,
        n_head=16,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)    

def main():
    
    # Initializing a GPT2 configuration
    # Initializing a GPT2 configuration
    configuration = GPT2MedConfig()
   
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
    
    print(torch.cuda.memory_allocated(device = device), flush = True)
    print(torch.cuda.memory_reserved(device = device), flush = True)
    a = model.to(device)
    print(torch.cuda.memory_allocated(device = device), flush = True)
    print(torch.cuda.memory_reserved(device = device), flush = True)
    
    #tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    #dataset_train = load_dataset('wikitext', 'wikitext-103-v1', split='train')
    #dataset_valid = load_dataset('wikitext', 'wikitext-103-v1', split='validation')
    
    #dataset_train = TextDataset(tokenizer=tokenizer, 
                                #file_path="/home/jovyan/greenAI_gpt/wikitext-103/wiki.train.tokens", 
                                #block_size=512)
    dataset_train = load_dataset('openwebtext', split = 'train')
    
    dataset_train = TextDatasetHugginFace(dataset = dataset_train, block_size = 512, tokenizer = tokenizer)


    dataset_valid = TextDataset(tokenizer=tokenizer, 
                                file_path="/notebook/greenAI/wikitext-103/wiki.valid.tokens", 
                                block_size=512)
    
    dataset_test = TextDataset(tokenizer=tokenizer, 
                                file_path="/notebook/greenAI/wikitext-103/wiki.test.tokens", block_size=512)
    print (len(dataset_train))
    
    train(args, dataset_train, dataset_valid, dataset_test, model, tokenizer)
    
if __name__ == "__main__":
    main()