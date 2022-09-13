from datasets import load_from_disk

import argparse
import os

from datasets import load_dataset
from transformers import TextDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.multiprocessing as mp

from torch.utils.data import Dataset
import sys


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
        has_tokens = True,
        tokens_path ='/notebook/greenAI/openwebtext2/train.tokens'
    ):
        self.examples = []
        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)
        if (not has_tokens):
            print ("tokenization start")
            dataset_train = dataset.map(tokenize_function, batched=True)
            tokenized_text = dataset_train
            tokenized_text.save_to_disk(tokens_path)
            print ("tokenization end")
        tokenized_text = load_from_disk(tokens_path)
        dataset_block_size = block_size*100
        for j in range(0, len(tokenized_text) - dataset_block_size + 1, dataset_block_size):
            block_dataset = flatten(tokenized_text[j : j + dataset_block_size]['input_ids'])
            trash = 500000
            if (j > trash):
                print (j, flush = True)
                trash +=trash
            for i in range(0, len(block_dataset) - block_size + 1, block_size):  # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(block_dataset[i : i + block_size]))
            
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)
