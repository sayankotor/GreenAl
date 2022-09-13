import torch
from torch.nn import functional as F
import numpy as np
# from transformers import BertModel, BertConfig, BertTokenizer, BertLMHeadModel, BertTokenizerFast
from datasets import load_dataset

import os
import sys
from tqdm import tqdm
import random
import math

import matplotlib.pyplot as plt

from scaletorch import stfs


def getListOfFiles(dirName):
    """ 
    Creates a list of file and sub directories  names in the given directory 
    Parameters:
    ----------
        dirName: str
            Path to directory with files.
        
    Returns:
    -------
        allFiles: list
            list of names of files in the directory
    """
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

class FileListIteratorMultiproc:
    
    """
    Class to iterate over text files in the given filelist considering multiprocessing.
    
    Attributes:
    ----------
    
    filelist: filelist
        List of text files over which Dataset is created.
    nfiles: int
        Number of files in filelist
    n_proc: int
        Number of processes which work with data.
    current_proc: int
        Number of process which Iterator object works in
    """
    
    def __init__(self, filelist, current_proc, n_proc):
        self.filelist = filelist
        self.nfiles = len(self.filelist)
        self.n_proc = n_proc
        self.current_proc = current_proc
        
    def __iter__(self):
        """
        Iterates over files in filelist and reads text in files.
        
        Returns:
        -------
            Yields (str) - text in current txt-file.
        
        """
        self.fileidx = 3060000
        self.shift = 1
        if self.n_proc > 1:
            #per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = self.current_proc
            self.fileidx = self.fileidx + worker_id
            self.shift = self.n_proc
            print ("multiproc")
            print ("worker id", worker_id)
            print ("self.fileidx ", self.filelist[self.fileidx])
            print ("shift", self.shift)
            print ("next", self.filelist[self.fileidx + self.shift])
            print ("\n\n")
            
        self.fin = open(self.filelist[self.fileidx], "r")
        # single-process data loading, return the full iterator
        while True:
            line = self.fin.readline()
            # self.lines += 1
            if line == "":
                # reached EOF
                # print('reached eof of file', self.fileidx, self.nfiles, self.lines)
                # self.lines = 0
                self.fin.close()
                self.fileidx += self.shift
                if self.fileidx > self.nfiles - self.n_proc:
                    # end of filelist
                    # print('reached end of filelist', self.fileidx)
                    break
                else:
                    self.fin = open(self.filelist[self.fileidx], "r")
                    line = self.fin.readline()
                    yield line.strip("\n")
            else:
                yield line.strip("\n")
                
class FileListIteratorMultiprocScaletorch:
    
    """
    Class to iterate over text files in the given filelist considering multiprocessing.
    
    Attributes:
    ----------
    
    filelist: filelist
        List of text files over which Dataset is created.
    nfiles: int
        Number of files in filelist
    n_proc: int
        Number of processes which work with data.
    current_proc: int
        Number of process which Iterator object works in
    """
    
    def __init__(self, filelist, current_proc, n_proc):
        self.filelist = filelist
        self.nfiles = len(self.filelist)
        self.n_proc = n_proc
        self.current_proc = current_proc
        
    def __iter__(self):
        """
        Iterates over files in filelist and reads text in files.
        
        Returns:
        -------
            Yields (str) - text in current txt-file.
        
        """
        self.fileidx = 0
        self.shift = 1
        if self.n_proc > 1:
            #per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = self.current_proc
            self.fileidx = self.fileidx + worker_id
            self.shift = self.n_proc
            print ("multiproc")
            print ("worker id", worker_id)
            print ("self.fileidx ", self.filelist[self.fileidx])
            print ("shift", self.shift)
            print ("next", self.filelist[self.fileidx + self.shift])
            print ("\n\n")
            
        self.fin = stfs.open('s3://' + self.filelist[self.fileidx], "r")
        # single-process data loading, return the full iterator
        while True:
            line = self.fin.readline()
            # self.lines += 1
            if line == "":
                # reached EOF
                # print('reached eof of file', self.fileidx, self.nfiles, self.lines)
                # self.lines = 0
                self.fin.close()
                self.fileidx += self.shift
                if self.fileidx > self.nfiles - self.n_proc:
                    # end of filelist
                    # print('reached end of filelist', self.fileidx)
                    break
                else:
                    self.fin = stfs.open('s3://' + self.filelist[self.fileidx], "r")
                    line = self.fin.readline()
                    yield line.strip("\n")
            else:
                yield line.strip("\n")
                
class FileListDataset(torch.utils.data.IterableDataset):
    """Class for create an iterable-style dataset from directory with text files
    
        Attributes:
        ----------
            _iterator: FileListIteratorMultiproc
               An object that iterates over text files considering multiprocessing. 
            _tokenizer: transformers.PreTrainedTokenizer
               The object for tokenization text in the file
            seq_len: int
               The length of yielded objects.
            filelist: filelist
               List of text files over which Dataset is created.
            
        
        """
    def __init__(self, iterator, tokenizer, seq_len, filelist = ''):
        self._iterator = iterator
        self._tokenizer =tokenizer
        self.seq_len = seq_len
        self.filelist = filelist
        
    
    @classmethod
    def from_filelist(cls, filelist, tokenizer, seq_len, current_proc=0, n_proc=1):
        """
        Create a dataset from the given file list.
        Parameters:
        ----------
            filelist: str
                The filelist from which Dataset will be created
            tokenizer: transformers.tokenizer
                The tokenizer for text processing
            seq_len: int
                The length of yielding tokenized text object
            current_proc: int
                The number of the process inside which Dataset is created (in case of distributed training)
            n_proc: int
                 Total number of processes, working with filelist (in case of distributed training)
        Returns:
        -------
            FileListDataset
        """
        
        worker_info = torch.utils.data.get_worker_info()
        print ("worker_info", worker_info)
        iterator = FileListIteratorMultiprocScaletorch(filelist=filelist, current_proc=current_proc, n_proc=n_proc)
        return cls(
            iterator=iterator,
            tokenizer=tokenizer,
            seq_len=seq_len,
            filelist = filelist
        )

    
    def __iter__(self):
        """
        Iterates over dataset objects, reading files by files subsequently.
        
        Returns:
        -------
            Yields (List[int])
        """
        ids = []
        for text in self._iterator:
            ids.extend(self._tokenizer.encode(text))
            while len(ids) >= self.seq_len:
                #if mlm
                #yield {"input_ids": ids[:self.seq_len],
                #       "labels": ids[1:self.seq_len+1]}
                yield {"input_ids": ids[:self.seq_len],
                       "labels": ids[:self.seq_len]}
                ids = ids[self.seq_len:]

    @classmethod
    def collate_fn(cls, item):
        """Collate function for DataLoader
        Parameters:
        -------
            item (List[dict[str, List[int]]])
        Returns:
        -------
            dict[str, torch.Tensor]:
        """
        keys = item[0].keys()
        dic = {
            key: torch.tensor([x[key] for x in item])
            for key in keys
        }
        return dic
    
    def __len__(self):
        """ 
        Returns number of objects in dataset.
            
        Returns:
        -------
            Int - size of filelist
        """
        return len(self.filelist)
    
    
class FileListDatasetWithMaxFiles(torch.utils.data.IterableDataset):
    """
    Class for create an iterable-style dataset from directory with text files, reading n = max_files files subsequently.
    
    Attributes:
    ----------
            _iterator: FileListIteratorMultiproc
               An object that iterates over text files considering multiprocessing. 
            _tokenizer: transformers.PreTrainedTokenizer
               The object for tokenization text in the file
            seq_len: int
               The length of yielded objects.
            filelist: filelist
               List of text files over which Dataset is created.
            max_files: int
               Number of files which should be read in common
    """
    def __init__(self, iterator, tokenizer, seq_len, filelist = '', max_files = 1):
        self._iterator = iterator
        self._tokenizer =tokenizer
        self.seq_len = seq_len
        self.filelist = filelist
        self.max_files = max_files
        
    
    @classmethod
    def from_filelist(cls, filelist, tokenizer, seq_len):
        """
        Create a dataset from the given file list.
        Parameters:
        ----------
            filelist: str
                The filelist from which Dataset will be created
            tokenizer: transformers.tokenizer
                The tokenizer for text processing
            seq_len: int
                The length of yielding tokenized text object
            current_proc: int
                The number of the process inside which Dataset is created (in case of distributed training)
            n_proc: int
                 Total number of processes, working with filelist (in case of distributed training)
        Returns:
        -------
            FileListDataset
        """
        worker_info = torch.utils.data.get_worker_info()
        print ("worker_info", worker_info)
        iterator = FileListIteratorMultiproc(filelist=filelist, current_proc=current_proc, n_proc=n_proc)
        return cls(
            iterator=iterator,
            tokenizer=tokenizer,
            seq_len=seq_len,
            filelist = filelist
        )
        iterator = FileListIterator(filelist=filelist)
        return cls(
            iterator=iterator,
            tokenizer=tokenizer,
            seq_len=seq_len,
            filelist = filelist
        )
    
    def __iter__(self):
        """
        Iterates over dataset objects, reading max_files files by max_files files subsequently.
        Returns:
        -------
            Yields (List[int])
        """
        assert torch.utils.data.get_worker_info() is None
        ids = []
        global_text = ''
        nproc_files = 0
        for text in self._iterator:
            while (nproc_files < self.max_files):
                global_text += text
                nproc_files += 1
            #print (self.seq_len)
            #print (len(global_text))
            #print (global_text)
            ids.extend(self._tokenizer.encode(global_text))
            while len(ids) >= self.seq_len+1:
                print ("yield")
                print (len(ids[:self.seq_len]))
                #print (ids[:self.seq_len])
                print ("\n\n\n")
                yield {"input_ids": ids[:self.seq_len],
                       "labels": ids[:self.seq_len]}
                ids = ids[self.seq_len:]

    @classmethod
    def collate_fn(cls, item):
        """Collate function for DataLoader
        Parameters:
        -------
            item (List[dict[str, List[int]]])
        Returns:
        -------
            (dict[str, torch.Tensor]):
        """
        keys = item[0].keys()
        dic = {
            key: torch.tensor([x[key] for x in item])
            for key in keys
        }
        return dic
    
    def __len__(self):
        """
        Returns number of objects in dataset.
            
        Returns:
        -------
            Int - size of filelist
        """
        return len(self.filelist)    