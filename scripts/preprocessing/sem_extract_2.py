
import argparse
import collections
import logging
import json
import re
from pymetamap import MetaMap
import sys
import pickle
from collections import OrderedDict
import random
import numpy as np
import scipy.sparse as sp

import torch
from transformers import *

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.nn.utils.rnn import pad_sequence

def seqPad(words_doc):
    if len(words_doc) < 51:
        words_doc = list(words_doc)
        word_pad = [0] * (50 - len(words_doc))
        words_doc = words_doc + word_pad
    return words_doc


def pad_mask(pad_tok_lis):
    pad_mask_lis = []
    for seq in pad_tok_lis:
        seq_mask = [float(i != 0) for i in seq] 
        pad_mask_lis.append(seq_mask)
    return pad_mask_lis


class sentDataset(Dataset):
    def __init__(self, sentLis):
        self.samples = []
        for sent in sentLis:
            self.samples.append(sent)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



vocab_sem = np.load('all_sem_lis.npy')
vocab_sem = vocab_sem.tolist()
index_to_word_sem = [x for x in vocab_sem]
word_to_index_sem = dict([(w,i) for i,w in enumerate(index_to_word_sem)])

sent_lis = np.load('sent_lis.npy')
sent_lis = sent_lis.tolist()
sent_len_lis = [len(sent.split()) for sent in sent_lis]
max_sent = max(sent_len_lis)

sem_lis = np.load('sem_lis.npy')
sem_lis = sem_lis.tolist()
sem_len_lis = [len(sem) for sem in sem_lis]
max_sem = max(sem_len_lis)

sem_type_lis = np.load('sem_type_lis.npy')
sem_type_lis = sem_type_lis.tolist()
sem_type_len_lis = [len(sem) for sem in sem_type_lis]
max_sem_type = max(sem_type_len_lis)

model_class = BertModel
tokenizer_class = BertTokenizer
pretrained_weights = 'biobert_v1.1_pubmed'

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)


emb_lis = np.empty((0, 50, 768)) 

Dataset = sentDataset(sent_lis)
dataloader = DataLoader(Dataset, batch_size=5, shuffle=False, num_workers=2)


for i, batch in enumerate(dataloader):
    batches = [sent.split('?') for sent in batch]
    input_ids = list(map(lambda t: torch.tensor(tokenizer.encode(t, add_special_tokens=True)), batches))
    input_ids_new = [id_tens.detach().numpy()[:] for id_tens in input_ids]
    input_text_padded = [seqPad(text) for text in input_ids_new]
    tokens_tensor = torch.LongTensor(input_text_padded)
    input_ids_mask = pad_mask(input_text_padded)
    tokens_mask_tensor = torch.LongTensor(input_ids_mask)
    sentence_embedding = model(tokens_tensor, attention_mask=tokens_mask_tensor)[0]
    tan_sent_emb_np = sentence_embedding.detach().numpy()[:]

    for j in range(len(batch)):
        emb_lis = np.append(emb_lis, [tan_sent_emb_np[j]], axis=0)


np.save('mednli_pre_data', emb_lis)





