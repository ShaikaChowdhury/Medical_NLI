"""
Dataset definition for MedNLI.
"""


import string
import torch
import numpy as np

from collections import Counter
from torch.utils.data import Dataset
import torch

class MedNLI(Dataset):
    """
    Dataset class for MedNLI dataset.
    """

    def __init__(self,
                 data):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses, lengths, umls concepts, candidate concepts, contextual evidence,
                relations and labels.
        """
        self.premises_lengths = data["premise_length"]
        self.hypotheses_lengths = data["hypothesis_length"]
        self.num_sequences = len(data["prem_pad"])
        self.max_premise_length = data["premise_length"][0]
        self.max_hypothesis_length = data["hypothesis_length"][0]
        self.sem_emb = np.asarray(data["rel_pad_p"]).shape[2]
        self.embedding_dim = 768

        weight = torch.Tensor(int(np.asarray(data["rel_pad_p"]).shape[0]), int(np.asarray(data["rel_pad_p"]).shape[1]), int(np.asarray(data["rel_pad_p"]).shape[2]), self.embedding_dim)

        self.data = {
                     "premises":torch.FloatTensor(data["prem"]),
                     "prem_pad": torch.LongTensor(data["prem_pad"]),
                     "prem_sem": torch.FloatTensor(data["prem_sem"]),
                     "hypotheses": torch.FloatTensor(data["hyp"]),
                     "hyp_pad": torch.LongTensor(data["hyp_pad"]),
                     "hyp_sem": torch.FloatTensor(data["hyp_sem"]),
                     "abb_1_p": torch.FloatTensor(data["abb_1_p"]),
                     "abb_2_p": torch.FloatTensor(data["abb_2_p"]),
                     "abb_3_p": torch.FloatTensor(data["abb_3_p"]),
                     "abb_4_p": torch.FloatTensor(data["abb_4_p"]),
                     "abb_5_p": torch.FloatTensor(data["abb_5_p"]),
                     "con_1_p": torch.FloatTensor(data["con_1_p"]),
                     "con_2_p": torch.FloatTensor(data["con_2_p"]),
                     "con_3_p": torch.FloatTensor(data["con_3_p"]),
                     "con_4_p": torch.FloatTensor(data["con_4_p"]),
                     "con_5_p": torch.FloatTensor(data["con_5_p"]), 
                     "rel_pad_0_p": torch.FloatTensor(data["rel_pad_0_p"]),
                     "rel_ent_0_p": torch.FloatTensor(data["rel_ent_0_p"]),
                     "rel_mask_0_p": torch.LongTensor(data["rel_mask_0_p"]),
                     "rel_pad_1_p": torch.FloatTensor(data["rel_pad_1_p"]),
                     "rel_ent_1_p": torch.FloatTensor(data["rel_ent_1_p"]),
                     "rel_mask_1_p": torch.LongTensor(data["rel_mask_1_p"]), 
                     "rel_pad_2_p": torch.FloatTensor(data["rel_pad_2_p"]),
                     "rel_ent_2_p": torch.FloatTensor(data["rel_ent_2_p"]),
                     "rel_mask_2_p": torch.LongTensor(data["rel_mask_2_p"]), 
                     "rel_pad_3_p": torch.FloatTensor(data["rel_pad_3_p"]),
                     "rel_ent_3_p": torch.FloatTensor(data["rel_ent_3_p"]),
                     "rel_mask_3_p": torch.LongTensor(data["rel_mask_3_p"]), 
                     "rel_pad_4_p": torch.FloatTensor(data["rel_pad_4_p"]),
                     "rel_ent_4_p": torch.FloatTensor(data["rel_ent_4_p"]),
                     "rel_mask_4_p": torch.LongTensor(data["rel_mask_4_p"]), 
                     "rel_pad_5_p": torch.FloatTensor(data["rel_pad_5_p"]),
                     "rel_ent_5_p": torch.FloatTensor(data["rel_ent_5_p"]),
                     "rel_mask_5_p": torch.LongTensor(data["rel_mask_5_p"]),                   
                     "abb_1_h": torch.FloatTensor(data["abb_1_h"]),
                     "abb_2_h": torch.FloatTensor(data["abb_2_h"]),
                     "abb_3_h": torch.FloatTensor(data["abb_3_h"]),
                     "abb_4_h": torch.FloatTensor(data["abb_4_h"]),
                     "abb_5_h": torch.FloatTensor(data["abb_5_h"]),
                     "con_1_h": torch.FloatTensor(data["con_1_h"]),
                     "con_2_h": torch.FloatTensor(data["con_2_h"]),
                     "con_3_h": torch.FloatTensor(data["con_3_h"]),
                     "con_4_h": torch.FloatTensor(data["con_4_h"]),
                     "con_5_h": torch.FloatTensor(data["con_5_h"]), 
                     "rel_pad_0_h": torch.FloatTensor(data["rel_pad_0_h"]),
                     "rel_ent_0_h": torch.FloatTensor(data["rel_ent_0_h"]),
                     "rel_mask_0_h": torch.LongTensor(data["rel_mask_0_h"]),
                     "rel_pad_1_h": torch.FloatTensor(data["rel_pad_1_h"]),
                     "rel_ent_1_h": torch.FloatTensor(data["rel_ent_1_h"]),
                     "rel_mask_1_h": torch.LongTensor(data["rel_mask_1_h"]), 
                     "rel_pad_2_h": torch.FloatTensor(data["rel_pad_2_h"]),
                     "rel_ent_2_h": torch.FloatTensor(data["rel_ent_2_h"]),
                     "rel_mask_2_h": torch.LongTensor(data["rel_mask_2_h"]),
                     "rel_pad_3_h": torch.FloatTensor(data["rel_pad_3_h"]),
                     "rel_ent_3_h": torch.FloatTensor(data["rel_ent_3_h"]),
                     "rel_mask_3_h": torch.LongTensor(data["rel_mask_3_h"]),
                     "rel_pad_4_h": torch.FloatTensor(data["rel_pad_4_h"]),
                     "rel_ent_4_h": torch.FloatTensor(data["rel_ent_4_h"]),
                     "rel_mask_4_h": torch.LongTensor(data["rel_mask_4_h"]),
                     "rel_pad_5_h": torch.FloatTensor(data["rel_pad_5_h"]),
                     "rel_ent_5_h": torch.FloatTensor(data["rel_ent_5_h"]),
                     "rel_mask_5_h": torch.LongTensor(data["rel_mask_5_h"]),
                     "weight_rel":weight, 
                     "labels": torch.tensor(data["label"], dtype=torch.long)}


    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        return {
                "premise": self.data["premises"][index],
                "prem_pad": self.data["prem_pad"][index],
                "prem_sem": self.data["prem_sem"][index],
                "premise_length": self.premises_lengths[index],
                "hypothesis": self.data["hypotheses"][index],
                "hyp_pad": self.data["hyp_pad"][index],
                "hyp_sem": self.data["hyp_sem"][index],
                "hypothesis_length":self.hypotheses_lengths[index],
                "abb_1_p": self.data["abb_1_p"][index],
                "abb_2_p": self.data["abb_2_p"][index],
                "abb_3_p": self.data["abb_3_p"][index],
                "abb_4_p": self.data["abb_4_p"][index],
                "abb_5_p": self.data["abb_5_p"][index],
                "con_1_p": self.data["con_1_p"][index],
                "con_2_p": self.data["con_2_p"][index],
                "con_3_p": self.data["con_3_p"][index],
                "con_4_p": self.data["con_4_p"][index],
                "con_5_p": self.data["con_5_p"][index],
                "rel_pad_0_p": self.data["rel_pad_0_p"][index],
                "rel_ent_0_p": self.data["rel_ent_0_p"][index],
                "rel_mask_0_p": self.data["rel_mask_0_p"][index],
                "rel_pad_1_p": self.data["rel_pad_1_p"][index],
                "rel_ent_1_p": self.data["rel_ent_1_p"][index],
                "rel_mask_1_p": self.data["rel_mask_1_p"][index],
                "rel_pad_2_p": self.data["rel_pad_2_p"][index],
                "rel_ent_2_p": self.data["rel_ent_2_p"][index],
                "rel_mask_2_p": self.data["rel_mask_2_p"][index],
                "rel_pad_3_p": self.data["rel_pad_3_p"][index],
                "rel_ent_3_p": self.data["rel_ent_3_p"][index],
                "rel_mask_3_p": self.data["rel_mask_3_p"][index],
                "rel_pad_4_p": self.data["rel_pad_4_p"][index],
                "rel_ent_4_p": self.data["rel_ent_4_p"][index],
                "rel_mask_4_p": self.data["rel_mask_4_p"][index],
                "rel_pad_5_p": self.data["rel_pad_5_p"][index],
                "rel_ent_5_p": self.data["rel_ent_5_p"][index],
                "rel_mask_5_p": self.data["rel_mask_5_p"][index],
                "abb_1_h": self.data["abb_1_h"][index],
                "abb_2_h": self.data["abb_2_h"][index],
                "abb_3_h": self.data["abb_3_h"][index],
                "abb_4_h": self.data["abb_4_h"][index],
                "abb_5_h": self.data["abb_5_h"][index],
                "con_1_h": self.data["con_1_h"][index],
                "con_2_h": self.data["con_2_h"][index],
                "con_3_h": self.data["con_3_h"][index],
                "con_4_h": self.data["con_4_h"][index],
                "con_5_h": self.data["con_5_h"][index],
                "rel_pad_0_h": self.data["rel_pad_0_h"][index],
                "rel_ent_0_h": self.data["rel_ent_0_h"][index],
                "rel_mask_0_h": self.data["rel_mask_0_h"][index],
                "rel_pad_1_h": self.data["rel_pad_1_h"][index],
                "rel_ent_1_h": self.data["rel_ent_1_h"][index],
                "rel_mask_1_h": self.data["rel_mask_1_h"][index],
                "rel_pad_2_h": self.data["rel_pad_2_h"][index],
                "rel_ent_2_h": self.data["rel_ent_2_h"][index],
                "rel_mask_2_h": self.data["rel_mask_2_h"][index],
                "rel_pad_3_h": self.data["rel_pad_3_h"][index],
                "rel_ent_3_h": self.data["rel_ent_3_h"][index],
                "rel_mask_3_h": self.data["rel_mask_3_h"][index],
                "rel_pad_4_h": self.data["rel_pad_4_h"][index],
                "rel_ent_4_h": self.data["rel_ent_4_h"][index],
                "rel_mask_4_h": self.data["rel_mask_4_h"][index],
                "rel_pad_5_h": self.data["rel_pad_5_h"][index],
                "rel_ent_5_h": self.data["rel_ent_5_h"][index],
                "rel_mask_5_h": self.data["rel_mask_5_h"][index],
                "weight_rel": self.data["weight_rel"][index],
                "label": self.data["labels"][index]}

