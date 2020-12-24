"""
Test the MUSKAN model.
"""

import sys
sys.path.append("../../../MUSKAN")
import time
import pickle
import argparse
import torch

import numpy as np

from torch.utils.data import DataLoader
from muskan.data import MedNLI
from muskan.model import MUSKAN
from muskan.utils import correct_predictions


def test(model, dataloader):
    """
    Test the accuracy of a model on some labelled test dataset.

    Args:
        model: The torch module on which testing must be performed.
        dataloader: A DataLoader object to iterate over some dataset.

    Returns:
        batch_time: The average time to predict the classes of a batch.
        total_time: The total time to process the whole dataset.
        accuracy: The accuracy of the model on the input data.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device

    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        count = 0
        for batch in dataloader:
            batch_start = time.time()


            premises = batch["premise"].to(device)
            prem_pad = batch["prem_pad"].to(device)
            prem_sem = batch["prem_sem"].to(device)
            premises_lengths = batch["premise_length"].to(device)
            hypotheses = batch["hypothesis"].to(device)
            hyp_pad = batch["hyp_pad"].to(device)
            hyp_sem = batch["hyp_sem"].to(device)
            hypotheses_lengths = batch["hypothesis_length"].to(device)
            abb_1_p = batch["abb_1_p"].to(device)
            abb_2_p = batch["abb_2_p"].to(device)
            abb_3_p = batch["abb_3_p"].to(device)
            abb_4_p = batch["abb_4_p"].to(device)
            abb_5_p = batch["abb_5_p"].to(device)
            con_1_p = batch["con_1_p"].to(device)
            con_2_p = batch["con_2_p"].to(device)
            con_3_p = batch["con_3_p"].to(device)
            con_4_p = batch["con_4_p"].to(device)
            con_5_p = batch["con_5_p"].to(device)
            rel_pad_0_p = batch["rel_pad_0_p"].to(device)
            rel_ent_0_p = batch["rel_ent_0_p"].to(device)
            rel_mask_0_p = batch["rel_mask_0_p"].to(device)
            rel_pad_1_p = batch["rel_pad_1_p"].to(device)
            rel_ent_1_p = batch["rel_ent_1_p"].to(device)
            rel_mask_1_p = batch["rel_mask_1_p"].to(device)
            rel_pad_2_p = batch["rel_pad_2_p"].to(device)
            rel_ent_2_p = batch["rel_ent_2_p"].to(device)
            rel_mask_2_p = batch["rel_mask_2_p"].to(device)
            rel_pad_3_p = batch["rel_pad_3_p"].to(device)
            rel_ent_3_p = batch["rel_ent_3_p"].to(device)
            rel_mask_3_p = batch["rel_mask_3_p"].to(device)
            rel_pad_4_p = batch["rel_pad_4_p"].to(device)
            rel_ent_4_p = batch["rel_ent_4_p"].to(device)
            rel_mask_4_p = batch["rel_mask_4_p"].to(device)
            rel_pad_5_p = batch["rel_pad_5_p"].to(device)
            rel_ent_5_p = batch["rel_ent_5_p"].to(device)
            rel_mask_5_p = batch["rel_mask_5_p"].to(device)
            abb_1_h = batch["abb_1_h"].to(device)
            abb_2_h = batch["abb_2_h"].to(device)
            abb_3_h = batch["abb_3_h"].to(device)
            abb_4_h = batch["abb_4_h"].to(device)
            abb_5_h = batch["abb_5_h"].to(device)
            con_1_h = batch["con_1_h"].to(device)
            con_2_h = batch["con_2_h"].to(device)
            con_3_h = batch["con_3_h"].to(device)
            con_4_h = batch["con_4_h"].to(device)
            con_5_h = batch["con_5_h"].to(device)
            rel_pad_0_h = batch["rel_pad_0_h"].to(device)
            rel_ent_0_h = batch["rel_ent_0_h"].to(device)
            rel_mask_0_h = batch["rel_mask_0_h"].to(device)
            rel_pad_1_h = batch["rel_pad_1_h"].to(device)
            rel_ent_1_h = batch["rel_ent_1_h"].to(device)
            rel_mask_1_h = batch["rel_mask_1_h"].to(device)
            rel_pad_2_h = batch["rel_pad_2_h"].to(device)
            rel_ent_2_h = batch["rel_ent_2_h"].to(device)
            rel_mask_2_h = batch["rel_mask_2_h"].to(device)
            rel_pad_3_h = batch["rel_pad_3_h"].to(device)
            rel_ent_3_h = batch["rel_ent_3_h"].to(device)
            rel_mask_3_h = batch["rel_mask_3_h"].to(device)
            rel_pad_4_h = batch["rel_pad_4_h"].to(device)
            rel_ent_4_h = batch["rel_ent_4_h"].to(device)
            rel_mask_4_h = batch["rel_mask_4_h"].to(device)
            rel_pad_5_h = batch["rel_pad_5_h"].to(device)
            rel_ent_5_h = batch["rel_ent_5_h"].to(device)
            rel_mask_5_h = batch["rel_mask_5_h"].to(device)            
            weight_rel = batch["weight_rel"].to(device)
            labels = batch["label"].to(device)

            logits, probs = model(premises,
                              prem_pad,
                              prem_sem,
                              premises_lengths,
                              hypotheses,
                              hyp_pad,
                              hyp_sem,
                              hypotheses_lengths,
                              abb_1_p,
                              abb_2_p,
                              abb_3_p,
                              abb_4_p,
                              abb_5_p,
                              con_1_p,
                              con_2_p,
                              con_3_p,
                              con_4_p,
                              con_5_p,
                              rel_pad_0_p,
                              rel_ent_0_p,
                              rel_mask_0_p,
                              rel_pad_1_p,
                              rel_ent_1_p,
                              rel_mask_1_p,
                              rel_pad_2_p,
                              rel_ent_2_p,
                              rel_mask_2_p,
                              rel_pad_3_p,
                              rel_ent_3_p,
                              rel_mask_3_p,
                              rel_pad_4_p,
                              rel_ent_4_p,
                              rel_mask_4_p,
                              rel_pad_5_p,
                              rel_ent_5_p,
                              rel_mask_5_p,
                              abb_1_h,
                              abb_2_h,
                              abb_3_h,
                              abb_4_h,
                              abb_5_h,
                              con_1_h,
                              con_2_h,
                              con_3_h,
                              con_4_h,
                              con_5_h,
                              rel_pad_0_h,
                              rel_ent_0_h,
                              rel_mask_0_h,
                              rel_pad_1_h,
                              rel_ent_1_h,
                              rel_mask_1_h,
                              rel_pad_2_h,
                              rel_ent_2_h,
                              rel_mask_2_h,
                              rel_pad_3_h,
                              rel_ent_3_h,
                              rel_mask_3_h,
                              rel_pad_4_h,
                              rel_ent_4_h,
                              rel_mask_4_h,
                              rel_pad_5_h,
                              rel_ent_5_h,
                              rel_mask_5_h,
                              weight_rel)
        
            accuracy += correct_predictions(probs, labels)
            batch_time += time.time() - batch_start

    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= (len(dataloader.dataset))

    return batch_time, total_time, accuracy


def main(test_file, pretrained_file, batch_size=32):
    """
    Test the MUSKAN model with pretrained weights on some dataset.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for testing ", 20 * "=")

    checkpoint = torch.load(pretrained_file)

    # Retrieving model parameters from checkpoint.
    emb_dim = checkpoint["model"]["_geb.weight"].size(2)
    hid = checkpoint["model"]["_geb.weight"].size(2)
    bilstm_hid = int(hid/2)
    sem_emb = checkpoint["model"]["_geb.weight"].size(1)
    max_len = checkpoint["model"]["_geb.weight"].size(0)
    num_classes = checkpoint["model"]["_classification.4.weight"].size(0) 

    print("\t* Loading test data...")
    with open(test_file, "rb") as pkl:
        test_data = MedNLI(pickle.load(pkl))

    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    print("\t* Building model...")
    model = MUSKAN(
                 emb_dim,
                 hid,
                 bilstm_hid,
                 sem_emb,
                 max_len,
                 num_classes=num_classes,
                 device=device).to(device)

    model.load_state_dict(checkpoint["model"]) 

    print(20 * "=",
          " Testing MUSKAN model on device: {} ".format(device),
          20 * "=")
    batch_time, total_time, accuracy = test(model, test_loader)

    print("-> Average batch processing time: {:.4f}s, total test time:\
 {:.4f}s, accuracy: {:.4f}%".format(batch_time, total_time, (accuracy*100)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the MUSKAN model on\
 MedNLI test dataset")
    parser.add_argument("test_data",
                        help="Path to a file containing preprocessed test data")
    parser.add_argument("checkpoint",
                        help="Path to a checkpoint with a pretrained model")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size to use during testing")
    args = parser.parse_args()

    main(args.test_data,
         args.checkpoint,
         args.batch_size)
