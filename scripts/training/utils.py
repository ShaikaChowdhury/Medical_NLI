"""
Utility functions for training and validating models.
"""

import time
import torch

import torch.nn as nn

from tqdm import tqdm
from muskan.utils import correct_predictions

import numpy as np
def train(model,
          dataloader,
          optimizer,
          criterion,
          epoch_number,
          max_gradient_norm):
    """
    Train a model for one epoch on some input data with a given optimizer and
    criterion.

    Args:
        model: A torch module that must be trained on some input data.
        dataloader: A DataLoader object to iterate over the training data.
        optimizer: A torch optimizer to use for training on the input model.
        criterion: A loss criterion to use for training.
        epoch_number: The number of the epoch for which training is performed.
        max_gradient_norm: Max. norm for gradient norm clipping.

    Returns:
        epoch_time: The total time necessary to train the epoch.
        epoch_loss: The training loss computed for the epoch.
        epoch_accuracy: The accuracy computed for the epoch.
    """
    # Switch the model to train mode.
    model.train()
    device = model.device

    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0

    tqdm_batch_iterator = tqdm(dataloader)
    count = 0
    for batch_index, batch in enumerate(tqdm_batch_iterator):
        batch_start = time.time()

        # Move input and output data to the GPU if it is used.
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

        optimizer.zero_grad()

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
                              weight_rel
                              )


        loss = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probs, labels)

        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
                      .format(batch_time_avg/(batch_index+1),
                              running_loss/(batch_index+1))
        tqdm_batch_iterator.set_description(description)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)

    return epoch_time, epoch_loss, epoch_accuracy


def validate(model, dataloader, criterion):
    """
    Compute the loss and accuracy of a model on some validation dataset.

    Args:
        model: A torch module for which the loss and accuracy must be
            computed.
        dataloader: A DataLoader object to iterate over the validation data.
        criterion: A loss criterion to use for computing the loss.
        epoch: The number of the epoch for which validation is performed.
        device: The device on which the model is located.

    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_accuracy: The accuracy computed on the entire validation set.
    """
    # Switch to evaluate mode.
    model.eval()
    device = model.device

    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        count = 0
        for batch in dataloader:
            # Move input and output data to the GPU if one is used.

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
                          weight_rel
                                  )


            loss = criterion(logits, labels)
              
            running_loss += loss.item()
            running_accuracy += correct_predictions(probs, labels)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))

    return epoch_time, epoch_loss, epoch_accuracy
