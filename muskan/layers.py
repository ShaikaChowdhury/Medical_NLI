"""
Definition of the layers for MUSKAN model.
"""


import torch.nn as nn

from .utils import masked_softmax, weighted_sum

import torch

torch.manual_seed(0)
# Class widely inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/modules/input_variational_dropout.py
class RNNDropout(nn.Dropout):
    """
    Dropout layer for the inputs of RNNs.

    Apply the same dropout mask to all the elements of the same sequence in
    a batch of sequences of size (batch, sequences_length, embedding_dim).
    """

    def forward(self, sequences_batch):
        """
        Apply dropout to the input batch of sequences.

        Args:
            sequences_batch: A batch of sequences of vectors that will serve
                as input to an RNN.
                Tensor of size (batch, sequences_length, emebdding_dim).

        Returns:
            A new tensor on which dropout has been applied.
        """
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0],
                                             sequences_batch.shape[-1])
        dropout_mask = nn.functional.dropout(ones, self.p, self.training,
                                             inplace=False)
        return dropout_mask.unsqueeze(1) * sequences_batch


class Seq2SeqEncoder(nn.Module):
    """
    RNN taking variable length padded sequences of vectors as input and
    encoding them into padded sequences of vectors of the same length.

    This module is useful to handle batches of padded sequences of vectors
    that have different lengths and that need to be passed through a RNN.
    The sequences are sorted in descending order of their lengths, packed,
    passed through the RNN, and the resulting sequences are then padded and
    permuted back to the original order of the input sequences.
    """

    def __init__(self,
                 rnn_type,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bias=True,
                 dropout=0.0,
                 bidirectional=False):
        """
        Args:
            rnn_type: The type of RNN to use as encoder in the module.
                Must be a class inheriting from torch.nn.RNNBase
                (such as torch.nn.LSTM for example).
            input_size: The number of expected features in the input of the
                module.
            hidden_size: The number of features in the hidden state of the RNN
                used as encoder by the module.
            num_layers: The number of recurrent layers in the encoder of the
                module. Defaults to 1.
            bias: If False, the encoder does not use bias weights b_ih and
                b_hh. Defaults to True.
            dropout: If non-zero, introduces a dropout layer on the outputs
                of each layer of the encoder except the last one, with dropout
                probability equal to 'dropout'. Defaults to 0.0.
            bidirectional: If True, the encoder of the module is bidirectional.
                Defaults to False.
        """
        assert issubclass(rnn_type, nn.RNNBase),\
            "rnn_type must be a class inheriting from torch.nn.RNNBase"

        super(Seq2SeqEncoder, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional

        self._encoder = rnn_type(input_size,
                                 hidden_size,
                                 num_layers=num_layers,
                                 bias=bias,
                                 batch_first=True,
                                 dropout=dropout,
                                 bidirectional=bidirectional)

    def forward(self, sequences_batch, sequences_lengths):
        """
        Args:
            sequences_batch: A batch of variable length sequences of vectors.
                The batch is assumed to be of size
                (batch, sequence, vector_dim).
            sequences_lengths: A 1D tensor containing the sizes of the
                sequences in the input batch.

        Returns:
            reordered_outputs: The outputs (hidden states) of the encoder for
                the sequences in the input batch, in the same order.
        """

        outputs, _ = self._encoder(sequences_batch, None)

        return outputs

class lin1_kb_block(nn.Module):
    '''Layer to compute the first term in the context vector.'''
    def __init__(self, D_in, H):
        super(lin1_kb_block, self).__init__()
        self.lin1 = torch.nn.Linear(D_in, H)
    def forward(self, x):
        activation = self.lin1(x)
        return activation

class lin2_kb_block(nn.Module):
    '''Layer to compute the second term in the context vector.'''
    def __init__(self, D_in, H):
        super(lin2_kb_block, self).__init__()
        self.lin2 = torch.nn.Linear(D_in, H)
    def forward(self, x):
        activation = self.lin2(x)
        return activation

class bilin1_kb_block(nn.Module):
    '''Layer to compute the alpha'''
    def __init__(self, in1_features, in2_features, out_features, bias):
        super(bilin1_kb_block, self).__init__()
        self.bilin1 = nn.Bilinear(in1_features, in2_features, out_features, bias)
    def forward(self, x, y):
        activation = self.bilin1(x, y)
        return activation


class bilin2_kb_block(nn.Module):
    '''Layer to compute beta'''
    def __init__(self, in1_features, in2_features, out_features, bias):
        super(bilin2_kb_block, self).__init__()
        self.bilin2 = nn.Bilinear(in1_features, in2_features, out_features, bias)
    def forward(self, x, y):
        activation = self.bilin2(x, y)
        return activation

class graph_embed_block(nn.Module):
    '''Layer to compute graph embedding'''
    def __init__(self, emb_dim, sem_emb, max_hypothesis_length):
        super(graph_embed_block, self).__init__()
        self.emb_dim = emb_dim
        self.lin_1 = nn.Linear(2*self.emb_dim, self.emb_dim)
        self.lin_2 = nn.Linear(self.emb_dim, self.emb_dim)
        self.act = nn.ReLU(inplace=False)
        self.weight = nn.Parameter(torch.Tensor(max_hypothesis_length, sem_emb, emb_dim))
        nn.init.xavier_normal_(self.weight)
    def forward(self, rel_pad, rel_ent, rel_mask, weight_rel):
        cat_rel = torch.cat((rel_pad, rel_ent), 3)
        rel_relu = self.act(self.lin_1(cat_rel)) 
        weight = self.weight
        rel_wei = self.lin_2(weight)
        rel_mult = rel_wei*rel_relu
        rel_wei_summed = rel_mult.sum(3)           
        rel_wei_prob = nn.functional.softmax(rel_wei_summed)
        rel_temp = torch.unsqueeze(rel_wei_prob, 3) * (torch.unsqueeze(rel_mask, 3) * rel_relu)        
        graph_embed = rel_temp.sum(2)
        return graph_embed 

class Seq2SeqEncoder_kb(nn.Module):
    """
    Layer to compute adaptive lexical embedding
    """

    def __init__(self,
                 rnn_type,
                 input_size,
                 hidden_size,
                 bilstm_size,
                 num_layers=1,
                 bias=True,
                 dropout=0.0,
                 bidirectional=False):

        assert issubclass(rnn_type, nn.RNNBase),\
            "rnn_type must be a class inheriting from torch.nn.RNNBase"

        super(Seq2SeqEncoder_kb, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bilstm_size = bilstm_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional

        self._encoder = rnn_type(input_size,
                                 bilstm_size,
                                 num_layers=num_layers,
                                 bias=bias,
                                 batch_first=True,
                                 dropout=dropout,
                                 bidirectional=bidirectional)

        self.lin1_kb_block = lin1_kb_block(self.hidden_size, self.hidden_size)
        self.lin2_kb_block = lin2_kb_block(self.hidden_size, self.hidden_size)
        self.bilin1_kb_block = bilin1_kb_block(self.hidden_size, self.hidden_size, self.hidden_size, bias = True)
        self.bilin2_kb_block = bilin2_kb_block(self.hidden_size, self.hidden_size, self.hidden_size, bias = True)
        self.lin_3 = nn.Linear(self.hidden_size, 1)
        self.sig = nn.Sigmoid()


    def forward(self, sequences_batch, sequences_sem, sequences_lengths, abb_1, abb_2, abb_3, abb_4,
    abb_5, con_1, con_2, con_3, con_4, con_5):


        outputs, _ = self._encoder(sequences_batch, None)

        alpha = self.bilin1_kb_block(sequences_batch, sequences_sem)
        g_term_1 = self.lin1_kb_block(outputs)
        g_term_2 = self.lin2_kb_block(sequences_batch)
        g_temp = g_term_1.add(g_term_2)
        g = self.sig(g_temp)

        beta_1 = self.bilin2_kb_block(con_1, g)
        beta_2 = self.bilin2_kb_block(con_2, g)
        beta_3 = self.bilin2_kb_block(con_3, g)
        beta_4 = self.bilin2_kb_block(con_4, g)
        beta_5 = self.bilin2_kb_block(con_5, g)
        beta = beta_1 + beta_2 + beta_3 + beta_4 + beta_5 
     
        cat_wei = torch.cat((alpha, beta), 2)
        max_wei, _ = torch.max(cat_wei, 2, keepdim=True)
        log_term = cat_wei - max_wei
        log_sum = max_wei + torch.log(torch.sum(torch.exp(log_term), dim=2, keepdim=True))

        alpha_new = alpha - log_sum
        beta_1_new = beta_1 - log_sum
        beta_2_new = beta_2 - log_sum
        beta_3_new = beta_3 - log_sum
        beta_4_new = beta_4 - log_sum
        beta_5_new = beta_5 - log_sum


        mixed_term_1 = sequences_sem*alpha_new
        mixed_term_1_2 = beta_1_new*abb_1
        mixed_term_2_2 = beta_2_new*abb_2
        mixed_term_3_2 = beta_3_new*abb_3
        mixed_term_4_2 = beta_4_new*abb_4
        mixed_term_5_2 = beta_5_new*abb_5

        mixed_term_2 = mixed_term_1_2 + mixed_term_2_2 + mixed_term_3_2+ mixed_term_4_2+ mixed_term_5_2

        m = mixed_term_1 + mixed_term_2
        emb = outputs+m 
        return emb, alpha_new, beta_1_new, beta_2_new, beta_3_new, beta_4_new, beta_5_new



class SoftmaxAttention(nn.Module):
    """
    Attention layer taking the encoded premises and hypotheses as input
    and computing the soft attention between their elements.

    """

    def forward(self,
                premise_batch,
                premise_mask,
                hypothesis_batch,
                hypothesis_mask,
                encoded_graph_emb_p,
                encoded_graph_emb_h):



        # Dot product between premises and hypotheses in each sequence of
        # the batch.
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1)
                                                              .contiguous())


        similarity_matrix_g = encoded_graph_emb_p.bmm(encoded_graph_emb_h.transpose(2, 1)
                                                              .contiguous())


        # Softmax attention weights.
        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2)
                                                        .contiguous(), premise_mask)

        prem_hyp_attn_g = masked_softmax(similarity_matrix_g, hypothesis_mask)
        hyp_prem_attn_g = masked_softmax(similarity_matrix_g.transpose(1, 2)
                                                        .contiguous(), premise_mask)

        tot_attn_p = prem_hyp_attn + prem_hyp_attn_g
        tot_attn_h = hyp_prem_attn + hyp_prem_attn_g


        attended_premises = weighted_sum(hypothesis_batch,
                                         tot_attn_p,
                                         premise_mask)
        attended_hypotheses = weighted_sum(premise_batch,
                                           tot_attn_h,
                                           hypothesis_mask)

        attended_premises_g = weighted_sum(encoded_graph_emb_h, prem_hyp_attn_g, premise_mask)
        attended_hypotheses_g = weighted_sum(encoded_graph_emb_p, hyp_prem_attn_g, hypothesis_mask)

        return tot_attn_p, tot_attn_h, attended_premises, attended_premises_g, attended_hypotheses, attended_hypotheses_g, prem_hyp_attn, hyp_prem_attn, prem_hyp_attn_g, hyp_prem_attn_g
