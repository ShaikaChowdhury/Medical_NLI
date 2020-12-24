"""
Definition of the MUSKAN model.
"""

import torch
import torch.nn as nn

from .layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention, Seq2SeqEncoder_kb, graph_embed_block
from .utils import get_mask, replace_masked
import numpy as np


class MUSKAN(nn.Module):


    def __init__(self,
                 embedding_dim,
                 hidden_size,
                 bilstm_size,
                 sem_emb,
                 max_hypothesis_length,
                 dropout=0.5,
                 num_classes=3,
                 device="cpu"):
        """
        Args:
            embedding_dim: The dimension of the word embeddings.
            hidden_size: The size of all the hidden layers in the network.
            bilstm_size: The size of the bilstm hidden units
            dropout: The dropout rate to use between the layers of the network.
                A dropout rate of 0 corresponds to using no dropout at all.
                Defaults to 0.5.
            num_classes: The number of classes in the output of the network.
                Defaults to 3.
            device: The name of the device on which the model is being
                executed. Defaults to 'cpu'.
        """
        super(MUSKAN, self).__init__()

        self.proj_dim_0 = 300
        self.proj_dim_1 = 150
        self.proj_dim = 50

        self.embedding_dim = self.proj_dim_0
        self.hidden_size = self.proj_dim_0
        self.bilstm_size = self.proj_dim_1
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device
        self.sem_emb = sem_emb
        self.max_hypothesis_length = max_hypothesis_length


        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)

        self._encoding = Seq2SeqEncoder(nn.LSTM,
                                        self.embedding_dim,
                                        self.hidden_size,
                                        bidirectional=True)

        self._abb = Seq2SeqEncoder_kb(nn.LSTM,
                                        self.embedding_dim,
                                        self.hidden_size,
                                        self.bilstm_size,
                                        bidirectional=True)

        self._attention = SoftmaxAttention()

        self._geb = graph_embed_block(self.embedding_dim, self.sem_emb, self.max_hypothesis_length)

        self._projection_inp = nn.Sequential(nn.Linear(self.embedding_dim,
                                                   self.proj_dim_0),
                                         nn.ReLU()) 


        self._projection = nn.Sequential(nn.Linear(7*self.proj_dim,
                                                   self.proj_dim),
                                         nn.ReLU())


        self._projection_0 = nn.Sequential(nn.Linear(self.embedding_dim,
                                                   self.proj_dim),
                                         nn.ReLU())                         

        self._composition = Seq2SeqEncoder(nn.LSTM,
                                           self.proj_dim,
                                           self.proj_dim,
                                           bidirectional=True)

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(8*self.proj_dim,
                                                       self.proj_dim),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.proj_dim,
                                                       self.num_classes))

        # Initialize all weights and biases in the model.
        self.apply(_init_muskan_weights)

    def forward(self,
                premises,
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
                ):

        premises_mask = get_mask(prem_pad, premises_lengths).to(self.device)
        hypotheses_mask = get_mask(hyp_pad, hypotheses_lengths)\
            .to(self.device)

        embedded_premises = self._projection_inp(premises)
        embedded_hypotheses = self._projection_inp(hypotheses)
        embedded_prem_sem = self._projection_inp(prem_sem)
        embedded_hyp_sem = self._projection_inp(hyp_sem)
        embedded_con_1_p = self._projection_inp(con_1_p)
        embedded_con_1_h = self._projection_inp(con_1_h)
        embedded_con_2_p = self._projection_inp(con_2_p)
        embedded_con_2_h = self._projection_inp(con_2_h)
        embedded_con_3_p = self._projection_inp(con_3_p)
        embedded_con_3_h = self._projection_inp(con_3_h)
        embedded_con_4_p = self._projection_inp(con_4_p)
        embedded_con_4_h = self._projection_inp(con_4_h)
        embedded_con_5_p = self._projection_inp(con_5_p)
        embedded_con_5_h = self._projection_inp(con_5_h)
        embedded_abb_1_p = self._projection_inp(abb_1_p)
        embedded_abb_1_h = self._projection_inp(abb_1_h)
        embedded_abb_2_p = self._projection_inp(abb_2_p)
        embedded_abb_2_h = self._projection_inp(abb_2_h)
        embedded_abb_3_p = self._projection_inp(abb_3_p)
        embedded_abb_3_h = self._projection_inp(abb_3_h)
        embedded_abb_4_p = self._projection_inp(abb_4_p)
        embedded_abb_4_h = self._projection_inp(abb_4_h)
        embedded_abb_5_p = self._projection_inp(abb_5_p)
        embedded_abb_5_h = self._projection_inp(abb_5_h)
        embedded_rel_pad_0_p = self._projection_inp(rel_pad_0_p)
        embedded_rel_pad_0_h = self._projection_inp(rel_pad_0_h)
        embedded_rel_pad_1_p = self._projection_inp(rel_pad_1_p)
        embedded_rel_pad_1_h = self._projection_inp(rel_pad_1_h)
        embedded_rel_pad_2_p = self._projection_inp(rel_pad_2_p)
        embedded_rel_pad_2_h = self._projection_inp(rel_pad_2_h)
        embedded_rel_pad_3_p = self._projection_inp(rel_pad_3_p)
        embedded_rel_pad_3_h = self._projection_inp(rel_pad_3_h)
        embedded_rel_pad_4_p = self._projection_inp(rel_pad_4_p)
        embedded_rel_pad_4_h = self._projection_inp(rel_pad_4_h)
        embedded_rel_pad_5_p = self._projection_inp(rel_pad_5_p)
        embedded_rel_pad_5_h = self._projection_inp(rel_pad_5_h)
        embedded_rel_ent_0_p = self._projection_inp(rel_ent_0_p)
        embedded_rel_ent_0_h = self._projection_inp(rel_ent_0_h)
        embedded_rel_ent_1_p = self._projection_inp(rel_ent_1_p)
        embedded_rel_ent_1_h = self._projection_inp(rel_ent_1_h)
        embedded_rel_ent_2_p = self._projection_inp(rel_ent_2_p)
        embedded_rel_ent_2_h = self._projection_inp(rel_ent_2_h)
        embedded_rel_ent_3_p = self._projection_inp(rel_ent_3_p)
        embedded_rel_ent_3_h = self._projection_inp(rel_ent_3_h)
        embedded_rel_ent_4_p = self._projection_inp(rel_ent_4_p)
        embedded_rel_ent_4_h = self._projection_inp(rel_ent_4_h)
        embedded_rel_ent_5_p = self._projection_inp(rel_ent_5_p)
        embedded_rel_ent_5_h = self._projection_inp(rel_ent_5_h)
        embedded_premises = self._projection_inp(premises)
        embedded_premises = self._projection_inp(premises)
        embedded_premises = self._projection_inp(premises)
        embedded_premises = self._projection_inp(premises)
        embedded_premises = self._projection_inp(premises)
        embedded_premises = self._projection_inp(premises)
        embedded_premises = self._projection_inp(premises)
        embedded_premises = self._projection_inp(premises)
        embedded_premises = self._projection_inp(premises)

        
        embedded_prem_sem = embedded_prem_sem.sum(2) # (N, seq_len, embd_size)
        embedded_hyp_sem = embedded_hyp_sem.sum(2) # (N, seq_len, embd_size)

        embedded_abb_1_p = embedded_abb_1_p.sum(2) # (N, seq_len, embd_size)
        embedded_abb_1_h = embedded_abb_1_h.sum(2) # (N, seq_len, embd_size)

        embedded_abb_2_p = embedded_abb_2_p.sum(2) # (N, seq_len, embd_size)
        embedded_abb_2_h = embedded_abb_2_h.sum(2) # (N, seq_len, embd_size)
        embedded_abb_3_p = embedded_abb_3_p.sum(2) # (N, seq_len, embd_size)
        embedded_abb_3_h = embedded_abb_3_h.sum(2) # (N, seq_len, embd_size)
        embedded_abb_4_p = embedded_abb_4_p.sum(2) # (N, seq_len, embd_size)
        embedded_abb_4_h = embedded_abb_4_h.sum(2) # (N, seq_len, embd_size)
        embedded_abb_5_p = embedded_abb_5_p.sum(2) # (N, seq_len, embd_size)
        embedded_abb_5_h = embedded_abb_5_h.sum(2) # (N, seq_len, embd_size)


      
        if self.dropout:
            embedded_premises = self._rnn_dropout(embedded_premises)
            embedded_hypotheses = self._rnn_dropout(embedded_hypotheses)


        encoded_graph_emb_0_p = self._geb(embedded_rel_pad_0_p, embedded_rel_ent_0_p,
                                          rel_mask_0_p, weight_rel)

        encoded_graph_emb_0_h = self._geb(embedded_rel_pad_0_h, embedded_rel_ent_0_h,
                                          rel_mask_0_h, weight_rel)
        
        encoded_graph_emb_1_p = self._geb(embedded_rel_pad_1_p, embedded_rel_ent_1_p,
                                          rel_mask_1_p, weight_rel)

        encoded_graph_emb_1_h = self._geb(embedded_rel_pad_1_h, embedded_rel_ent_1_h,
                                          rel_mask_1_h, weight_rel)

        encoded_graph_emb_2_p = self._geb(embedded_rel_pad_2_p, embedded_rel_ent_2_p,
                                          rel_mask_2_p, weight_rel)

        encoded_graph_emb_2_h = self._geb(embedded_rel_pad_2_h, embedded_rel_ent_2_h,
                                          rel_mask_2_h, weight_rel)

        encoded_graph_emb_3_p = self._geb(embedded_rel_pad_3_p, embedded_rel_ent_3_p,
                                          rel_mask_3_p, weight_rel)

        encoded_graph_emb_3_h = self._geb(embedded_rel_pad_3_h, embedded_rel_ent_3_h,
                                          rel_mask_3_h, weight_rel)

        encoded_graph_emb_4_p = self._geb(embedded_rel_pad_4_p, embedded_rel_ent_4_p,
                                          rel_mask_4_p, weight_rel)

        encoded_graph_emb_4_h = self._geb(embedded_rel_pad_4_h, embedded_rel_ent_4_h,
                                          rel_mask_4_h, weight_rel)

        encoded_graph_emb_5_p = self._geb(embedded_rel_pad_5_p, embedded_rel_ent_5_p,
                                          rel_mask_5_p, weight_rel)

        encoded_graph_emb_5_h = self._geb(embedded_rel_pad_5_h, embedded_rel_ent_5_h,
                                          rel_mask_5_h, weight_rel)

        encoded_premises, alpha_p, beta_1_p, beta_2_p, beta_3_p, beta_4_p, beta_5_p = self._abb(embedded_premises, embedded_prem_sem,
                                          premises_lengths, embedded_abb_1_p, embedded_abb_2_p
                                          , embedded_abb_3_p, embedded_abb_4_p, embedded_abb_5_p
                                          , embedded_con_1_p, embedded_con_2_p, embedded_con_3_p
                                          , embedded_con_4_p, embedded_con_5_p)

        
        encoded_hypotheses, alpha_h, beta_1_h, beta_2_h, beta_3_h, beta_4_h, beta_5_h = self._abb(embedded_hypotheses, embedded_hyp_sem,
                                            hypotheses_lengths, embedded_abb_1_h, embedded_abb_2_h
                                            , embedded_abb_3_h, embedded_abb_4_h, embedded_abb_5_h
                                            , embedded_con_1_h, embedded_con_2_h, embedded_con_3_h
                                            , embedded_con_4_h, embedded_con_5_h)        


        gr_emb_0_p = alpha_p*encoded_graph_emb_0_p
        gr_emb_1_p = beta_1_p*encoded_graph_emb_1_p
        gr_emb_2_p = beta_2_p*encoded_graph_emb_2_p
        gr_emb_3_p = beta_3_p*encoded_graph_emb_3_p
        gr_emb_4_p = beta_4_p*encoded_graph_emb_4_p
        gr_emb_5_p = beta_5_p*encoded_graph_emb_5_p
      

        gr_emb_p = gr_emb_0_p + gr_emb_1_p + gr_emb_2_p + gr_emb_3_p+ gr_emb_4_p+ gr_emb_5_p

        gr_emb_0_h = alpha_h*encoded_graph_emb_0_h
        gr_emb_1_h = beta_1_h*encoded_graph_emb_1_h
        gr_emb_2_h = beta_2_h*encoded_graph_emb_2_h
        gr_emb_3_h = beta_3_h*encoded_graph_emb_3_h
        gr_emb_4_h = beta_4_h*encoded_graph_emb_4_h
        gr_emb_5_h = beta_5_h*encoded_graph_emb_5_h
        

        gr_emb_h = gr_emb_0_h + gr_emb_1_h + gr_emb_2_h + gr_emb_3_h+ gr_emb_4_h+ gr_emb_5_h
        
        tot_attn_p, tot_attn_h, attended_premises, attended_premises_g, attended_hypotheses, attended_hypotheses_g, prem_hyp_attn, hyp_prem_attn, prem_hyp_attn_g, hyp_prem_attn_g =\
            self._attention(encoded_premises, premises_mask,
                            encoded_hypotheses, hypotheses_mask, gr_emb_p, gr_emb_h)

        attended_premises, attended_hypotheses, tot_attn_p, tot_attn_h, attended_p_g, attended_h_g, prem_hyp_attn, hyp_prem_attn, prem_hyp_attn_g, hyp_prem_attn_g =\
            self._attention(encoded_premises, premises_mask,
                            encoded_hypotheses, hypotheses_mask, gr_emb_p, gr_emb_h)
        

        p_encoded_premises = self._projection_0(encoded_premises)
        p_gr_emb_p = self._projection_0(gr_emb_p)
        p_tot_attn_p = self._projection_0(tot_attn_p)
        enhanced_premises = torch.cat([p_encoded_premises,
                                       p_gr_emb_p,
                                       p_tot_attn_p,
                                       p_encoded_premises - p_tot_attn_p,
                                       p_gr_emb_p - p_tot_attn_p,
                                       p_encoded_premises * p_tot_attn_p, 
                                       p_gr_emb_p * p_tot_attn_p],
                                      dim=-1)

        p_encoded_hypotheses = self._projection_0(encoded_hypotheses)
        p_gr_emb_h = self._projection_0(gr_emb_h)
        p_tot_attn_h = self._projection_0(tot_attn_h)
        enhanced_hypotheses = torch.cat([p_encoded_hypotheses,
                                         p_gr_emb_h,
                                         p_tot_attn_h,
                                         p_encoded_hypotheses - p_tot_attn_h,
                                         p_gr_emb_h - p_tot_attn_h,
                                         p_encoded_hypotheses * p_tot_attn_h, 
                                         p_gr_emb_h * p_tot_attn_h],
                                         dim=-1)
        projected_premises = self._projection(enhanced_premises)
        projected_hypotheses = self._projection(enhanced_hypotheses)

        if self.dropout:
            projected_premises = self._rnn_dropout(projected_premises)
            projected_hypotheses = self._rnn_dropout(projected_hypotheses)

        v_ai = self._composition(projected_premises, premises_lengths)
        v_bj = self._composition(projected_hypotheses, hypotheses_lengths)

        v_a_avg = torch.sum(v_ai * premises_mask.unsqueeze(1)
                                                .transpose(2, 1), dim=1)\
            / torch.sum(premises_mask, dim=1, keepdim=True)

        v_b_avg = torch.sum(v_bj * hypotheses_mask.unsqueeze(1)
                                                  .transpose(2, 1), dim=1)\
            / torch.sum(hypotheses_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)
        
        logits = self._classification(v)
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits, probabilities


def _init_muskan_weights(module):

    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0
