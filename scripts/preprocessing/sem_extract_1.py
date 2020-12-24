# -*- coding: utf-8 -*-

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

def _removeNonAscii(s): return "".join(i for i in s if ord(i)<128)

# dict_tot = OrderedDict()
# with open('mli_test_v1.jsonl') as json_file:
# #with open('mli_dev_v1.jsonl') as json_file:

#     #data = json.load(json_file)
#     json_list = list(json_file)
#     #print('json_list', json_list)
#     print('len(json_list)', len(json_list))
#     for json_str in json_list:
#         result = json.loads(json_str)
#         #print("result: {}".format(result))
#         #print(isinstance(result, dict))
#         sent_1 = result['sentence1']
#         sent_2 = result['sentence2']
#         id = result['pairID']
#         label = result['gold_label']
#         dict_tot[id] = [sent_1, sent_2, label]
#
# print('len(dict_tot)', len(dict_tot))
# # print('dict_tot', dict_tot)

# with open('sem_mednli_dict.pickle', 'rb') as handle:
# with open('sem_mednli_dict_valid.pickle', 'rb') as handle:
##with open('sem_mednli_dict_test.pickle', 'rb') as handle:
##    sem_dict = pickle.load(handle)


mm = MetaMap.get_instance('metamap18')

sem_dict_new = OrderedDict()

count = 0
temp_sem_dict = OrderedDict()
for key, value in sem_dict.items():
    if count >= 0 and count < len(sem_dict):
        temp_sem_dict[key] = value
    count = count + 1



count = 0
for key, value in temp_sem_dict.items():
    sent_1_temp = _removeNonAscii(value[0])
    sent_1_temp = sent_1_temp.replace('"', '')
    sent_1_temp = sent_1_temp.strip()
    sent_1 = [sent_1_temp]
    sent_2_temp = _removeNonAscii(value[1])
    sent_2_temp = sent_2_temp.replace('"', '')
    sent_2_temp = sent_2_temp.strip()
    sent_2 = [sent_2_temp]
    sent_1_lis = sent_1[0].split()
    sent_2_lis = sent_2[0].split()

    concepts_1, error_1 = mm.extract_concepts(sent_1)
    dict_1 = OrderedDict()
    dict_1 = dict.fromkeys(sent_1_lis,[])
    dict_2 = OrderedDict()
    dict_2 = dict.fromkeys(sent_2_lis, [])
    for concept in concepts_1:
        score = concept[2]
        pref_name = concept[3]
        cui = concept[4]
        sem_typ = concept[5]
        semt_lis = sem_typ.split(',')
        sem_type = semt_lis[0]
        sem_type = sem_type.replace('[', '')
        sem_type = sem_type.replace(']', '')
        sem_type = sem_type.strip()
        trigger = concept[6]
        t_lis = trigger.split('-')
        if len(t_lis) == 6:
            tok_l = t_lis[3]
            tok_li = tok_l.split()

            tok_li = [tok.replace('"', '') for tok in tok_li]
            tok_li = [tok.replace('(', '') for tok in tok_li]
            tok_li = [tok.replace(')', '') for tok in tok_li]
            tok_li = [tok.replace('[', '') for tok in tok_li]
            tok_li = [tok.replace(']', '') for tok in tok_li]
            tok_li = [tok.strip() for tok in tok_li]
            for tok in tok_li:
                for k,v in dict_1.items():
                    if tok == k:
                        if len(dict_1[tok]) == 0:
                            dict_1[tok] = [[score, pref_name, cui, sem_type]]
                        else:
                            dict_1[tok].append([score, pref_name, cui, sem_type])
                    elif tok == '(' or tok == ')':
                        pass
                    elif re.search(tok, k) is None:
                        pass
                    elif re.search(tok, k):
                        if len(dict_1[k]) == 0:
                            dict_1[k] = [[score, pref_name, cui, sem_type]]
                        else:
                            dict_1[k].append([score, pref_name, cui, sem_type])
                    else:
                         pass


    for k,v in dict_1.items():
        if v:
            if len(v) > 1:
                score_lis = [item[0] for item in v]
                max_score = max(score_lis)
                ind_max = score_lis.index(max_score)
                dict_1[k] = [v[ind_max]]
        else:
            dict_1[k] = "NA"


    sem_val = list(dict_1.values())
    sem_values = []
    sem_type_values = []
    lab_values = []
    ind_lis = []
    ind = 0
    for sem in sem_val:
        if sem == 'NA':
            sem_values.append('NA')
            sem_type_values.append('NA')
            lab_values.append('NA')
        else:
            sem_values.append(sem[0][1])
            sem_type_values.append(sem[0][3])
            lab_values.append([1.0, 0.0])
            ind_lis.append(ind)
        ind = ind + 1


    if key in sem_dict_new:
        sem_dict_new[key].append((sent_1[0], sem_values, sem_type_values, lab_values, [1.0, 0.0]))
    else:
        sem_dict_new[key] = [(sent_1[0], sem_values, sem_type_values, lab_values, [1.0, 0.0])]



    concepts_2, error_2 = mm.extract_concepts(sent_2)
    for concept in concepts_2:
        score = concept[2]
        pref_name = concept[3]
        cui = concept[4]
        sem_typ = concept[5]
        semt_lis = sem_typ.split(',')
        sem_type = semt_lis[0]
        sem_type = sem_type.replace('[', '')
        sem_type = sem_type.replace(']', '')
        sem_type = sem_type.strip()
        trigger = concept[6]
        t_lis = trigger.split('-')
        if len(t_lis) == 6:
            tok_l = t_lis[3]
            tok_li = tok_l.split()
            tok_li = [tok.replace('"', '') for tok in tok_li]
            tok_li = [tok.replace('(', '') for tok in tok_li]
            tok_li = [tok.replace(')', '') for tok in tok_li]
            tok_li = [tok.replace('[', '') for tok in tok_li]
            tok_li = [tok.replace(']', '') for tok in tok_li]
            tok_li = [tok.strip() for tok in tok_li]
            for tok in tok_li:
                for k,v in dict_2.items():
                    if tok == k:
                        if len(dict_2[tok]) == 0:
                            dict_2[tok] = [[score, pref_name, cui, sem_type]]
                        else:
                            dict_2[tok].append([score, pref_name, cui, sem_type])
                    elif tok == '(' or tok == ')':
                        pass
                    elif re.search(tok, k) is None:
                        pass
                    elif re.search(tok, k):
                        if len(dict_2[k]) == 0:
                            dict_2[k] = [[score, pref_name, cui, sem_type]]
                        else:
                            dict_2[k].append([score, pref_name, cui, sem_type])
                    else:
                        pass


    for k,v in dict_2.items():
        if v:
            if len(v) > 1:
                score_lis = [item[0] for item in v]
                max_score = max(score_lis)
                ind_max = score_lis.index(max_score)
                dict_2[k] = [v[ind_max]]
        else:
            dict_2[k] = "NA"


    sem_val = list(dict_2.values())
    sem_values = []
    sem_type_values = []
    lab_values = []
    ind_lis = []
    ind = 0
    for sem in sem_val:
        if sem == 'NA':
            sem_values.append('NA')
            sem_type_values.append('NA')
            lab_values.append('NA')
        else:
            sem_values.append(sem[0][1])
            sem_type_values.append(sem[0][3])
            lab_values.append([1.0, 0.0])
            ind_lis.append(ind)
        ind = ind + 1


    if key in sem_dict_new:
        sem_dict_new[key].append((sent_2[0], sem_values, sem_type_values, lab_values, [1.0, 0.0]))
    else:
        sem_dict_new[key] = [(sent_2[0], sem_values, sem_type_values, lab_values, [1.0, 0.0])]
    count = count + 1


with open('data_dict.pickle', 'wb') as handle:
     pickle.dump(sem_dict_new, handle)

