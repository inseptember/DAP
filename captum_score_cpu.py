import json
import pickle
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoConfig, BertTokenizer
from model_flash_dep_v4 import RoFormerForSequenceCls

from captum.attr import (
    IntegratedGradients,
    TokenReferenceBase,
    visualization,
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer
)
from captum.attr._utils.input_layer_wrapper import ModelInputWrapper


prefix = '4'
days = '365'
max_length = 2500
# project_path = '/opt/storage2/workspace/hospital'
# njinfocenter_path = '/opt/storage/workspace/njinfocenter'
project_path = '/data/fengwei/workspace/hospital'
njinfocenter_path = '/data/fengwei/workspace/njinfocenter'

base_dir = f'{project_path}/train/category/njcenter_v4_wdrg'
exam_name_path = f'{project_path}/train/category/exam_names.json'
# df_dm_date = pd.read_pickle(f'{njinfocenter_path}/preprocess_2021/3_dep_events/dataset/dm_date.pkl')
# df_all = pd.read_pickle(f'{njinfocenter_path}/desc/dataset/in_hospital_raw_drg_all.pkl')

batch_size = 50


# file_path = f'{base_dir}/{prefix}_train_{days}.json'
model_path = f'{base_dir}/{prefix}_output_flash_dep_{days}'
person_cols = ['age', 'gender_name', 'marriage_status_name', 'temperature', 'pulse', 'systolicbloodpressure',
               'diastolicbloodpressure', 'dm_duration']
exam_cols = ['ADA', 'AFP', 'ALB', 'ALP', 'ALT', 'APTT',
             'AST', 'BA%', 'CA199', 'CEA', 'CK', 'CYFRA21-1', 'Ca', 'Cl',
             'Cr', 'DBIL', 'DD2', 'EO%', 'FIB',
             'FT3', 'FT4', 'GGT', 'GLB', 'GLU', 'HBDH', 'HBsAb', 'HCT', 'HDL-C', 'HGB',
             'HbA1c', 'K', 'LDH', 'LDL-C', 'LPa',
             'LY%', 'MCH', 'MCHC', 'MCV', 'MO%',
             'MPV', 'Mg', 'NE%', 'NSE', 'Na', 'PCT', 'PDW',
             'PLT', 'PT', 'PT-INR', 'Phos', 'RBC', 'RBC_U',
             'RBP', 'RDW-CV', 'SG', 'TBIL', 'TC', 'TG', 'TP',
             'TSH', 'TT', 'UA', 'Urea', 'WBC', 'WBC_U']
exam_cols = person_cols + exam_cols
padding = False
skip_token_ids = [101, 102, 103, 104, 105]

# df_test_all = pd.read_json(file_path, lines=True, orient='records')
# df_test_all['visit_sn_source'] = df_test_all['visit_sn_source'].astype(str)
# df_test_all = df_test_all[df_test_all['icd_labels2'].apply(lambda x: len([i for i in ['焦虑', '抑郁', '双相'] if i in x]) == 0)]
#
# print('************************Total %d records.************************************' % len(df_test_all))
# #%%
# def marriage_status(x):
#     for i in ['丧偶']:
#         if i in x:
#             return '丧偶'
#     for i in ['离异', '离婚']:
#         if i in x:
#             return '离婚'
#     for i in ['适龄婚配', '已婚', '适龄婚育', '自由结婚', '适龄结婚']:
#         if i in x:
#             return '已婚'
#     for i in ['未婚']:
#         if i in x:
#             return '未婚'
#     return None
# def add_plus_cols(dataframe):
#     dataframe = dataframe.rename(columns={
#         'drug_name': 'drug_names',
#         'temp': 'temperature',
#         'pr': 'pulse',
#         'sbp': 'systolicbloodpressure',
#         'dbp': 'diastolicbloodpressure'
#     })
#     dataframe['gender_name'] = dataframe['sex'].astype(str).replace({
#         '1': '男', '2': '女'
#     })
#     dataframe['marriage_status_name'] = dataframe['obs_his'].apply(marriage_status)
#     return dataframe
# df_all = add_plus_cols(df_all)
#
# df_dm_date['visit_sn_source'] = df_dm_date['source'].astype(int).astype(str) + df_dm_date['visit_sn'].astype(str)
# df_all = pd.merge(left=df_all, left_on='visit_sn_source',
#                   right=df_dm_date[['visit_sn_source', 'dm_duration']], right_on='visit_sn_source', how='left')
# df_test_all = pd.merge(left=df_test_all, left_on='visit_sn_source',
#                    right=df_all[['visit_sn_source'] + exam_cols].rename(
#                        columns={k:k+'_O' for k in exam_cols}
#                    ), right_on='visit_sn_source', how='left')
# del df_all
sub_file = int(sys.argv[1])
df_test_all = pd.read_pickle('sub_df_test_%d.pkl' % sub_file)

batch_size_indx = -1 if len(sys.argv) < 3 else int(sys.argv[2])

print('Read sub file %d, length is %d.' % (sub_file, len(df_test_all)))
print('Start from batch index %d.' % batch_size_indx)

config = AutoConfig.from_pretrained(model_path, num_labels=2)
tokenizer = BertTokenizer.from_pretrained(model_path)


baseline_diag_text = tokenizer(['<S> %s %s %s' % (
                tokenizer.cls_token,
                ('%s %s' % (tokenizer.sep_token, tokenizer.cls_token)).join(''.split(';')),
                tokenizer.sep_token
            )], padding=padding, max_length=max_length, truncation=True, return_tensors='pt')
baseline_drug_text = tokenizer(['<T> %s %s %s' % (
                tokenizer.cls_token,
                ('%s %s' % (tokenizer.sep_token, tokenizer.cls_token)).join(''.split(';')),
                tokenizer.sep_token
            )], padding=padding, max_length=max_length, truncation=True, return_tensors='pt')


def preprocess_function(examples):
    result = tokenizer(examples['text'], padding=padding, max_length=max_length, truncation=True)

    icd_labels = []
    for i in examples['icd_labels2']:
        icd_labels.append(
            '<S> %s %s %s' % (
                tokenizer.cls_token,
                ('%s %s' % (tokenizer.sep_token, tokenizer.cls_token)).join(i.split(';')),
                tokenizer.sep_token
            )
        )
    diags_input = tokenizer(icd_labels, padding=padding, max_length=max_length, truncation=True)
    result['diag_input_ids'] = diags_input['input_ids']
    result['diag_attention_mask'] = diags_input['attention_mask']

    drug_names = []
    for i in examples['drug_names']:
        i = '' if i is None else i
        drug_names.append(
            '<T> %s %s %s' % (
                tokenizer.cls_token,
                ('%s %s' % (tokenizer.sep_token, tokenizer.cls_token)).join(i.split(';')),
                tokenizer.sep_token
            )
        )
    drug_input = tokenizer(drug_names, padding=padding, max_length=max_length, truncation=True)
    result['drug_input_ids'] = drug_input['input_ids']
    result['drug_attention_mask'] = drug_input['attention_mask']

    nan_token = 103
    cls_token = 101
    # exam_data = list(zip(*[[j for j in examples[i]] for i in exam_cols]))
    exam_data = list(
        zip(*[[j for j in examples.get(i, [nan_token] * len(examples[exam_cols[0]]))] for i in exam_cols]))
    exam_data = [(cls_token,) + i for i in exam_data]
    exam_mask = [[0 if j == nan_token else 1 for j in i] for i in exam_data]

    result['exam_input_values'] = exam_data
    result['exam_value_mask'] = exam_mask

    result["labels"] = examples["y"]

    return result


def get_batch(features):
    features1 = [{
        'input_ids': i, 'attention_mask': j, 'labels': k
    } for i, j, k in zip(features['input_ids'], features['attention_mask'], features['labels'])]
    batch = tokenizer.pad(
        features1,
        padding=True,
        max_length=None,
        pad_to_multiple_of=8,
        return_tensors='pt',
    )

    features1 = [{
        'input_ids': i, 'attention_mask': j
    } for i, j in zip(features['diag_input_ids'], features['diag_attention_mask'])]
    batch2 = tokenizer.pad(
        features1,
        padding=True,
        max_length=max_length,
        pad_to_multiple_of=None,
        return_tensors='pt',
    )

    batch["diag_input_ids"] = batch2['input_ids']
    batch["diag_attention_mask"] = batch2['attention_mask']

    features1 = [{
        'input_ids': i, 'attention_mask': j
    } for i, j in zip(features['drug_input_ids'], features['drug_attention_mask'])]
    batch2 = tokenizer.pad(
        features1,
        padding=True,
        max_length=max_length,
        pad_to_multiple_of=None,
        return_tensors='pt',
    )

    batch["drug_input_ids"] = batch2['input_ids']
    batch["drug_attention_mask"] = batch2['attention_mask']

    batch["exam_input_values"] = torch.tensor([list(i) for i in features['exam_input_values']], dtype=torch.float)
    batch["exam_value_mask"] = torch.tensor([list(i) for i in features['exam_value_mask']], dtype=torch.float)

    # for k in batch.keys():
    #     batch[k] = batch[k].squeeze(0)

    return batch


def get_background(input_data):
    features = preprocess_function(input_data)
    batch = get_batch(features)
    return batch


def get_ref(tensors, ref_id=0):
    r = tensors.clone()
    mask = torch.zeros_like(tensors, dtype=torch.bool)
    for i in skip_token_ids:
        mask[tensors == i] = True
    r[~mask] = ref_id
    return r


def get_baselines(batch):
    data = {}
    for k in batch.keys():
        if '_input_' in k:
            data[k] = torch.zeros(batch[k].shape)
            data[k][:, 0] = batch[k][:, 0]
        else:
            data[k] = batch[k]
    return data


def to_tuple(batch):
    input_ids = batch['input_ids'].int()
    attention_mask = batch['attention_mask'].int()
    exam_input_values = batch['exam_input_values'].int()
    exam_value_mask = batch['exam_value_mask'].int()
    diag_input_ids = batch['diag_input_ids'].int()
    diag_attention_mask = batch['diag_attention_mask'].int()
    drug_input_ids = batch['drug_input_ids'].int()
    drug_attention_mask = batch['drug_attention_mask'].int()
    return (input_ids, attention_mask, exam_input_values, exam_value_mask, diag_input_ids,
            diag_attention_mask, drug_input_ids, drug_attention_mask)


def to_input_data(batch, m):
    input_ids, attention_mask, exam_input_values, exam_value_mask, diag_input_ids, diag_attention_mask, \
    drug_input_ids, drug_attention_mask = batch
    text_hiddens = m.model.get_text_hiddens(
        input_ids, attention_mask
    )
    diag_hiddens = m.model.get_text_hiddens(
        diag_input_ids, diag_attention_mask
    )
    drug_hiddens = m.model.get_text_hiddens(
        drug_input_ids, drug_attention_mask
    )
    exam_hiddens = m.model.exam_encoder.embeddings(exam_input_values.int())
    return (text_hiddens, exam_hiddens,
            diag_hiddens, drug_hiddens, input_ids, attention_mask, exam_input_values, exam_value_mask, diag_input_ids,
            diag_attention_mask, drug_input_ids, drug_attention_mask)


def get_baseline_hiddens(batch, m):
    text_hiddens, exam_hiddens, \
    diag_hiddens, drug_hiddens, \
    input_ids, attention_mask, exam_input_values, exam_value_mask, diag_input_ids, \
    diag_attention_mask, drug_input_ids, drug_attention_mask = batch

    text_hiddens_ref = m.model.get_text_hiddens(
        get_ref(input_ids).int(), attention_mask
    )
    exam_hiddens_ref = m.model.exam_encoder.embeddings(
        get_ref(exam_input_values.int(), 103).int()
    )
    diag_hiddens_ref = m.model.get_text_hiddens(
        get_ref(diag_input_ids).int(), diag_attention_mask.int()
    )
    drug_hiddens_ref = m.model.get_text_hiddens(
        get_ref(drug_input_ids).int(), drug_attention_mask.int()
    )
    return (text_hiddens_ref, exam_hiddens_ref,
            diag_hiddens_ref, drug_hiddens_ref, input_ids, attention_mask, exam_input_values, exam_value_mask,
            diag_input_ids,
            diag_attention_mask, drug_input_ids, drug_attention_mask)


def get_one_batch(batch, i):
    text_hiddens, exam_hiddens, \
    diag_hiddens, drug_hiddens, input_ids, attention_mask, exam_input_values, exam_value_mask, diag_input_ids, \
    diag_attention_mask, drug_input_ids, drug_attention_mask = batch
    text_hiddens = text_hiddens[i].unsqueeze(0)
    exam_hiddens = exam_hiddens[i].unsqueeze(0)
    diag_hiddens = diag_hiddens[i].unsqueeze(0)
    drug_hiddens = drug_hiddens[i].unsqueeze(0)
    attention_mask = attention_mask[i].unsqueeze(0)
    exam_value_mask = exam_value_mask[i].unsqueeze(0)
    diag_attention_mask = diag_attention_mask[i].unsqueeze(0)
    drug_attention_mask = drug_attention_mask[i].unsqueeze(0)
    return (text_hiddens, exam_hiddens,
            diag_hiddens, drug_hiddens, input_ids, attention_mask, exam_input_values, exam_value_mask, diag_input_ids,
            diag_attention_mask, drug_input_ids, drug_attention_mask)


# def batch_to_gpu(*params):
#     return tuple([p.cuda(device=device_id) for p in params])
#
#
# def batch_dict_to_gpu(params):
#     for k in params.keys():
#         params[k] = params[k].cuda(device=device_id)
#     return params


def write_pkl(file_name, obj):
    with open(f'{sub_file}_{file_name}.pkl', 'wb') as f:
        pickle.dump(obj, f)


df_test_list = np.array_split(df_test_all, batch_size)
for t_indx, df_test in enumerate(tqdm.tqdm(df_test_list)):
    if len(df_test) == 0:
        continue
    if batch_size_indx >= 0 and batch_size_indx > t_indx:
        print('pass %d' % t_indx)
        continue
    df_test_sub = df_test.to_dict(orient='list')
    visit_sn_sources = df_test['visit_sn_source'].values.tolist()
    diag_word_scores = []
    exam_scores = []
    drug_scores = []
    record_attrs = []
    model = RoFormerForSequenceCls(config, eos_token_id=tokenizer.convert_tokens_to_ids('[SEP]'))
    model.load_state_dict(torch.load(model_path + '/pytorch_model.bin', map_location='cpu'), strict=False)
    model.model.set_input_encoder()
    batch_data = get_background(df_test_sub)

    # batch_data = batch_dict_to_gpu(batch_data)


    target = batch_data['labels'].int()
    batch_data = to_tuple(batch_data)
    input_data = to_input_data(batch_data, model)

    baseline_data = get_baseline_hiddens(input_data, model)

    # model = model.cuda(device=device_id)

    # scaler = GradScaler()

    attr = IntegratedGradients(model)
    model = model.eval()


    def get_icd_labels(examples):
        icd_labels = []
        for i in examples['icd_labels2']:
            icd_labels.append(
                '<S> %s %s %s' % (
                    tokenizer.cls_token,
                    ('%s %s' % (tokenizer.sep_token, tokenizer.cls_token)).join(i.split(';')),
                    tokenizer.sep_token
                )
            )
        return icd_labels


    icd_labels = get_icd_labels(df_test)


    def get_drug_names(examples):
        drug_names = []
        for i in examples['drug_names']:
            i = '' if i is None else i
            drug_names.append(
                '<T> %s %s %s' % (
                    tokenizer.cls_token,
                    ('%s %s' % (tokenizer.sep_token, tokenizer.cls_token)).join(i.split(';')),
                    tokenizer.sep_token
                )
            )
        return drug_names


    drug_names = get_drug_names(df_test)

    for k in tqdm.tqdm(range(input_data[0].shape[0])):
        visit_sn_source = visit_sn_sources[k]

        model.zero_grad()
        input_data_sub = get_one_batch(input_data, k)
        baseline_data_sub = get_one_batch(baseline_data, k)

        # input_data_sub = batch_to_gpu(*input_data_sub)
        # baseline_data_sub = batch_to_gpu(*baseline_data_sub)

        target_sub = target[k].long().item()

        with torch.no_grad():
            pred, pred_idx = F.softmax(model(*input_data_sub), dim=-1).cpu().max(dim=1)
            attributions, delta = attr.attribute(
                inputs=input_data_sub[:4], baselines=baseline_data_sub[:4],
                additional_forward_args=input_data_sub[4:],
                target=target_sub, n_steps=20, return_convergence_delta=True
            )

        record_attrs.append([visit_sn_source, attributions[0].sum(), attributions[1].sum(), attributions[2].sum()])

        word_scores = []
        last_word = []
        last_word_score = 0
        for t, a in zip(
                [tokenizer.convert_ids_to_tokens(t).replace('<S>', '[S]') for t in
                 tokenizer(icd_labels[k], padding=padding, max_length=max_length, truncation=True)['input_ids']],
                attributions[2].sum(dim=2).squeeze(0)
        ):
            if t == '[CLS]':
                last_word = []
                last_word_score = 0
            elif t == '[SEP]' and len(last_word) > 0:
                word_scores.append((visit_sn_source, ''.join(last_word), last_word_score.item()))
                last_word = []
                last_word_score = 0
            elif t != '[SEP]':
                last_word_score = last_word_score + a
                last_word.append(t)
        diag_word_scores += word_scores

        word_scores = []
        for t, v, a in zip(
                ['[CLS]'] + exam_cols,
                ['[CLS]'] + [df_test['%s_O' % t].values[k] for t in exam_cols],
                attributions[1].sum(dim=2).squeeze(0)
        ):
            if t == '[CLS]' or str(v) == 'nan' or v is None or len(str(v)) == 0:
                continue
            word_scores.append((visit_sn_source, t, a.item(), v))
        exam_scores += word_scores

        word_scores = []
        last_word = []
        last_word_score = 0
        for t, a in zip(
                [tokenizer.convert_ids_to_tokens(t).replace('<T>', '[T]') for t in
                 tokenizer(drug_names[k], padding=padding, max_length=max_length, truncation=True)['input_ids']],
                attributions[3].sum(dim=2).squeeze(0)
        ):
            if t == '[CLS]':
                last_word = []
                last_word_score = 0
            elif t == '[SEP]' and len(last_word) > 0:
                word_scores.append((visit_sn_source, ''.join(last_word), last_word_score.item()))
                last_word = []
                last_word_score = 0
            elif t != '[SEP]':
                last_word_score = last_word_score + a
                last_word.append(t)
        drug_scores += word_scores



    write_pkl('diag_word_scores_%d' % t_indx, diag_word_scores)
    write_pkl('exam_scores_%d' % t_indx, exam_scores)
    write_pkl('drug_scores_%d' % t_indx, drug_scores)
    write_pkl('record_scores_%d' % t_indx, record_attrs)
