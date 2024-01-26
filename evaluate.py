# /usr/bin/env python
# coding=utf-8
"""Evaluate the model"""
import json
import logging
import random
import argparse

from tqdm import tqdm
import os

import torch
import numpy as np
import pandas as pd

from metrics import tag_mapping_major, tag_mapping_minor, tag_mapping_single,get_nary,get_event,get_mention
from utils import Label2IdxSub, Label2IdxObj, Label2IdxRel
import utils
from dataloader import CustomDataLoader

# load args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2020, help="random seed for initialization")
parser.add_argument('--ex_index', type=str, default=1)
parser.add_argument('--corpus_type', type=str, default="military", help="military")
parser.add_argument('--device_id', type=int, default=0, help="GPU index")
parser.add_argument('--restore_file', default='best', help="name of the file containing weights to reload")
parser.add_argument('--mode', type=str, default="train", help="train,test")
parser.add_argument('--major_threshold', type=float, default=0.1, help="threshold of major relation judgement")
parser.add_argument('--minor_threshold', type=float, default=0.1, help="threshold of minor relation judgement")


def get_metrics(correct_num, predict_num, gold_num):
    p = correct_num / predict_num if predict_num > 0 else 0
    r = correct_num / gold_num if gold_num > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {
        'correct_num': correct_num,
        'predict_num': predict_num,
        'gold_num': gold_num,
        'precision': p,
        'recall': r,
        'f1': f1
    }

def concat(token_list):
    result = ''
    for idx, t in enumerate(token_list):
        if idx == 0:
            result = t
        elif t.startswith('##'):
            result += t.lstrip('##')
        else:
            result += ' ' + t
    return result

def major2str(triples, tokens,off_mapping,text):
    output = []
    mention = []
    for triple in triples:
        rel = triple[-2]
        sub_tokens = tokens[triple[0][1]:triple[0][2]]
        for i in range(len(sub_tokens)):
            if sub_tokens[i]=='[UNK]':
                head,tail = off_mapping[triple[0][1]+i+1]
                sub_tokens[i] = text[head:tail]
        obj_tokens = tokens[triple[1][1]:triple[1][2]]
        for i in range(len(obj_tokens)):
            if obj_tokens[i]=='[UNK]':
                head,tail = off_mapping[triple[1][1]+i+1]
                obj_tokens[i] = text[head:tail]
        rel_tokens = tokens[triple[-1][1]:triple[-1][2]]
        for i in range(len(rel_tokens)):
            if rel_tokens[i]=='[UNK]':
                head,tail = off_mapping[triple[-1][1]+i+1]
                rel_tokens[i] = text[head:tail]
        sub = concat(sub_tokens)
        obj = concat(obj_tokens)
        rel_men = concat(rel_tokens)
        if sub and obj:
            output.append((sub, obj, rel))
        if rel_men:
            mention.append((rel_men, rel))
    return output, mention

def span2str(triples, tokens, off_mapping, text):
    output = []
    for triple in triples:
        rel = triple[-1]
        if triple[0][1]>len(tokens):
            continue
        sub_tokens = tokens[triple[0][1]:triple[0][2]]
        for i in range(len(sub_tokens)):
            if sub_tokens[i]=='[UNK]':
                head,tail = off_mapping[triple[0][1]+i+1]
                sub_tokens[i] = text[head:tail]
        if triple[1][1]>len(tokens):
            continue
        obj_tokens = tokens[triple[1][1]:triple[1][2]]
        for i in range(len(obj_tokens)):
            if obj_tokens[i] == '[UNK]':
                head, tail = off_mapping[triple[1][1] + i+1]
                obj_tokens[i] = text[head:tail]
        sub = concat(sub_tokens)
        obj = concat(obj_tokens)
        if sub == '[UNK]':
            head,tail = off_mapping[triple[0][1]]
            sub = text[head:tail]
        if obj == '[UNK]':
            head,tail = off_mapping[triple[1][1]]
            obj = text[head:tail]
        if obj:
            output.append((sub, obj, rel))
    return output

def single2str(span,tokens,off_mapping,text):
    output = []
    for en in span:
        rel = en[1]
        if en[0][1]>len(tokens):
            continue
        mention = tokens[en[0][1]:en[0][2]]
        for i in range(len(mention)):
            if mention[i] == '[UNK]':
                head, tail = off_mapping[en[0][1] + i+1]
                mention[i] = text[head:tail]
        if mention:
            mention= concat(mention)
            output.append((mention, rel))
    return output




def evaluate(model, data_iterator, params, ex_params, mark='Val'):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()

    predictions_major = []
    ground_truths_major = []
    correct_num_major, predict_num_major, gold_num_major = 0, 0, 0
    predictions_minor = []
    ground_truths_minor = []
    predictions_nary = []
    ground_truths_nary = []
    correct_num_minor, predict_num_minor, gold_num_minor = 0, 0, 0
    correct_num_nary, predict_num_nary, gold_num_nary = 0, 0, 0

    for batch in tqdm(data_iterator, unit='Batch', ascii=True):
        # to device
        batch = tuple(t.to(params.device) if isinstance(t, torch.Tensor) else t for t in batch)
        input_ids, attention_mask, major_triples, minor_triples, input_tokens,off_mapping,text= batch
        bs, seq_len = input_ids.size()
        # inference
        with torch.no_grad():
            pred_major_tag, pred_minor_tag, xi, xi_minor, pred_major,\
            pred_minor, head_minor = model(input_ids, attention_mask=attention_mask,
                                                         ex_params=ex_params)

        xi_index = np.cumsum(xi).tolist()
        xi_minor_index = np.cumsum(xi_minor).tolist()
        xi_index.insert(0, 0)
        xi_minor_index.insert(0, 0)
        # (bs+1,)

        for idx in range(bs):
            pre_major_triples = tag_mapping_major(predict_tags=pred_major_tag[xi_index[idx]:xi_index[idx + 1]],
                                                          pre_rels=pred_major[xi_index[idx]:xi_index[idx + 1]],
                                                          seq_len=attention_mask[idx].sum(),
                                                          label2idx_sub=Label2IdxSub,
                                                          label2idx_obj=Label2IdxObj)
            pre_major_mention = tag_mapping_single(predict_tags=pred_major_tag[xi_index[idx]:xi_index[idx + 1]],
                                                           pre_rels=pred_major[xi_index[idx]:xi_index[idx + 1]],
                                                           seq_len=attention_mask[idx].sum(),
                                                           index=2,
                                                           label2idx=Label2IdxRel)
            pre_minor_triples = []
            if head_minor:
                pre_minor_triples = tag_mapping_minor(head=head_minor[idx],
                                                      predict_tags=pred_minor_tag[
                                                                   xi_minor_index[idx]:xi_minor_index[idx + 1]],
                                                      seq_len=attention_mask[idx].sum(),
                                                      pre_rels=pred_minor[
                                                               xi_minor_index[idx]:xi_minor_index[idx + 1]],
                                                      label2idx_obj=Label2IdxObj)

            gold_major_triples, rel_mention = major2str(major_triples[idx], input_tokens[idx],off_mapping[idx],text[idx])

            gold_minor_triples = span2str(minor_triples[idx], input_tokens[idx],off_mapping[idx],text[idx])
            pre_major_triples = span2str(pre_major_triples, input_tokens[idx],off_mapping[idx],text[idx])
            pre_mention = single2str(pre_major_mention, input_tokens[idx],off_mapping[idx],text[idx])
            pre_minor_triples = span2str(pre_minor_triples, input_tokens[idx],off_mapping[idx],text[idx])
            gold_major_triples = list(set(gold_major_triples))
            pre_major_triples = list(set(pre_major_triples))
            gold_minor_triples = list(set(gold_minor_triples))
            pre_minor_triples = list(set(pre_minor_triples))
            pre_nary = get_nary(pre_major_triples, pre_mention, pre_minor_triples)
            gold_nary = get_nary(gold_major_triples, rel_mention, gold_minor_triples)
            # pre_nary = get_event(pre_major_triples, pre_mention, pre_minor_triples)
            # gold_nary = get_event(gold_major_triples, rel_mention, gold_minor_triples)
            ground_truths_major.append(gold_major_triples)
            predictions_major.append(pre_major_triples)
            ground_truths_minor.append(gold_minor_triples)
            predictions_minor.append(pre_minor_triples)
            # counter
            gold_nary_set = []
            pred_nary_set = []
            for pre in pre_nary:
                pred_nary_set.append([pre[0],pre[1]])
            for gold in gold_nary:
                gold_nary_set.append([gold[0],gold[1]])
                for pre in pre_nary:
                    if gold[0] == pre[0]:
                        if gold[2]:
                            correct_num_nary += len(set(gold[2]) & set(pre[2])) / len(set(gold[2]))


            predictions_nary.append(pred_nary_set)
            ground_truths_nary.append(gold_nary_set)

            correct_num_major += len(set(pre_major_triples) & set(gold_major_triples))
            predict_num_major += len(pre_major_triples)
            gold_num_major += len(gold_major_triples)
            correct_num_minor += len(set(pre_minor_triples) & set(gold_minor_triples))
            predict_num_minor += len(pre_minor_triples)
            gold_num_minor += len(gold_minor_triples)
            predict_num_nary += len(pre_major_triples)
            gold_num_nary += len(gold_major_triples)

    major_metrics = get_metrics(correct_num_major, predict_num_major, gold_num_major)
    minor_metrics = get_metrics(correct_num_minor, predict_num_minor, gold_num_minor)
    nary_metrics = get_metrics(correct_num_nary, predict_num_nary, gold_num_nary)
    # logging loss, f1 and report
    major_metrics_str = "; ".join("Main_Relation_{}: {:05.3f}".format(k, v) for k, v in major_metrics.items())
    minor_metrics_str = "; ".join("Auxiliary_Relation_{}: {:05.3f}".format(k, v) for k, v in minor_metrics.items())
    nary_metrics_str = "; ".join("Nary_Relation_{}: {:05.3f}".format(k, v) for k, v in nary_metrics.items())
    logging.info("Main Relation - {} metrics:\n".format(mark) + major_metrics_str)
    logging.info("Auxiliary Relation - {} metrics:\n".format(mark) + minor_metrics_str)
    logging.info("Nary Relation - {} metrics:\n".format(mark) + nary_metrics_str)
    return major_metrics, minor_metrics, nary_metrics, predictions_major, predictions_minor, \
           predictions_nary, ground_truths_major, ground_truths_minor, ground_truths_nary


if __name__ == '__main__':
    args = parser.parse_args()
    params = utils.Params(ex_index=args.ex_index, corpus_type=args.corpus_type)
    ex_params = {
        'major_threshold': args.major_threshold,
        'minor_threshold': args.minor_threshold,
    }

    print('current device:', torch.cuda.current_device())
    mode = args.mode
    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed

    # Set the logger
    utils.set_logger()

    # get dataloader
    dataloader = CustomDataLoader(params)

    # Define the model
    logging.info('Loading the model...')
    logging.info(f'Path: {os.path.join(params.model_dir, args.restore_file)}.pth.tar')
    # Reload weights from the saved file
    model, optimizer = utils.load_checkpoint(os.path.join(params.model_dir, args.restore_file + '.pth.tar'))
    model.to(params.device)
    logging.info('- done.')

    logging.info("Loading the dataset...")
    loader = dataloader.get_dataloader(data_sign=mode, ex_params=ex_params)
    logging.info('-done')

    logging.info("Starting prediction...")

    _, _1, _2, predictions_major, predictions_minor, predictions_nary, ground_truths_major, ground_truths_minor,\
    ground_truths_nary = evaluate(model, loader, params, ex_params, mark=mode)

    with open(params.data_dir / f'{mode}_narys.json', 'r', encoding='utf-8') as f_src:
        src = json.load(f_src)
        df = pd.DataFrame(
            {
                'text': [sample['text'] for sample in src],
                'pre_major': predictions_major,
                'truth_major': ground_truths_major,
                'pre_minor': predictions_minor,
                'truth_minor': ground_truths_minor,
                'pre_nary': predictions_nary,
                'truth_nary': ground_truths_nary,
            }
        )
        df.to_csv(params.ex_dir / f'{mode}_result.csv')

    logging.info('-done')
