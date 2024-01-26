# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import random
from multiprocessing import Pool
import functools
import numpy as np
from collections import defaultdict
from itertools import chain

from utils import Label2IdxSub, Label2IdxObj, Label2IdxRel


class InputExample(object):
    """a single set of samples of data
    """

    def __init__(self, text, major_pair_list,minor_pair_list,major_re_list,minor_re_list, major2ens,minor2ens
                 ,major_rel_mention):
        self.text = text
        self.major_pair_list = major_pair_list
        self.minor_pair_list = minor_pair_list
        self.major_re_list = major_re_list
        self.minor_re_list = minor_re_list
        self.major2ens = major2ens
        self.minor2ens = minor2ens
        self.major_rel_mention = major_rel_mention


class InputFeatures(object):
    """
    Desc:
        a single set of features of data
    """

    def __init__(self,
                 input_tokens,
                 input_ids,
                 attention_mask,
                 seq_tags=None,
                 relation=None,
                 major_triples=None,
                 minor_triples=None,
                 major_rel_tag=None,
                 minor_rel_tag=None,
                 off_mapping=None,
                 text=None
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.seq_tags = seq_tags
        self.relation = relation
        self.major_triples = major_triples
        self.minor_triples = minor_triples
        self.major_rel_tag = major_rel_tag
        self.minor_rel_tag = minor_rel_tag
        self.off_mapping=off_mapping
        self.text = text




def read_examples(data_dir, data_sign, major2idx,minor2idx):
    """load data to InputExamples
    """
    examples = []

    # read src data
    with open(data_dir / f'{data_sign}_narys.json', "r", encoding='utf-8') as f:
        data = json.load(f)
        for sample in data:
            text = sample['text']
            major2ens = defaultdict(list)
            minor2ens = defaultdict(list)
            major_pair_list = []
            minor_pair_list = []
            major_re_list = []
            minor_re_list = []
            major_rel_mention={}
            for nary in sample['nary_list']:
                temp_rel_mention=nary[1]['mention']
                temp_major_pair = [nary[0]['subject'],nary[2]['object']]
                temp_major_rel = major2idx[nary[1]['relation']]
                major_rel_mention[temp_major_rel]=temp_rel_mention
                major_pair_list.append(temp_major_pair)
                major_re_list.append(temp_major_rel)
                major2ens[temp_major_rel].append((temp_major_pair[0],temp_major_pair[1]))
                for element in nary:
                    if len(element['auxiliary'])>0:
                        inf = text[element['mention'][0]:element['mention'][1] + 1]
                        for auxiliary in element['auxiliary']:
                            minor_pair_list.append([inf, auxiliary[1]])
                            minor_re_list.append(minor2idx[auxiliary[0]])
                            minor2ens[minor2idx[auxiliary[0]]].append((inf,auxiliary[1]))
            example = InputExample(text=text,
                                   major_pair_list=major_pair_list,
                                   minor_pair_list=minor_pair_list,
                                   major_re_list=major_re_list,
                                   minor_re_list=minor_re_list,
                                   major2ens=major2ens,
                                   minor2ens=minor2ens,
                                   major_rel_mention=major_rel_mention)
            examples.append(example)
    print("InputExamples:", len(examples))
    return examples


def find_head_idx(source, target,off_mapping,text,en):
    target_len = len(target)
    if '[UNK]' in target :
        head = text.find(en)
        for i in range(1,len(off_mapping)-1):
            if off_mapping[i][0]==head:
                return i-1
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1

def _get_head(en,tokenizer,text_tokens,off_mapping,text):
    mention = tokenizer.tokenize(en)
    men_head = find_head_idx(source=text_tokens, target=mention,off_mapping=off_mapping,text=text,en=en)

    return men_head,mention

def _get_so_head(en_pair, tokenizer, text_tokens,off_mapping,text):
    sub = tokenizer.tokenize(en_pair[0])
    obj = tokenizer.tokenize(en_pair[1])
    sub_head = find_head_idx(source=text_tokens, target=sub,off_mapping=off_mapping,text=text,en=en_pair[0])
    if en_pair[0] == en_pair[1]:
        obj_head = find_head_idx(source=text_tokens[sub_head+ len(sub):], target=obj,
                                 off_mapping=off_mapping[sub_head+ len(sub):],
                                 text=text[sub_head+ len(sub):],en=en_pair[1])
        if obj_head != -1:
            obj_head += sub_head + len(sub)
        else:
            obj_head = sub_head
    else:
        obj_head = find_head_idx(source=text_tokens, target=obj,off_mapping=off_mapping,text=text,en=en_pair[1])
    return sub_head, obj_head, sub, obj

def convert(example, max_text_len, tokenizer, major2idx,minor2idx, data_sign):
    """convert function
    """
    text_tokens = tokenizer.tokenize(example.text)
    off_mapping = tokenizer(example.text,return_offsets_mapping=True)['offset_mapping']

    # cut off
    if len(text_tokens) > max_text_len:
        text_tokens = text_tokens[:max_text_len]

    # token to id
    input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
    attention_mask = [1] * len(input_ids)
    # zero-padding up to the sequence length
    if len(input_ids) < max_text_len:
        pad_len = max_text_len - len(input_ids)
        # token_pad_id=0
        input_ids += [0] * pad_len
        attention_mask += [0] * pad_len

    # train data
    if data_sign == 'train':
        # construct tags of correspondence and relation
        major_rel = len(major2idx) * [0]
        minor_rel = len(minor2idx) * [0]
        for major_re in example.major_re_list:
            major_rel[major_re] = 1
        for minor_re in example.minor_re_list:
            minor_rel[minor_re] = 1
        major_t = []
        major_tr = []
        major_r = []
        # major samples
        for major_re, major_en in example.major2ens.items():
            tags_sub = max_text_len * [Label2IdxSub['O']]
            tags_obj = max_text_len * [Label2IdxObj['O']]
            tags_men = max_text_len * [Label2IdxRel['O']]
            rel_index = example.major_rel_mention[major_re]
            major_tri = []
            for en in major_en:
                sub_head, obj_head, sub, obj = _get_so_head(en, tokenizer, text_tokens,off_mapping,example.text)
                rel_head,rel = _get_head(example.text[rel_index[0]:rel_index[1]+1],tokenizer, text_tokens,off_mapping,example.text)
                if sub_head != -1 and obj_head != -1:
                    h_chunk = ('H', sub_head, sub_head + len(sub), en[0],len(text_tokens))
                    t_chunk = ('T', obj_head, obj_head + len(obj), en[1],len(text_tokens))
                    r_chunk = ('R', rel_head, rel_head + len(rel), example.text[rel_index[0]:rel_index[1]+1],len(text_tokens))
                    major_tri.append((h_chunk, t_chunk, major_re, r_chunk))
                    if sub_head + len(sub) <= max_text_len:
                        tags_sub[sub_head] = Label2IdxSub['B-H']
                        tags_sub[sub_head + 1:sub_head + len(sub)] = (len(sub) - 1) * [Label2IdxSub['I-H']]
                    if obj_head + len(obj) <= max_text_len:
                        tags_obj[obj_head] = Label2IdxObj['B-T']
                        tags_obj[obj_head + 1:obj_head + len(obj)] = (len(obj) - 1) * [Label2IdxObj['I-T']]
                    if rel_head + len(rel) <= max_text_len:
                        tags_men[rel_head] = Label2IdxRel['B-R']
                        tags_men[rel_head + 1:rel_head + len(rel)] = (len(rel) - 1) * [Label2IdxRel['I-R']]
            major_tags = [tags_sub, tags_obj, tags_men]
            major_t.append(major_tags)
            major_tr.append(major_tri)
            major_r.append(major_re)


        minor_t = []
        minor_tr = []
        minor_r = []
        for minor_re, minor_en in example.minor2ens.items():
            tags_obj = max_text_len * [Label2IdxObj['O']]
            pad = max_text_len * [Label2IdxObj['O']]
            minor_trip = []
            for en in minor_en:
                sub_head, obj_head, sub, obj = _get_so_head(en, tokenizer, text_tokens, off_mapping, example.text)
                if sub_head != -1 and obj_head != -1:
                    h_chunk = ('H', sub_head, sub_head + len(sub),en[0],len(text_tokens))
                    t_chunk = ('T', obj_head, obj_head + len(obj),en[1],len(text_tokens))
                    minor_trip.append((h_chunk, t_chunk, minor_re))
                    if obj_head + len(obj) <= max_text_len:
                        tags_obj[obj_head] = Label2IdxObj['B-T']
                        tags_obj[obj_head + 1:obj_head + len(obj)] = (len(obj) - 1) * [Label2IdxObj['I-T']]
            minor_tags = [pad, tags_obj, pad]
            minor_t.append(minor_tags)
            minor_tr.append(minor_trip)
            minor_r.append(minor_re)
        sub_feats = [InputFeatures(
            input_tokens=text_tokens,
            input_ids=input_ids,
            attention_mask=attention_mask,
            seq_tags=(major_t,minor_t),
            major_triples=major_tr,
            minor_triples=minor_tr,
            relation=(major_r,minor_r),
            major_rel_tag=major_rel,
            minor_rel_tag=minor_rel
        )]

    else:
        major_triples = []
        minor_triples = []

        for major_rel, major_en in zip(example.major_re_list, example.major_pair_list):
            # get sub and obj head
            rel_index = example.major_rel_mention[major_rel]
            sub_head, obj_head, sub, obj = _get_so_head(major_en, tokenizer, text_tokens,off_mapping,example.text)
            rel_head, rel = _get_head(example.text[rel_index[0]:rel_index[1] + 1], tokenizer, text_tokens,off_mapping,example.text)
            if sub_head != -1 and obj_head != -1:
                h_chunk = ('H', sub_head, sub_head + len(sub),major_en[0],len(text_tokens))
                t_chunk = ('T', obj_head, obj_head + len(obj),major_en[1],len(text_tokens))
                r_chunk = ('R', rel_head, rel_head + len(rel),example.text[rel_index[0]:rel_index[1] + 1],len(text_tokens))
                major_triples.append((h_chunk, t_chunk, major_rel, r_chunk))
        for minor_rel, minor_en in zip(example.minor_re_list, example.minor_pair_list):
            # get sub and obj head
            sub_head, obj_head, sub, obj = _get_so_head(minor_en, tokenizer, text_tokens,off_mapping,example.text)
            if sub_head != -1 and obj_head != -1:
                h_chunk = ('H', sub_head, sub_head + len(sub),minor_en[0],len(text_tokens))
                t_chunk = ('T', obj_head, obj_head + len(obj),minor_en[1],len(text_tokens))
                minor_triples.append((h_chunk, t_chunk, minor_rel))
        sub_feats = [
            InputFeatures(
                input_tokens=text_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                major_triples=major_triples,
                minor_triples=minor_triples,
                off_mapping=off_mapping,
                text = example.text
            )
        ]

    # get sub-feats
    return sub_feats


def convert_examples_to_features(params, examples, tokenizer, major2idx,minor2idx, data_sign):
    """convert examples to features.
    :param examples (List[InputExamples])
    """
    max_text_len = params.max_seq_length
    # multi-process

    ### 10->5
    features = []
    for example in examples:
        temp = convert(example,max_text_len,tokenizer,major2idx,minor2idx,data_sign)
        for m in temp:
            features.append(m)
    return features
