# /usr/bin/env python
# coding=utf-8
"""Dataloader"""

import os
import json

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import BertTokenizerFast

from dataloader_utils import read_examples, convert_examples_to_features
from utils import major_num,minor_num

class FeatureDataset(Dataset):
    """Pytorch Dataset for InputFeatures
    """

    def __init__(self, features):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


class CustomDataLoader(object):
    def __init__(self, params):
        self.params = params

        self.train_batch_size = params.train_batch_size
        self.val_batch_size = params.val_batch_size
        self.test_batch_size = params.test_batch_size

        self.data_dir = params.data_dir
        self.max_seq_length = params.max_seq_length
        self.tokenizer = BertTokenizerFast(vocab_file=os.path.join(params.bert_model_dir, 'vocab.txt'),
                                       do_lower_case=False)
        self.data_cache = params.data_cache

    @staticmethod
    def collate_fn_train(features):
        """将InputFeatures转换为Tensor
        Args:
            features (List[InputFeatures])
        Returns:
            tensors (List[Tensors])
        """
        input_ids = []
        attention_mask = []
        seq_tags = []
        major_triples = []
        minor_triples = []
        poten_relations = []
        major_rel_tags = []
        minor_rel_tags = []
        for f in features:
            major_tags, minor_tags = f.seq_tags
            major_re, minor_re = f.relation
            for m in range(len(major_tags)):
                input_ids.append(f.input_ids)
                attention_mask.append(f.attention_mask)
                seq_tags.append(major_tags[m])
                major_triples.append(f.major_triples[m])
                minor_triples.append(None)
                poten_relations.append(major_re[m])
                major_rel_tags.append(f.major_rel_tag)
                minor_rel_tags.append(minor_num * [0])
            for m in range(len(minor_tags)):
                input_ids.append(f.input_ids)
                attention_mask.append(f.attention_mask)
                seq_tags.append(minor_tags[m])
                major_triples.append(None)
                minor_triples.append(f.minor_triples[m])
                poten_relations.append(minor_re[m])
                major_rel_tags.append(major_num * [0])
                minor_rel_tags.append(f.minor_rel_tag)
        input_ids = torch.tensor(input_ids,dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        seq_tags = torch.tensor(seq_tags, dtype=torch.long)
        poten_relations = torch.tensor(poten_relations, dtype=torch.long)
        major_rel_tags = torch.tensor(major_rel_tags, dtype=torch.long)
        minor_rel_tags = torch.tensor(minor_rel_tags, dtype=torch.long)
        tensors = [input_ids, attention_mask, seq_tags, poten_relations, major_rel_tags,minor_rel_tags,major_triples,minor_triples]
        return tensors

    @staticmethod
    def collate_fn_test(features):
        """将InputFeatures转换为Tensor
        Args:
            features (List[InputFeatures])
        Returns:
            tensors (List[Tensors])
        """
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        major_triples = [f.major_triples for f in features]
        minor_triples = [f.minor_triples for f in features]
        input_tokens = [f.input_tokens for f in features]
        off_mapping = [f.off_mapping for f in features]
        text = [f.text for f in features]
        tensors = [input_ids, attention_mask, major_triples,minor_triples, input_tokens, off_mapping,text]

        return tensors

    def get_features(self, data_sign, ex_params):
        """convert InputExamples to InputFeatures
        :param data_sign: 'train', 'val' or 'test'
        """
        print("=*=" * 10)
        print("Loading {} data...".format(data_sign))
        # get features
        cache_path = os.path.join(self.data_dir, "{}.cache.{}".format(data_sign, str(self.max_seq_length)))
        if os.path.exists(cache_path) and self.data_cache:
            features = torch.load(cache_path)
        else:
            # get relation to idx
            with open(self.data_dir / f'major2id.json', 'r', encoding='utf-8') as f_re:
                major2idx = json.load(f_re)[-1]
            with open(self.data_dir / f'auxiliary2id.json', 'r', encoding='utf-8') as f_re:
                minor2idx = json.load(f_re)[-1]
            # get examples
            if data_sign in ("train", "val", "test"):
                examples = read_examples(self.data_dir, data_sign=data_sign, major2idx=major2idx,minor2idx=minor2idx)
            else:
                raise ValueError("please notice that the data can only be train/val/test!!")
            features = convert_examples_to_features(self.params, examples, self.tokenizer, major2idx,minor2idx, data_sign)
            # save data
            if self.data_cache:
                torch.save(features, cache_path)
        return features

    def get_dataloader(self, data_sign="train", ex_params=None):
        """construct dataloader
        :param data_sign: 'train', 'val' or 'test'
        """
        # InputExamples to InputFeatures
        features = self.get_features(data_sign=data_sign, ex_params=ex_params)
        dataset = FeatureDataset(features)
        print(f"{len(features)} {data_sign} data loaded!")
        print("=*=" * 10)
        # construct dataloader
        # RandomSampler(dataset) or SequentialSampler(dataset)
        if data_sign == "train":
            datasampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.train_batch_size,
                                    collate_fn=self.collate_fn_train)
        elif data_sign == "val":
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.val_batch_size,
                                    collate_fn=self.collate_fn_test)
        elif data_sign == "test":
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.test_batch_size,
                                    collate_fn=self.collate_fn_test)
        else:
            raise ValueError("please notice that the data can only be train/val/test !!")
        return dataloader



