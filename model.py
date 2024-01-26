from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertPreTrainedModel, BertModel
import metrics
from utils import Label2IdxSub,Label2IdxRel,Label2IdxObj


class BaseClassifier(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(BaseClassifier, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        features_output = self.hidden2tag(features_tmp)
        return features_output



class BertForRE(BertPreTrainedModel):
    def __init__(self, config, params):
        super().__init__(config)
        self.max_seq_len = params.max_seq_length
        self.seq_tag_size = params.seq_tag_size
        self.major_num = params.major_num
        self.minor_num = params.minor_num
        self.lamda = torch.tensor(0.1,requires_grad=True,dtype=torch.float).to(params.device)
        #self.lamda = torch.zeros(1, requires_grad=True).to(params.device)
        self.temperature = 0.07
        # pretrain model
        self.bert = BertModel(self.config)
        # major relation tagging
        self.major_tagging_subject = BaseClassifier(config.hidden_size * 2, self.seq_tag_size, params.drop_prob)
        self.major_tagging_object = BaseClassifier(config.hidden_size * 2, self.seq_tag_size, params.drop_prob)
        self.major_tagging_mention = BaseClassifier(config.hidden_size * 2, self.seq_tag_size, params.drop_prob)
        # self.minor_tagging_object = BaseClassifier(config.hidden_size * 2, self.seq_tag_size, params.drop_prob)
        self.combination = BaseClassifier(config.hidden_size * 3, 1, params.drop_prob)

        # relation judgement
        self.major_judgement = BaseClassifier(config.hidden_size, params.major_num, params.drop_prob)
        self.minor_judgement = BaseClassifier(config.hidden_size, params.minor_num, params.drop_prob)
        self.major_embedding = nn.Embedding(params.major_num, config.hidden_size)
        self.minor_embedding = nn.Embedding(params.minor_num, config.hidden_size)

        self.init_weights()

    def get_sc_loss(self, rel_log: torch.Tensor, rel_types: torch.Tensor):

        mask = torch.matmul(rel_types,rel_types.T)  #
        rel_log = rel_log.view(mask.shape[0],-1) # logits
        anchor_dot_contrast = torch.div(torch.matmul(rel_log, rel_log.T),self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = -torch.div(anchor_dot_contrast,logits_max.detach())
        # logits = anchor_dot_contrast - logits_max.detach()
        # logits = anchor_dot_contrast

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(rel_log.shape[0]).view(-1, 1).to(mask.device),
            0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask #所有的

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # exp_logits_pos = torch.exp(logits) * mask
        # mask2 = mask.view(rel_logits.shape[0],-1).contiguous()
        # loc = torch.where(mask2 == 0)
        # mask2[loc] = 1
        a = mask.sum(1)
        loc = torch.where(a == 0)
        a[loc] = 1
        mean_log_prob_pos = (mask * log_prob).sum(1) / a #sum里面有0，所以会出现inf
        loss = - (self.temperature / self.temperature) * mean_log_prob_pos
        loss = loss.view(rel_log.shape[0]).mean()
        # d = exp_logits.sum(1, keepdim=True)
        # e = torch.log(exp_logits.sum(1, keepdim=True))
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # a = torch.log(exp_logits_pos.sum(1) / exp_logits.sum(1))
        # b = exp_logits_pos.sum(1) / exp_logits.sum(1)
        # log_prob = logits - torch.log(exp_logits_pos.sum(1) / exp_logits.sum(1))

        # a = (mask * log_prob).sum(1)
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # b = exp_logits.sum(1)
        # c = exp_logits_pos.sum(1)
        # mean_pos = torch.log(exp_logits_pos.sum(1) / exp_logits.sum(1))

        # loss = - (self.temperature / self.temperature) * log_prob
        # loss = loss.mean()
        return loss

    @staticmethod
    def masked_avgpool(sent, mask):
        mask_ = mask.masked_fill(mask == 0, -1e9).float()
        score = torch.softmax(mask_, -1)
        return torch.matmul(score.unsqueeze(1), sent).squeeze(1)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            seq_tags=None,
            major_rel_tags=None,
            minor_rel_tags=None,
            potential_rel=None,
            major_triples=None,
            minor_triples=None,
            ex_params=None
    ):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            major_tags: (bs, major_num)
            minor_tags: (bs, minor_num)
            potential_rels: (bs,), only in train stage.
            seq_tags: (bs, 3, seq_len)
            ex_params: experiment parameters
        """
        # get params for experiments
        major_threshold , minor_threshold =ex_params.get('major_threshold', 0.5),ex_params.get('minor_threshold', 0.4)
        #print(input_ids)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        #print(outputs[0])
        sequence_output = outputs[0]
        # if torch.isnan(outputs[0][0]).sum():
        #     exit(0)
        bs, seq_len, h = sequence_output.size()
        # relation prediction
        h_k_avg = self.masked_avgpool(sequence_output, attention_mask)
        # (bs, rel_num)

        if seq_tags is None:
            major_pred = self.major_judgement(h_k_avg)
            minor_pred = self.minor_judgement(h_k_avg)
            major_pred_onehot = torch.where(torch.sigmoid(major_pred) > major_threshold,
                                            torch.ones(major_pred.size(), device=major_pred.device),
                                            torch.zeros(major_pred.size(), device=major_pred.device))

            minor_pred_onehot = torch.where(torch.sigmoid(minor_pred) > minor_threshold,
                                            torch.ones(minor_pred.size(), device=minor_pred.device),
                                            torch.zeros(minor_pred.size(), device=minor_pred.device))

            # if major relation is null
            for idx, sample in enumerate(major_pred_onehot):
                if 1 not in sample:
                    # (rel_num,)
                    max_index = torch.argmax(major_pred[idx])
                    sample[max_index] = 1
                    major_pred_onehot[idx] = sample

            # if minor relation is null
            for idx, sample in enumerate(minor_pred_onehot):
                if 1 not in sample:
                    # (rel_num,)
                    max_index = torch.argmax(minor_pred[idx])
                    sample[max_index] = 1
                    minor_pred_onehot[idx] = sample

            bs_idxs_major, pred_major = torch.nonzero(major_pred_onehot, as_tuple=True)
            bs_idxs_minor, pred_minor = torch.nonzero(minor_pred_onehot, as_tuple=True)
            # get x_i 每个句子中有几个关系
            xi_dict1 = Counter(bs_idxs_major.tolist())
            xi_dict2 = Counter(bs_idxs_minor.tolist())
            xi = [[xi_dict1[idx]] for idx in range(bs)]
            xi_minor = [[xi_dict2[idx]] for idx in range(bs)]
            major_seq_output = []
            major_potential_rel = []
            minor_seq_output = []
            minor_potential_rel = []
            for bs_idx, rel_idx in zip(bs_idxs_major, pred_major):
                # (seq_len, h)
                major_seq_output.append(sequence_output[bs_idx])
                major_potential_rel.append(rel_idx)
            for bs_idx, rel_idx in zip(bs_idxs_minor, pred_minor):
                minor_seq_output.append(sequence_output[bs_idx])
                minor_potential_rel.append(rel_idx)
            # (sum(x_i), seq_len, h)
            major_sequence = torch.stack(major_seq_output, dim=0)

            # (sum(x_i),)
            potential_major = torch.stack(major_potential_rel, dim=0)

            minor_sequence = torch.stack(minor_seq_output, dim=0)
            # (sum(x_i),)
            potential_minor = torch.stack(minor_potential_rel, dim=0)
            # (bs/sum(x_i), h)
            major_emb = self.major_embedding(potential_major)
            minor_emb = self.minor_embedding(potential_minor)
            # relation embedding vector fusion
            _, seq_len, h = major_sequence.size()
            major_emb = major_emb.unsqueeze(1).expand(-1, seq_len, h)
            minor_emb = minor_emb.unsqueeze(1).expand(-1, seq_len, h)
            major_decode_input = torch.cat([major_sequence, major_emb], dim=-1)


            minor_decode_input = torch.cat([minor_sequence, minor_emb], dim=-1)
            # (bs/sum(x_i), seq_len, tag_size)
            major_subject = self.major_tagging_subject(major_decode_input)
            major_object = self.major_tagging_object(major_decode_input)
            major_mention = self.major_tagging_mention(major_decode_input)
            minor_object = self.major_tagging_object(minor_decode_input)
            pred_major_sub = torch.argmax(torch.softmax(major_subject, dim=-1), dim=-1)
            pred_major_obj = torch.argmax(torch.softmax(major_object, dim=-1), dim=-1)
            pred_major_men = torch.argmax(torch.softmax(major_mention, dim=-1), dim=-1)
            pred_minor_obj = torch.argmax(torch.softmax(minor_object, dim=-1), dim=-1)
            # (sum(x_i), 3(1), seq_len)
            pred_major_tag = torch.cat([pred_major_sub.unsqueeze(1), pred_major_obj.unsqueeze(1),
                                    pred_major_men.unsqueeze(1)], dim=1)
            pred_minor_tag = torch.cat([pred_minor_obj.unsqueeze(1), pred_minor_obj.unsqueeze(1)], dim=1)


            xi = np.array(xi)
            xi_minor = np.array(xi_minor)
            xi_index = np.cumsum(xi).tolist()
            xi_minor_index = np.cumsum(xi_minor).tolist()
            xi_index.insert(0, 0)
            xi_minor_index.insert(0, 0)
            pred_major_tag = pred_major_tag.detach().cpu().numpy()
            pred_minor_tag = pred_minor_tag.detach().cpu().numpy()
            pred_major = pred_major.detach().cpu().numpy()
            pred_minor = pred_minor.detach().cpu().numpy()
            head_minor=[]
            for idx in range(bs):
                head_temp=[]
                pre_major_triples = metrics.tag_mapping_major(predict_tags=pred_major_tag[xi_index[idx]:xi_index[idx + 1]],
                                                      pre_rels=pred_major[xi_index[idx]:xi_index[idx + 1]],
                                                              seq_len=attention_mask[idx].sum(),
                                                      label2idx_sub=Label2IdxSub,
                                                      label2idx_obj=Label2IdxObj)
                pre_major_mention = metrics.tag_mapping_single(predict_tags=pred_major_tag[xi_index[idx]:xi_index[idx + 1]],
                                                       pre_rels=pred_major[xi_index[idx]:xi_index[idx + 1]],
                                                               seq_len=attention_mask[idx].sum(),
                                                       index=2,
                                                       label2idx=Label2IdxRel)
                pre_minor = metrics.tag_mapping_single(predict_tags=pred_minor_tag[xi_minor_index[idx]:xi_minor_index[idx + 1]],
                                                       pre_rels=pred_minor[xi_minor_index[idx]:xi_minor_index[idx + 1]],
                                                       seq_len=attention_mask[idx].sum(),
                                                       index=1,
                                                       label2idx=Label2IdxObj)
                pre_obj =[]
                temp = []
                ref = []
                for major in pre_major_triples:
                    mention = metrics.get_mention(major[-1], pre_major_mention)
                    subject = major[0]
                    object = major[1]
                    subject = torch.mean(major_sequence[xi_index[idx],subject[1]:subject[2],:],dim=0).view(1,-1)
                    object = torch.mean(major_sequence[xi_index[idx],object[1]:object[2],:],dim=0).view(1,-1)
                    temp.append(subject)
                    ref.append(major[0])
                    temp.append(object)
                    ref.append(major[1])
                    for men in mention:
                        res = torch.mean(major_sequence[xi_index[idx], men[1]:men[2], :], dim=0).view(1,-1)
                        temp.append(res)
                        ref.append(men)
                candidate = []
                if temp:
                    candidate = torch.cat(temp, dim=0)
                rel = []
                for minor in pre_minor:
                    pre_obj.append(torch.mean(minor_sequence[xi_minor_index[idx],minor[0][1]:minor[0][2],:],dim=0).view(1,-1))
                    rel.append(minor[-1])
                pre_minor_rel = self.minor_embedding(torch.LongTensor(rel).to(sequence_output.device))
                if pre_obj and temp:
                    pre_obj = torch.cat(pre_obj, dim=0)
                    # l_1*2h
                    minor_trip = torch.cat([pre_minor_rel, pre_obj], dim=1)
                    for index in range(minor_trip.shape[0]):
                        candidate_comb = minor_trip[index, :].view(1, -1).expand(len(ref), -1)
                        combinations = torch.cat([candidate, candidate_comb], dim=1)
                        temp_result = self.combination(combinations)
                        target = torch.argmax(temp_result, dim=0)
                        head_temp.append(ref[target])

                head_minor.append(head_temp)
            # head每个元素代表一个次关系对应的所有主语，排列顺序依照次关系宾语
            return pred_major_tag, pred_minor_tag, xi, xi_minor, pred_major, pred_minor,head_minor

        else:
            major_bs = []
            minor_bs = []
            for index in range(bs):
                if major_triples[index]:
                   major_bs.append(index)
                if minor_triples[index]:
                   minor_bs.append(index)
                if major_triples[index] and minor_triples[index]:
                   raise ValueError('Cannot be main and auxiliary the same time!!')

            major_en = [major_triples[i] for i in major_bs]
            minor_en = [minor_triples[i] for i in minor_bs]
            major_bs = torch.LongTensor(major_bs).to(sequence_output.device)
            minor_bs = torch.LongTensor(minor_bs).to(sequence_output.device)
            major_sequence = sequence_output.index_select(0,major_bs)
            major_attention = attention_mask.index_select(0,major_bs)
            minor_sequence = sequence_output.index_select(0,minor_bs)
            minor_attention = attention_mask.index_select(0,minor_bs)
            h_k_avg_major = self.masked_avgpool(major_sequence, major_attention)
            h_k_avg_minor = self.masked_avgpool(minor_sequence, minor_attention)
            major_pred = self.major_judgement(h_k_avg_major)
            minor_pred = self.minor_judgement(h_k_avg_minor)
            major_emb = self.major_embedding(potential_rel.index_select(0,major_bs))
            minor_emb = self.minor_embedding(potential_rel.index_select(0,minor_bs))
            major_emb = major_emb.unsqueeze(1).expand(-1, seq_len, h)
            minor_emb = minor_emb.unsqueeze(1).expand(-1, seq_len, h)
            major_decode_input = torch.cat([major_sequence, major_emb], dim=-1)
            minor_decode_input = torch.cat([minor_sequence, minor_emb], dim=-1)
            major_subject = self.major_tagging_subject(major_decode_input)
            major_object = self.major_tagging_object(major_decode_input)
            major_mention = self.major_tagging_mention(major_decode_input)
            minor_object = self.major_tagging_object(minor_decode_input)
            bs_major = []
            bs_candidate = []
            for idx in range(len(major_en)):
                temp = []
                major_candidate = []
                for major in major_en[idx]:
                    subject = torch.mean(major_sequence[idx, major[0][1]:major[0][2], :], dim=0).view(1, -1)
                    object = torch.mean(major_sequence[idx, major[1][1]:major[1][2], :], dim=0).view(1, -1)
                    mention = torch.mean(major_sequence[idx, major[-1][1]:major[-1][2], :], dim=0).view(1, -1)
                    temp.append(subject)
                    major_candidate.append(major[0][1:5])
                    temp.append(object)
                    major_candidate.append(major[1][1:5])
                    temp.append(mention)
                    major_candidate.append(major[-1][1:5])
                if temp:
                    bs_major.append(torch.cat(temp, dim=0))
                else:
                    bs_major.append([])
                bs_candidate.append(major_candidate)
            bs_rel = []
            bs_minor = []
            candidate_pair = []
            for idx in range(len(minor_en)):
                rel = []
                temp_minor = []
                temp_pair = []
                for minor in minor_en[idx]:
                    object = torch.mean(minor_sequence[idx, minor[1][1]:minor[1][2], :], dim=0).view(1, -1)
                    temp_minor.append(object)
                    for i in range(len(bs_candidate)):
                        if minor[0][1:5] in bs_candidate[i]:
                            temp_pair.append([i,bs_candidate[i].index(minor[0][1:5])])
                    rel.append(minor[-1])
                if not temp_pair:
                    temp_pair.append([])
                bs_rel.append(rel)
                if temp_minor:
                    bs_minor.append(torch.cat(temp_minor, dim=0))
                else:
                    bs_minor.append([])
                if len(temp_pair)!=len(rel):
                    temp_pair = temp_pair[:len(rel)]
                candidate_pair.append(temp_pair)
            rel_li = []
            tri_li = []
            loss_func1 = nn.BCEWithLogitsLoss(reduction='none')
            loss_combination = 0
            add_count = 0
            for j in range(len(bs_rel)):
                if [] not in candidate_pair[j]:
                    pre_minor_rel = self.minor_embedding(torch.tensor(bs_rel[j], dtype=torch.long, device=sequence_output.device))
                    rel_type = F.one_hot(torch.tensor(bs_rel[j], dtype=torch.long, device=sequence_output.device), num_classes=self.minor_num).float()
                    minor_re = torch.cat([pre_minor_rel, bs_minor[j]], dim=1)
                    head_extend = bs_major[candidate_pair[j][0][0]].unsqueeze(1).expand(-1,minor_re.shape[0],-1)
                    tail_extend = minor_re.unsqueeze(0).expand(bs_major[candidate_pair[j][0][0]].shape[0], -1, -1)
                    combination_matrix = torch.cat([head_extend, tail_extend], dim=2)
                    combination_pred = self.combination(combination_matrix).squeeze(-1)
                    pair = []
                    for i in range(len(candidate_pair[j])):
                        pair.append(candidate_pair[j][i][1])
                    matrix_initial = torch.scatter(
                        torch.zeros_like(combination_pred, device=sequence_output.device),
                        0,
                        torch.tensor(pair, dtype=torch.long, device=sequence_output.device).view(1,-1),
                        1)
                    loss_combination += loss_func1(combination_pred.view(-1), matrix_initial.view(-1)).sum() / \
                                       torch.ones_like(combination_pred).sum()
                    add_count += 1
                    for i in range(combination_pred.shape[1]):
                        for k in range(combination_pred.shape[0]):
                            rel_li.append(rel_type[i])
                            tri_li.append(combination_matrix[k][i])

            if rel_li:
                rel_li = torch.stack(rel_li, dim=0)
                tri_li = torch.stack(tri_li, dim=0)
                loss_cl = self.get_sc_loss(tri_li, rel_li)
            else:
                loss_cl = torch.zeros(1, requires_grad=True).to(params.device)
            #combination_tag = matrix_initial.detach()
            major_attention_mask = major_attention.view(-1)
            minor_attention_mask = minor_attention.view(-1)
            major_tags = seq_tags.index_select(0, major_bs)
            minor_tags = seq_tags.index_select(0, minor_bs)
            loss_func = nn.CrossEntropyLoss(reduction='none')


            loss_major_sub = (loss_func(major_subject.view(-1, self.seq_tag_size),
                                        major_tags[:, 0, :].reshape(
                                            -1)) * major_attention_mask).sum() / major_attention_mask.sum()
            loss_major_obj = (loss_func(major_object.view(-1, self.seq_tag_size),
                                        major_tags[:, 1, :].reshape(
                                            -1)) * major_attention_mask).sum() / major_attention_mask.sum()
            loss_major_men = (loss_func(major_mention.view(-1, self.seq_tag_size),
                                        major_tags[:, 2, :].reshape(
                                            -1)) * major_attention_mask).sum() / major_attention_mask.sum()
            loss_minor_obj = (loss_func(minor_object.view(-1, self.seq_tag_size),
                                        minor_tags[:, 1, :].reshape(
                                            -1)) * minor_attention_mask).sum() / minor_attention_mask.sum()

            loss_seq = (loss_major_sub + loss_major_men + loss_major_obj + loss_minor_obj) / 4
            if not (loss_major_sub > 0):
                loss_seq = loss_minor_obj
            if not (loss_minor_obj > 0):
                loss_seq = (loss_major_sub + loss_major_men + loss_major_obj)/3
            # init
            major_rel = major_rel_tags.index_select(0, major_bs)
            minor_rel = minor_rel_tags.index_select(0, minor_bs)
            loss_func = nn.BCEWithLogitsLoss(reduction='mean')
            loss_major_rel = loss_func(major_pred, major_rel.float())
            loss_minor_rel = loss_func(minor_pred, minor_rel.float())
            loss_rel = (loss_major_rel + loss_minor_rel) / 2
            if not (loss_major_rel > 0):
                loss_rel = loss_minor_rel
            if not (loss_minor_rel > 0):
                loss_rel = loss_major_rel
            loss = loss_seq + loss_rel + (1-self.lamda)*loss_combination/add_count + self.lamda*loss_cl
            # loss = loss_seq + loss_rel +  loss_combination/add_count
            # loss = loss_seq + loss_rel
            return loss, loss_seq, loss_rel,loss_cl ,loss_combination





if __name__ == '__main__':
    from transformers import BertConfig
    import utils
    import os

    params = utils.Params()
    # Prepare model
    bert_config = BertConfig.from_json_file(os.path.join(params.bert_model_dir, 'bert_config.json'))
    model = BertForRE.from_pretrained(config=bert_config,
                                      pretrained_model_name_or_path=params.bert_model_dir,
                                      params=params)
    model.to(params.device)

    for n, _ in model.named_parameters():
        print(n)

