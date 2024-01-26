# /usr/bin/env python
# coding=utf-8
from collections import defaultdict

def get_mention(rel, rel_mention):
    mention=[]
    for men in rel_mention:
        if men[-1]==rel:
            mention.append(men[0])
    return mention


def get_nary(major_triples, rel_mention, minor_triples):
    nary_list=[]
    for major in major_triples:
        mention = get_mention(major[-1],list(set(rel_mention)))
        minor_list= defaultdict(list)
        minor_candidate=[]
        for minor in minor_triples:
            if minor[0] in mention:
                minor_list['mention'].append(minor)
                minor_candidate.append(minor)
            elif minor[0] == major[0] and minor[1] != major[1]:
                minor_list['subject'].append(minor)
                minor_candidate.append(minor)
            elif minor[0] == major[1]:
                minor_list['object'].append(minor)
                minor_candidate.append(minor)
        nary_list.append([major,dict(minor_list),minor_candidate])
    return nary_list

def get_event(major_triples, rel_mention, minor_triples):
    nary_list = []
    for major in major_triples:
        mention = get_mention(major[-1], list(set(rel_mention)))
        minor_list = defaultdict(list)
        minor_candidate = []
        for minor in minor_triples:
            minor_candidate.append(minor[1:3])
            if minor[0] in mention:
                minor_list['mention'].append(minor)
            elif minor[0] == major[0] and minor[1] != major[1]:
                minor_list['subject'].append(minor)
            elif minor[0] == major[1]:
                minor_list['object'].append(minor)
        nary_list.append([major, dict(minor_list), minor_candidate])
    return nary_list

def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    content = tag_name.split('-')
    tag_class = content[0]
    if len(content) == 1:
        return tag_class
    ht = content[-1]
    return tag_class, ht


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position
    Args:
        seq: np.array[4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    default1 = tags['O']
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default1 and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default1:
            res = get_chunk_type(tok, idx_to_tag)
            if len(res) == 1:
                continue
            tok_chunk_class, ht = get_chunk_type(tok, idx_to_tag)
            tok_chunk_type = ht
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
    return chunks


def tag_mapping_major(predict_tags,seq_len,pre_rels=None, label2idx_sub=None, label2idx_obj=None):
    """
    implement of the heuristic nearest principle
    Args:
        predict_tags: np.array, (xi, 3, max_sen_len)
        pre_rels: (xi,)
    """
    rel_num = predict_tags.shape[0]
    pre_triples = []
    for idx in range(rel_num):
        heads, tails = [], []
        pred_chunks_sub = get_chunks(predict_tags[idx][0], label2idx_sub)
        pred_chunks_obj = get_chunks(predict_tags[idx][1], label2idx_obj)

        pred_chunks = pred_chunks_sub + pred_chunks_obj
        for ch in pred_chunks:
            if ch[1] >= seq_len:
                continue
            if ch[0] == 'H':
                heads.append(ch)
            elif ch[0] == 'T':
                tails.append(ch)
        # the heuristic nearest principle
        if len(heads) != 0 and len(tails) != 0:
            if len(heads) < len(tails):
                heads += [heads[-1]] * (len(tails) - len(heads))
            if len(heads) > len(tails):
                tails += [tails[-1]] * (len(heads) - len(tails))

        for h_t in zip(heads, tails):
            if pre_rels is not None:
                triple = list(h_t) + [pre_rels[idx]]
            else:
                triple = list(h_t) + [idx]
            pre_triples.append(tuple(triple))
    return pre_triples

def tag_mapping_minor(head,seq_len, predict_tags, pre_rels=None,  label2idx_obj=None):
    """
    implement of the heuristic nearest principle
    Args:
        predict_tags: np.array, (xi, 3, max_sen_len)
        pre_rels: (xi,)
    """
    pre_triples = []
    if head:
        rel_num = predict_tags.shape[0]
        pointer = 0
        for idx in range(rel_num):
            heads, tails = [], []
            pred_chunks_obj = get_chunks(predict_tags[idx][1], label2idx_obj)
            for ch in pred_chunks_obj:
                if ch[1] >= seq_len:
                    continue
                heads.append(head[pointer])
                pointer += 1
                tails.append(ch)
            # the heuristic nearest principle
            if len(heads) != 0 and len(tails) != 0:
                if len(heads) < len(tails):
                    heads += [heads[-1]] * (len(tails) - len(heads))
                if len(heads) > len(tails):
                    tails += [tails[-1]] * (len(heads) - len(tails))

            for h_t in zip(heads, tails):
                if pre_rels is not None:
                    triple = list(h_t) + [pre_rels[idx]]
                else:
                    triple = list(h_t) + [idx]
                pre_triples.append(tuple(triple))
    return pre_triples



def tag_mapping_single(predict_tags,seq_len,pre_rels, index, label2idx):
    """
    implement of the heuristic nearest principle
    Args:
        predict_tags: np.array, (xi, 3, max_sen_len)
    """
    output=[]
    rel_num = predict_tags.shape[0]
    for idx in range(rel_num):
        pred_chunks = get_chunks(predict_tags[idx][index], label2idx)

        for en in pred_chunks:
            if en[1] >= seq_len:
                continue
            output.append((en,pre_rels[idx]))
        # the heuristic nearest principle
    return output




