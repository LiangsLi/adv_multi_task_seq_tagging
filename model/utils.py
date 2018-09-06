# -*- coding: utf-8 -*-
# Created on 2018/9/6


def get_begin_end(idxs, tag, FLAGS):
    if FLAGS.num_classes == 4:
        b_l = []
        e_l = []
        find_b = False
        for i in range(len(idxs)):
            if not find_b and tag.idx2tag[idxs[i]] == 'M':
                b_l.append(i)
                find_b = True
                continue
            if i > 0 and tag.idx2tag[idxs[i]] == 'O' and tag.idx2tag[idxs[i - 1]] == 'M' and find_b:
                e_l.append(i - 1)
                find_b = False
                continue
            if i > 0 and tag.idx2tag[idxs[i]] == 'B' and tag.idx2tag[idxs[i - 1]] == 'M' and find_b:
                e_l.append(i - 1)
                b_l.append(i)
                find_b = True
                continue
            if i == len(idxs) - 1 and tag.idx2tag[idxs[i]] == 'M' and find_b:
                e_l.append(i)
                continue
            if tag.idx2tag[idxs[i]] == 'B':
                b_l.append(i)
                find_b = True
                continue
            if tag.idx2tag[idxs[i]] == 'E':
                e_l.append(i)
                continue
            if tag.idx2tag[idxs[i]] == 'O':
                find_b = False
        
        if len(b_l) != len(e_l):
            min_length = min(len(b_l), len(e_l))
            b_l = b_l[:min_length]
            e_l = e_l[:min_length]
        result = []
        for a, b in zip(b_l, e_l):
            if a <= b:
                result.append((a, b))
        return result


def get_match_size(pred_one, real_one):
    if (real_one[0], real_one[1]) == (-1, -1) or (pred_one[0], pred_one[1]) == (-1, -1):
        return 0
    b_, e_ = (max(real_one[0], pred_one[0]), min(real_one[1], pred_one[1]))
    match_size = e_ - b_ + 1
    if match_size < 0:
        match_size = 0
    return match_size


if __name__ == '__main__':
    pass
