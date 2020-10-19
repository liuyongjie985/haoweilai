import pickle as pkl
import json
import sys
import random
from torch import nn
import numpy as np
import torch
import math
import copy

'''
分词函数
传入：未分词的latex字符串,词表
传出：分词后的latex list
'''


def deal_symbol_label(symbol_label, symbol_dict, max_length=999):
    result_str = []
    start_index = 0
    while start_index < len(symbol_label):
        if symbol_label[start_index] == " " or symbol_label[start_index] == "$":
            start_index += 1
            continue
        sign = 0
        i = 0
        while i < max_length:
            if symbol_label[start_index:start_index + max_length - i] in symbol_dict:
                result_str.append(symbol_label[start_index:start_index + max_length - i])
                start_index = start_index + max_length - i
                sign = 1
                break
            i += 1
        if sign == 0:
            print(symbol_label + " 中有字典中没有的字符")
            result_str = []
    return result_str


# ================================================================================================

'''
减均值μx 除以标准差δx

'''


def z_score(traceid2xy):
    u_x_numerator = 0
    u_x_denominator = 0
    u_y_numerator = 0
    u_y_denominator = 0
    if len(traceid2xy) == 0 or len(traceid2xy[0]) == 0:
        return traceid2xy, -1, -1
    for i, x in enumerate(traceid2xy):
        for j, y in enumerate(x):
            if j == 0:
                continue
            print("y", y)
            L = ((y[0] - x[j - 1][0]) ** 2 + (y[1] - x[j - 1][1]) ** 2) ** 0.5
            u_x_numerator += L * (y[0] + x[j - 1][0]) / 2
            u_x_denominator += L
            u_y_numerator += L * (y[1] + x[j - 1][1]) / 2
            u_y_denominator += L
    u_x = u_x_numerator / u_x_denominator
    u_y = u_y_numerator / u_y_denominator
    delta_x_numerator = 0
    delta_x_denominator = 0
    for i, x in enumerate(traceid2xy):
        for j, y in enumerate(x):
            if j == 0:
                continue
            L = ((y[0] - x[j - 1][0]) ** 2 + (y[1] - x[j - 1][1]) ** 2) ** 0.5
            delta_x_numerator += L / 3 * (
                    (y[0] - u_x) ** 2 + (x[j - 1][0] - u_x) ** 2 + (x[j - 1][0] - u_x) * (y[0] - u_x))
            delta_x_denominator += L

    delta_x = (delta_x_numerator / delta_x_denominator) ** 0.5

    new_traceid2xy = []
    count_x = 0
    count_y = 0
    count = 0
    for i, x in enumerate(traceid2xy):
        temp = []
        for j, y in enumerate(x):
            temp.append([(y[0] - u_x) / delta_x, (y[1] - u_y) / delta_x, y[2], y[3]])
            count_x += (y[0] - u_x) / delta_x
            count_y += (y[1] - u_y) / delta_x
            count += 1
        new_traceid2xy.append(temp)
    avg_x = count_x / count
    avg_y = count_y / count
    return new_traceid2xy, avg_x, avg_y


'''
计算T_cos与T_dis，去除多余点，以每个笔画为单位
'''


def rve_duplicate(traceid2xy, T_dis, T_cos, NO_PRINT=True):
    count = 0
    for i, x in enumerate(traceid2xy):
        j = 0
        while j < len(x):
            if j == 0:
                # temp_list.append([x[j][0], x[j][1], x[j][2], x[j][3]])
                j += 1
                continue
            real_dis = ((x[j][0] - x[j - 1][0]) ** 2 + (x[j][1] - x[j - 1][1]) ** 2) ** 0.5
            if not real_dis < T_dis:
                # temp_list.append([x[j][0], x[j][1], x[j][2], x[j][3]])
                j += 1
            else:
                if j != len(x) - 1:
                    x.pop(j)
                    if NO_PRINT != True:
                        print("因为距离删除一个点")
                else:
                    if NO_PRINT != True:
                        print("原本要删除的点为抬笔点，保留")
                    j += 1

                count += 1
    for i, x in enumerate(traceid2xy):
        j = 0
        while j < len(x):
            if j == 0 or j == len(x) - 1:
                j += 1
                continue
            if (((x[j][0] - x[j - 1][0]) ** 2 + (x[j][1] - x[j - 1][1]) ** 2) ** 0.5 * (
                    ((x[j + 1][0] - x[j][0]) ** 2 + (x[j + 1][1] - x[j][1]) ** 2) ** 0.5)) == 0:
                j += 1
                continue
            real_cos = abs(
                ((x[j][0] - x[j - 1][0]) * (x[j + 1][0] - x[j][0]) + (x[j][1] - x[j - 1][1]) * (
                        x[j + 1][1] - x[j][1])) / (
                        ((x[j][0] - x[j - 1][0]) ** 2 + (x[j][1] - x[j - 1][1]) ** 2) ** 0.5 * (
                        ((x[j + 1][0] - x[j][0]) ** 2 + (x[j + 1][1] - x[j][1]) ** 2) ** 0.5)))
            if not real_cos < T_cos:
                j += 1
            else:
                x.pop(j)
                if NO_PRINT != True:
                    print("因为角度删除一个点")
                count += 1
    return traceid2xy, count


def zscore_first_feature_extraction(traceid2xy):
    new_traceid2xy = []
    for i, x in enumerate(traceid2xy):
        for j, y in enumerate(x):
            if j != len(x) - 1:
                if j != len(x) - 2:
                    new_traceid2xy.append([y[0], y[1], x[j + 1][0] - y[0], x[j + 1][1] - y[1],
                                           x[j + 2][0] - y[0], x[j + 2][1] - y[1], 0.0, y[2], y[3]])
                else:
                    new_traceid2xy.append([y[0], y[1], x[j + 1][0] - y[0], x[j + 1][1] - y[1],
                                           0.0, 0.0, 0.0, y[2], y[3]])
            else:
                new_traceid2xy.append([y[0], y[1], 0.0, 0.0, 0.0, 0.0, 0.0, y[2], y[3]])
    return new_traceid2xy


# =========================================================================================
'''load dictionary'''


def load_dict(dictFile):
    fp = open(dictFile)
    stuff = fp.readlines()
    fp.close()
    lexicon = {}
    for l in stuff:
        w = l[:-1].split("\t")
        lexicon[w[0]] = int(w[1])
    print('total words/phones', len(lexicon))
    return lexicon


# =========================================================================================
def dataGenerator(feature_file, label_file, dictionary, batch_size, maxlen, match_batch=True):
    fp = open(feature_file, 'rb')  # read kaldi scp file
    features = pkl.load(fp)  # load features in dict
    fp.close()

    fp2 = open(label_file, 'r')
    labels = json.load(fp2)
    fp2.close()

    # **********************************Symbol classify 's label**********************************

    targets = {}
    # map word to int with dictionary
    for l in labels:
        tmp = l
        assert len(tmp) == 2
        uid = tmp[0]
        w_list = []
        for w in tmp[1]:
            if w in dictionary:
                w_list.append(dictionary[w])
            else:
                print('a word not in the dictionary !! sentence ', uid, 'word ', w)
                sys.exit()
        targets[uid] = w_list
    # **********************************************************************************************
    # ××××××××××××××××××××××××××××××××××××××××收集所有样例拥有坐标点数并排序××××××××××××××××××××××××××××××
    sentLen = {}
    for uid, fea in features.items():
        sentLen[uid] = len(fea)

    sentLen = sorted(sentLen.items(),
                     key=lambda d: d[1])  # sorted by sentence length,  return a list with each triple element
    # ×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    # ××××××××××××××××××××××××××××××××××××按照坐标点数排序生成batch××××××××××××××××××××××××××××××××××××××
    feature_batch = []
    label_batch = []

    feature_total = []
    label_total = []

    uidList = []

    i = 0
    max_length_fea = -1
    min_length_fea = 9999999999
    for uid, length in sentLen:
        fea = features[uid]
        if len(fea) > max_length_fea:
            max_length_fea = len(fea)
        if len(fea) < min_length_fea:
            min_length_fea = len(fea)
        lab = targets[uid]

        if len(lab) > maxlen or len(fea) == 0:
            if len(fea) == 0:
                print('sentence', uid, 'x length equal ', 0, ' ignore')
            else:
                print('sentence', uid, 'y length bigger than ', maxlen, ' ignore')
        else:
            uidList.append(uid)
            if i == batch_size:  # a batch is full
                feature_total.append(feature_batch)
                label_total.append(label_batch)
                i = 0
                feature_batch = []
                label_batch = []
                feature_batch.append(fea)
                label_batch.append(lab)
                i = i + 1
            else:
                feature_batch.append(fea)
                label_batch.append(lab)
                i = i + 1
    if match_batch:
        # last batch
        while len(feature_batch) < batch_size:
            index1 = random.randint(0, len(feature_total) - 1)
            index2 = random.randint(0, batch_size - 1)
            feature_batch.append(feature_total[index1][index2])
            label_batch.append(label_total[index1][index2])
        assert len(feature_batch) == batch_size and len(label_batch) == batch_size
        feature_total.append(feature_batch)
        label_total.append(label_batch)
    else:
        feature_total.append(feature_batch)
        label_total.append(label_batch)
    # ×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    print('total ', len(feature_total), 'batch data loaded')
    print('最长特征点 ', max_length_fea)
    print('最短特征点 ', min_length_fea)
    new_total = []
    for i, x in enumerate(feature_total):
        new_total.append((x, label_total[i]))
    return new_total, len(feature_total), uidList


# =========================================================================================


# init model params
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias != None:
            nn.init.constant_(m.bias.data, 0.)

    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias != None:
            nn.init.constant_(m.bias.data, 0.)


def argmax_dataIterator(params, seqs_x, seqs_y, maxlen=None, n_words_src=30000, n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []

        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)

        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x) + 1
    maxlen_y = np.max(lengths_y) + 1
    x = np.zeros((maxlen_x, n_samples, params['dim_feature'])).astype('float32')  # SeqX * batch * dim
    y = np.zeros((maxlen_y, n_samples)).astype('int64')  # the <eol> must be 0     in the dict !!!
    x_mask = np.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = np.zeros((maxlen_y, n_samples)).astype('float32')

    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:lengths_x[idx], idx, :] = s_x  # the zeros frame is a padding frame     to align <eol>
        x_mask[:lengths_x[idx] + 1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx] + 1, idx] = 1.
    return x, x_mask, y, y_mask


# beam search
def gen_sample(model, x, params, gpu_flag, k=1, maxlen=30, stochastic=False,
               argmax=False):
    sample = []
    sample_score = []
    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = np.zeros(live_k).astype(np.float32)
    # print("valid Outside: input size", x.size())
    if gpu_flag:
        next_state, ctx0 = model.module.tap_f_init(params, x)
    else:
        next_state, ctx0 = model.tap_f_init(params, x)
    next_w = -1 * np.ones((1,)).astype(np.int64)
    next_w = torch.from_numpy(next_w).cuda()
    SeqL = x.shape[0]
    hidden_sizes = params['hidden_size']
    for i in range(len(hidden_sizes)):
        if params['down_sample'][i] == 1:
            SeqL = math.ceil(SeqL / 2.)

    next_alpha_past = torch.zeros(1, int(SeqL)).cuda()  # start position

    ctx0 = ctx0.cpu().numpy()

    for ii in range(maxlen):
        ctx = np.tile(ctx0, [live_k, 1])
        ctx = torch.from_numpy(ctx).cuda()
        if gpu_flag:
            next_p, next_state, next_alpha_past = model.module.tap_f_next(params, next_w, ctx, next_state,
                                                                          next_alpha_past)
        else:
            next_p, next_state, next_alpha_past = model.tap_f_next(params, next_w, ctx, next_state, next_alpha_past)
        next_p = next_p.cpu().numpy()
        next_state = next_state.cpu().numpy()
        next_alpha_past = next_alpha_past.cpu().numpy()

        cand_scores = hyp_scores[:, None] - np.log(next_p)
        cand_flat = cand_scores.flatten()
        ranks_flat = cand_flat.argsort()[:(k - dead_k)]

        voc_size = next_p.shape[1]
        trans_indices = ranks_flat // voc_size
        word_indices = ranks_flat % voc_size
        costs = cand_flat[ranks_flat]

        new_hyp_samples = []
        new_hyp_scores = np.zeros(k - dead_k).astype(np.float32)
        new_hyp_states = []
        new_hyp_alpha_past = []
        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            new_hyp_samples.append(hyp_samples[ti] + [wi])
            new_hyp_scores[idx] = copy.copy(costs[idx])
            new_hyp_states.append(copy.copy(next_state[ti]))
            new_hyp_alpha_past.append(copy.copy(next_alpha_past[ti]))

        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_states = []
        hyp_alpha_past = []
        for idx in range(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1] == 0:
                sample.append(new_hyp_samples[idx])
                sample_score.append(new_hyp_scores[idx])
                dead_k += 1
            else:
                new_live_k += 1
                hyp_samples.append(new_hyp_samples[idx])
                hyp_scores.append(new_hyp_scores[idx])
                hyp_states.append(new_hyp_states[idx])
                hyp_alpha_past.append(new_hyp_alpha_past[idx])
        hyp_scores = np.array(hyp_scores)
        live_k = new_live_k

        # whether finish beam search
        if new_live_k < 1:
            break
        if dead_k >= k:
            break

        next_w = np.array([w[-1] for w in hyp_samples])
        next_state = np.array(hyp_states)
        next_alpha_past = np.array(hyp_alpha_past)
        next_w = torch.from_numpy(next_w).cuda()
        next_state = torch.from_numpy(next_state).cuda()
        next_alpha_past = torch.from_numpy(next_alpha_past).cuda()
    return sample, sample_score
