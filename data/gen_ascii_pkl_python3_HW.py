#!/usr/bin/env python

import pickle as pkl
import numpy as np
import json

FLAG = 'train'

feature_path = "./" + FLAG + "_ascii/"  # for test.pkl, change 'train' to 'test'
outFile = 'haoweilai_' + FLAG + '.pkl'
oupFp_feature = open(outFile, 'wb')

features = {}

sentNum = 0
already = 0

scpFile = open("haoweilai_" + FLAG + '_caption.txt')
data = json.load(scpFile)
for i, x in enumerate(data):
    key = x[0]
    print(key)
    feature_file = feature_path + key + '.ascii'
    mat = np.loadtxt(feature_file)
    sentNum = sentNum + 1
    if key in features:
        print(key, " already have")
        already += 1
    features[key] = mat
    # generate stroke_mask

    # penup_index = np.where(mat[:,-1] == 1)[0] # 0 denote pen down, 1 denote pen up
    # strokeNum = len(penup_index)
    # stroke_mat = np.zeros([strokeNum, mat.shape[0]], dtype='float32')
    # for i in range(0,strokeNum):
    # Mask
    # if i == 0:
    #    stroke_mat[i,0:(penup_index[i]+1)] = 1
    # else:
    # stroke_mat[i,(penup_index[i-1]+1):(penup_index[i]+1)] = 1
    # Index
    # stroke_mat[i,penup_index[i]] = 1
    # normMask
    # if i == 0:
    #    stroke_mat[i,0:(penup_index[i]+1)] = 1 / (penup_index[i]+1)
    # else:
    #    stroke_mat[i,(penup_index[i-1]+1):(penup_index[i]+1)] = 1 / ((penup_index[i]+1) - (penup_index[i-1]+1))
    if sentNum // 500 == sentNum * 1.0 / 500:
        print('process sentences ' + str(sentNum))

print('load ascii file done. sentence number ' + str(sentNum), "reduplicate " + str(already))
pkl.dump(features, oupFp_feature)
print('save file done')
oupFp_feature.close()
