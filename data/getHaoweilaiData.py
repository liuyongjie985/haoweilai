'''
从线上/ssd2/shared/data/ocr_data/online/kousuan/label.txt 整理latex
再从/ssd2/shared/data/ocr_data/online/kousuan/online_strokes 读取轨迹点

'''

import os
import json
import random
from formula_functions import deal_symbol_label
# from util import BoxDiagram, z_score_stroken_level, my_z_score, z_score, zscore_first_feature_extraction, \
#     scaleTrace, scaleTraceBoth, drawPictureByTrace, rve_duplicate
from formula_functions import z_score, rve_duplicate, zscore_first_feature_extraction

# ********************读取在线手写训练数据**********************
print("读取在线手写训练数据")
caption_dict = {}
file = open("/ssd2/shared/data/ocr_data/online/kousuan/label.txt")
while 1:
    lines = file.readlines(100000)
    if not lines:
        break
    for line in lines:
        temp_list = line.strip().split(r" ")
        if len(temp_list) >= 2:
            caption_dict[temp_list[0]] = temp_list[1]
file.close()
print("caption_dict", caption_dict)
# ****************************************************************
# ******************************读取字典**********************************
print("读取字典")
input_path1 = "/ssd2/exec/liuyongjie/haoweilai/formula_split_dict.txt"
symbol_dict = {}
file = open(input_path1)

while 1:
    lines = file.readlines(100000)
    if not lines:
        break
    for line in lines:
        temp_list = line.strip().split(r" ")
        if len(temp_list) >= 2:
            symbol_dict[temp_list[0]] = 1
file.close()
print("symbol_dict", symbol_dict)

trace_path = "/ssd2/shared/data/ocr_data/online/kousuan/online_strokes"
ascii_output_path1 = "/ssd2/exec/liuyongjie/haoweilai/train_ascii/"
ascii_output_path2 = "/ssd2/exec/liuyongjie/haoweilai/test_ascii/"

train_caption_list = []
test_caption_list = []
caption_dict = [[k, v] for k, v in caption_dict.items()]
count = 0
# =========将caption_dict中的未分词latext分开==========# ********************读取对应轨迹点**********************
while len(caption_dict) > 0:
    idx = random.randint(0, len(caption_dict) - 1)
    k = caption_dict[idx][0]
    v = caption_dict[idx][1]
    caption_dict.pop(idx)
    traceid2xy = []
    print(k, " doing")
    v = deal_symbol_label(v, symbol_dict)
    file_path = os.path.join(trace_path, k + ".json")
    traces = json.load(open(file_path))
    total_x = 0
    total_y = 0
    num_points = 0
    for trace in traces:
        temp_result = []
        for i, xy in enumerate(trace):
            x = float(xy["point"][0])
            y = float(xy["point"][1])
            total_x += x
            total_y += y
            if i != len(trace) - 1:
                temp_result.append([x, y, 1, 0])
            else:
                temp_result.append([x, y, 0, 1])
            num_points += 1
        traceid2xy.append(temp_result)
    traceid2xy, avg_x, avg_y = z_score(traceid2xy)
    traceid2xy, remove_point = rve_duplicate(traceid2xy, .017, -999999)
    # 转换为8维特征
    traceid2xy = zscore_first_feature_extraction(traceid2xy)
    if count < 1000:
        ascii_output_path = ascii_output_path2
    else:
        ascii_output_path = ascii_output_path1
    # ***************将feature输出到文件*******************
    o = open(os.path.join(ascii_output_path, k) + ".ascii", "w")
    for x in traceid2xy:
        for j, y in enumerate(x):
            o.write(str(round(y, 2)))
            if j != len(x) - 1:
                o.write(" ")
        o.write("\n")
    o.close()

    # ***********************************************************
    if count < 1000:
        test_caption_list.append([k, v])
    else:
        train_caption_list.append([k, v])
    count += 1

caption_file = "haoweilai_train_caption.txt"
caption_file = open(caption_file, "w")
json.dump(train_caption_list, caption_file)
caption_file.close()

caption_file = "haoweilai_test_caption.txt"
caption_file = open(caption_file, "w")
json.dump(test_caption_list, caption_file)
caption_file.close()
