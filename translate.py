import argparse
import numpy as np
import os
import re
import torch
from utils import load_dict, gen_sample, argmax_dataGenerator, argmax_dataIterator, beamSearch_dataGenerator
from encoder_decoder import Encoder_Decoder
import time

multi_gpu_flag = False


def main(model_path, dictionary_target, test_file, target_file, result_file, calculate_result, beam_k=5,
         batch_size=None):
    # model architecture
    params = {}
    scale = 1
    # TAP model ---encoder
    params["tap_gru_U"] = [int(250 * scale), int(500 * scale)]
    params["tap_gru_Ux"] = [int(250 * scale), int(250 * scale)]
    params["tap_gru_W0"] = [9, int(500 * scale)]
    params["tap_gru_Wx0"] = [9, int(250 * scale)]
    params["tap_gru_W"] = [int(500 * scale), int(500 * scale)]
    params["tap_gru_Wx"] = [int(500 * scale), int(250 * scale)]
    params["hidden_size"] = [int(250 * scale), int(250 * scale), int(250 * scale), int(250 * scale)]
    params["down_sample"] = [0, 0, 1, 1]
    params['dim_feature'] = 9
    params["ff_state_W"] = [int(500 * scale), int(256 * scale)]
    params['num_target'] = 689
    params['word_dim'] = int(256 * scale)
    params['tap_decoder_Wcx'] = [int(500 * scale), int(256 * scale)]
    params['tap_decoder_Wc_att'] = [int(500 * scale), int(500 * scale)]
    params['tap_decoder_Wx'] = [int(256 * scale), int(256 * scale)]
    params['tap_decoder_W'] = [int(256 * scale), int(512 * scale)]
    params['tap_decoder_Wyg'] = [int(256 * scale), int(500 * scale)]
    params['tap_decoder_U'] = [int(256 * scale), int(512 * scale)]
    params['tap_decoder_Ux'] = [int(256 * scale), int(256 * scale)]
    params['tap_decoder_Whg'] = [int(256 * scale), int(500 * scale)]
    params['tap_decoder_Umg'] = [int(256 * scale), int(500 * scale)]
    params['tap_decoder_W_comb_att'] = [int(256 * scale), int(500 * scale)]
    params['tap_decoder_conv_Uf'] = [int(256 * scale), int(500 * scale)]
    params['tap_decoder_U_att'] = [int(500 * scale), 1]
    params['tap_decoder_W_m_att'] = [int(500 * scale), int(500 * scale)]
    params['tap_decoder_U_when_att'] = [int(500 * scale), 1]
    params['tap_decoder_U_nl'] = [int(256 * scale), int(512 * scale)]
    params['tap_decoder_Wc'] = [int(500 * scale), int(512 * scale)]
    params['tap_decoder_Ux_nl'] = [int(256 * scale), int(256 * scale)]

    params["ff_logit_lstm"] = [int(256 * scale), int(256)]
    params["ff_logit_prev"] = [int(256 * scale), int(256)]
    params["ff_logit_ctx"] = [int(500 * scale), int(256)]
    params["ff_logit"] = [128, params['num_target']]
    params['gamma'] = 0.1

    # load model
    tap_model = Encoder_Decoder(params)
    tap_model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    tap_model.cuda()

    # load dictionary
    worddicts = load_dict(dictionary_target)
    worddicts_r = [None] * len(worddicts)
    for kk, vv in worddicts.items():
        worddicts_r[vv] = kk

    # testing
    tap_model.eval()
    total_time = 0
    total_num = 0
    with torch.no_grad():
        fpp_sample = open(result_file, 'w')
        test_count_idx = 0
        print('Decoding ... ')
        if k < 0:  # argmax
            print("argmax")
            tap_valid, _, tap_valid_uid_list = argmax_dataGenerator(test_file, target_file,
                                                                    worddicts,
                                                                    batch_size=batch_size,
                                                                    maxlen=32, match_batch=False)
            for out_index, (tap_x, tap_y) in enumerate(tap_valid):
                tap_x, tap_x_mask, tap_y, tap_y_mask = argmax_dataIterator(
                    params, tap_x, tap_y, maxlen=32)

                assert not np.any(np.isnan(tap_x))
                assert not np.any(np.isnan(tap_x_mask))
                assert not np.any(np.isnan(tap_y))
                assert not np.any(np.isnan(tap_y_mask))

                tap_x = torch.from_numpy(tap_x).cuda()
                tap_x_mask = torch.from_numpy(tap_x_mask).cuda()
                # batch_size × seq_y
                tap_y = torch.from_numpy(tap_y).cuda()
                tap_y_mask = torch.from_numpy(tap_y_mask).cuda()

                # print(tap_xx_pad.shape)
                start = time.time()
                # seq_y * batch_size
                result = tap_model(params, tap_x.permute([1, 0, 2]),
                                   tap_x_mask.permute([1, 0]),
                                   tap_y.permute([1, 0]),
                                   tap_y_mask.permute([1, 0]), 32, False)
                # write decoding results
                end = time.time()
                print("用时", str(end - start))
                total_time += end - start
                result = [x.cpu().numpy() for x in result]
                # batch_size*seq_y
                result = np.transpose(result)
                total_num += len(result)

                for valid_index, vv in enumerate(result):
                    fpp_sample.write(
                        tap_valid_uid_list[out_index * batch_size + valid_index])
                    for c in vv:
                        if c == 0:
                            break
                        fpp_sample.write(" " + worddicts_r[c])
                    fpp_sample.write('\n')
                    fpp_sample.flush()
        else:
            print("beamsearch")
            # load data
            tap_valid, tap_valid_uid_list = beamSearch_dataGenerator(test_file, target_file,
                                                                     worddicts,
                                                                     batch_size=1, maxlen=32)
            for x, y in tap_valid:
                for tap_xx in x:
                    print('%d : %s' % (test_count_idx + 1, tap_valid_uid_list[test_count_idx]))
                    ss = []
                    if not (tap_xx.shape[0] == 0 or tap_xx.shape[1] == 0):
                        tap_xx_pad = np.zeros((tap_xx.shape[0] + 1, tap_xx.shape[1]), dtype='float32')
                        tap_xx_pad[:tap_xx.shape[0], :] = tap_xx
                        tap_xx_pad = torch.from_numpy(tap_xx_pad).cuda()
                        stochastic = False
                        start = time.time()
                        sample, score = gen_sample(tap_model, tap_xx_pad[:, None, :], params, multi_gpu_flag, k=10,
                                                   maxlen=32,
                                                   stochastic=stochastic,
                                                   argmax=False)

                        score = score / np.array([len(s) for s in sample])
                        ss = sample[score.argmin()]
                        end = time.time()
                        print("用时", str(end - start))
                        total_time += end - start
                        total_num += 1
                    # write decoding results
                    fpp_sample.write(tap_valid_uid_list[test_count_idx])
                    test_count_idx = test_count_idx + 1
                    # symbols (without <eos>)
                    for vv in ss:
                        if vv == 0:  # <eos>
                            break
                        fpp_sample.write(' ' + worddicts_r[vv])
                    fpp_sample.write('\n')
    fpp_sample.close()
    print('test set decode done')
    os.system('python compute-wer.py ' + result_file + ' ' + target_file + ' ' + calculate_result)
    fpp = open(calculate_result)
    stuff = fpp.readlines()
    fpp.close()
    m = re.search('WER (.*)\n', stuff[0])
    test_per = 100. * float(m.group(1))
    m = re.search('ExpRate (.*)\n', stuff[1])
    test_sacc = 100. * float(m.group(1))
    print('Valid WER: %.2f%%, ExpRate: %.2f%%' % (test_per, test_sacc))
    print("平均", str(total_time / total_num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', default=-1, type=int, help='beam search size k, -1 indicate using argmax')
    parser.add_argument('--model_path', required=True, type=str, help='the path of model')
    parser.add_argument('--dictionary_target', required=True, type=str, help='the file of target dict')
    parser.add_argument('--test_file', required=True, type=str, help='the file of test')
    parser.add_argument('--target_file', required=True, type=str, help='the target file of test')
    parser.add_argument('--result_file', required=True, type=str, help='the result file of test')
    parser.add_argument('--calculate_result', required=True, type=str,
                        help='the compare result of test result and target')
    parser.add_argument('--batch_size', required=True, type=int,
                        help='valid batch size')

    args = parser.parse_args()

    k = args.k
    model_path = args.model_path
    dictionary_target = args.dictionary_target
    test_file = args.test_file
    target_file = args.target_file
    result_file = args.result_file
    calculate_result = args.calculate_result
    batch_size = args.batch_size
    main(model_path, dictionary_target, test_file, target_file, result_file, calculate_result, k, batch_size)
