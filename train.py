import time
import os
import re
import numpy as np
import random
import torch
from torch import optim, nn
from formula_functions import load_dict, gen_sample, weight_init, dataGenerator, argmax_dataIterator
from encoder_decoder import Encoder_Decoder
import argparse

print("GPU可用否")
print(torch.cuda.is_available())

parser = argparse.ArgumentParser()

parser.add_argument('--reload', action="store_true", help="if use fine tuning")
parser.add_argument('--multi_gpu_flag', action="store_true", help='if use multi gpu')
parser.add_argument('--dictionary', required=True, type=str, help='the path of target dictionary')
parser.add_argument('--valid_result', required=True, type=str, help='the result of valid')
parser.add_argument('--train_dataset1', required=True, type=str, help='the file of train fea')
parser.add_argument('--train_dataset2', required=True, type=str, help='the file of train target')
parser.add_argument('--valid_dataset1', required=True, type=str, help='the file of valid fea')
parser.add_argument('--valid_dataset2', required=True, type=str, help='the file of valid target')
parser.add_argument('--model_path', required=True, type=str, help='the path of model')
parser.add_argument('--train_batch_size', required=True, type=int,
                    help='train batch size')
parser.add_argument('--valid_batch_size', required=True, type=int,
                    help='valid batch size')
parser.add_argument('--max_len', required=True, type=int, help='the max length of y')
parser.add_argument('--lr', default=1, type=float, help='lr')
parser.add_argument('--scale', default=256, type=int, help='the base dim of model')
parser.add_argument('--calculate_result', required=True, type=str,
                    help='the compare result of test result and target')
parser.add_argument('--method', required=True, type=str,
                    help='beamsearch or argmax')

args = parser.parse_args()

RELOAD = args.reload

# whether use multi-GPUs
multi_gpu_flag = args.multi_gpu_flag
print("multi_gpu_flag", multi_gpu_flag)

# load configurations

dictionary = [args.dictionary]
valid_result = [args.valid_result]
tap_datasets = [args.train_dataset1, args.train_dataset2]
tap_valid_datasets = [args.valid_dataset1, args.valid_dataset2]
saveto = args.model_path

# training settings
if multi_gpu_flag:
    batch_size = args.train_batch_size
    valid_batch_size = args.valid_batch_size
else:
    batch_size = args.train_batch_size
    valid_batch_size = args.valid_batch_size

maxlen = args.max_len
start_lr = args.lr
scale = args.scale / 256

max_epochs = 5000
my_eps = 1e-8
decay_c = 0.0005
clip_c = -1000.

# early stop
estop = False
halfLrFlag = 0
bad_counter = 0
patience = 15
finish_after = 10000000

# load dictionary
worddicts = load_dict(dictionary[0])
worddicts_r = [None] * len(worddicts)
for kk, vv in worddicts.items():
    worddicts_r[vv] = kk

# model architecture
params = {}
params['num_target'] = len(worddicts)
params["tap_gru_U"] = [int(256 * scale), int(512 * scale)]
params["tap_gru_Ux"] = [int(256 * scale), int(256 * scale)]
params["tap_gru_W0"] = [9, int(512 * scale)]
params["tap_gru_Wx0"] = [9, int(256 * scale)]
params["tap_gru_W"] = [int(512 * scale), int(512 * scale)]
params["tap_gru_Wx"] = [int(512 * scale), int(256 * scale)]
params["hidden_size"] = [int(256 * scale), int(256 * scale), int(256 * scale), int(256 * scale)]
params["down_sample"] = [0, 0, 1, 1]
params['dim_feature'] = 9
params["ff_state_W"] = [int(512 * scale), int(256 * scale)]
params['word_dim'] = int(256 * scale)
params['tap_decoder_Wcx'] = [int(512 * scale), int(256 * scale)]
params['tap_decoder_Wc_att'] = [int(512 * scale), int(512 * scale)]
params['tap_decoder_Wx'] = [int(256 * scale), int(256 * scale)]
params['tap_decoder_W'] = [int(256 * scale), int(512 * scale)]
params['tap_decoder_Wyg'] = [int(256 * scale), int(512 * scale)]
params['tap_decoder_U'] = [int(256 * scale), int(512 * scale)]
params['tap_decoder_Ux'] = [int(256 * scale), int(256 * scale)]
params['tap_decoder_Whg'] = [int(256 * scale), int(512 * scale)]
params['tap_decoder_Umg'] = [int(256 * scale), int(512 * scale)]
params['tap_decoder_W_comb_att'] = [int(256 * scale), int(512 * scale)]
params['tap_decoder_conv_Uf'] = [int(256 * scale), int(512 * scale)]
params['tap_decoder_U_att'] = [int(512 * scale), 1]
params['tap_decoder_W_m_att'] = [int(512 * scale), int(512 * scale)]
params['tap_decoder_U_when_att'] = [int(512 * scale), 1]
params['tap_decoder_U_nl'] = [int(256 * scale), int(512 * scale)]
params['tap_decoder_Wc'] = [int(512 * scale), int(512 * scale)]
params['tap_decoder_Ux_nl'] = [int(256 * scale), int(256 * scale)]

params["ff_logit_lstm"] = [int(256 * scale), int(256 * scale)]
params["ff_logit_prev"] = [int(256 * scale), int(256 * scale)]
params["ff_logit_ctx"] = [int(512 * scale), int(256 * scale)]
params["ff_logit"] = [int(128 * scale), params['num_target']]
params['gamma'] = 0.1
beamsearch_k = None
# ******************************************************************************************************************
if args.method == "argmax":
    beamsearch_k = -1
    tap_train, tap_num_batch, _ = dataGenerator(tap_datasets[0], tap_datasets[1],
                                                worddicts,
                                                batch_size=batch_size, maxlen=maxlen)

    tap_valid, _, tap_valid_uid_list = dataGenerator(tap_valid_datasets[0], tap_valid_datasets[1],
                                                     worddicts,
                                                     batch_size=valid_batch_size,
                                                     maxlen=maxlen, match_batch=False)
elif args.method == "beamsearch":
    beamsearch_k = 10
    tap_train, _, _ = dataGenerator(tap_datasets[0], tap_datasets[1],
                                    worddicts,
                                    batch_size=batch_size, maxlen=maxlen)
    tap_valid, _, tap_valid_uid_list = dataGenerator(tap_valid_datasets[0], tap_valid_datasets[1],
                                                     worddicts,
                                                     batch_size=valid_batch_size, maxlen=maxlen, match_batch=False)
else:
    print("argmax or beamsearch not selected")

# ******************************************************************************************************************

# display
uidx = 0  # count batch
loss_s = 0.  # count loss
ud_s = 0  # time for training an epoch
validFreq = -1
saveFreq = -1
sampleFreq = -1
dispFreq = 100
if validFreq == -1:
    validFreq = len(tap_train)
if saveFreq == -1:
    saveFreq = len(tap_train)
if sampleFreq == -1:
    sampleFreq = len(tap_train)

# initialize model
TAP_model = Encoder_Decoder(params)
if RELOAD == True:
    TAP_model.load_state_dict(
        torch.load(saveto, map_location=lambda storage, loc: storage))
else:
    TAP_model.apply(weight_init)

if multi_gpu_flag:
    TAP_model = nn.DataParallel(TAP_model.cuda())
else:
    TAP_model.cuda()

# print model's parameters
model_params = TAP_model.named_parameters()
for k, v in model_params:
    print(k)

# loss function
criterion = torch.nn.CrossEntropyLoss(reduce=False)
# optimizer
optimizer = optim.Adadelta(TAP_model.parameters(), lr=start_lr, eps=my_eps, weight_decay=decay_c)

print('Optimization')
for param_group in optimizer.param_groups:
    print(param_group['eps'])

# statistics
history_errs = []
myeidx = 0
for eidx in range(max_epochs):
    n_samples = 0
    ud_epoch = time.time()
    random.shuffle(tap_train)

    for tap_x, tap_y in tap_train:
        TAP_model.train()
        ud_start = time.time()
        n_samples += len(tap_x)
        uidx += 1

        tap_x, tap_x_mask, tap_y, tap_y_mask = argmax_dataIterator(
            params, tap_x, tap_y, maxlen=maxlen)

        assert not np.any(np.isnan(tap_x))
        assert not np.any(np.isnan(tap_x_mask))
        assert not np.any(np.isnan(tap_y))
        assert not np.any(np.isnan(tap_y_mask))

        tap_x = torch.from_numpy(tap_x).cuda()
        tap_x_mask = torch.from_numpy(tap_x_mask).cuda()
        # batch_size × seq_y
        tap_y = torch.from_numpy(tap_y).cuda()
        tap_y_mask = torch.from_numpy(tap_y_mask).cuda()

        # forward
        tap_ctx = TAP_model(params, tap_x.permute([1, 0, 2]),
                            tap_x_mask.permute([1, 0]),
                            tap_y.permute([1, 0]),
                            tap_y_mask.permute([1, 0]), None, True)

        # print("\tOUT Train Model: first_layer size", tap_ctx.shape)
        # print("\tOUT Train Model: first_layer", tap_ctx)

        tap_ctx = tap_ctx.permute([1, 0, 2])
        tap_ctx = torch.reshape(tap_ctx, [-1, tap_ctx.shape[2]])

        # loss = criterion(scores, y.view(-1))
        loss = criterion(tap_ctx, torch.reshape(tap_y, [-1]))
        # loss:seq_y × batch_size
        loss = torch.reshape(loss, [tap_y.shape[0], tap_y.shape[1]])

        # loss:1 × batch_size
        # loss = ((loss * tap_y_mask).sum(0) + params['gamma'] * cost_alphas.sum(0).sum(1)) / tap_y_mask.sum(0)
        loss = ((loss * tap_y_mask).sum(0))
        # loss = (loss * tap_y_mask).sum(0)
        # loss = (loss * tap_y_mask).sum(0) / tap_y_mask.sum(0)
        loss = loss.mean()
        # print("当前损失", loss)
        loss_s += loss.item()
        # print("当前预测")
        # for ii, it in enumerate(tap_ctx):
        #    print(str(ii), it)

        # print("当前目标", tap_y)

        # backward
        optimizer.zero_grad()
        loss.backward()
        if clip_c > 0.:
            torch.nn.utils.clip_grad_norm_(TAP_model.parameters(), clip_c)
        # max_grad = -1
        # min_grad = 999999
        # for x in TAP_model.parameters():
        #     temp_gard = torch.reshape(x.grad.data, [-1]).sum(0)
        #     if temp_gard > max_grad:
        #         max_grad = temp_gard
        #     if temp_gard < min_grad:
        #         min_grad = temp_gard
        # print("当前最大梯度", max_grad)
        # print("当前最小梯度", min_grad)
        # update
        optimizer.step()
        ud = time.time() - ud_start
        # print(str(ud / 60))
        ud_s += ud

        # display
        if np.mod(uidx, dispFreq) == 0:
            ud_s /= 60.
            loss_s /= dispFreq
            print('Epoch ', eidx, 'Update ', uidx, 'Cost ', loss_s, 'UD ', ud_s, 'lrate ', args.lr, 'eps', my_eps,
                  'bad_counter', bad_counter)
            ud_s = 0
            loss_s = 0.

        # validation
        valid_stop = False
        if np.mod(uidx, sampleFreq) == 0:
            total_time = 0
            total_num = 0
            TAP_model.eval()
            with torch.no_grad():
                fpp_sample = open(valid_result[0], 'w')
                test_count_idx = 0
                print('Decoding ... ')
                if beamsearch_k < 0:  # argmax
                    print("argmax")
                    for out_index, (tap_x, tap_y) in enumerate(tap_valid):
                        tap_x, tap_x_mask, tap_y, tap_y_mask = argmax_dataIterator(
                            params, tap_x, tap_y, maxlen=maxlen)
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
                        result = TAP_model(params, tap_x.permute([1, 0, 2]),
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
                    if valid_stop:
                        break
                else:
                    print("beamsearch")
                    # load data

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
                                sample, score = gen_sample(TAP_model, tap_xx_pad[:, None, :], params, multi_gpu_flag,
                                                           k=beamsearch_k,
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
            print('valid set decode done')
            ud_epoch = (time.time() - ud_epoch) / 60.
            print('epoch cost time ... ', ud_epoch)
            print("平均", str(total_time / total_num))

            # calculate wer and expRate
        if np.mod(uidx, validFreq) == 0 and valid_stop == False:
            os.system('python compute-wer.py ' + valid_result[0] + ' ' + tap_valid_datasets[
                1] + ' ' + valid_result[0])

            fpp = open(valid_result[0])
            stuff = fpp.readlines()
            fpp.close()
            m = re.search('WER (.*)\n', stuff[0])
            valid_err = 100. * float(m.group(1))
            m = re.search('ExpRate (.*)\n', stuff[1])
            valid_sacc = 100. * float(m.group(1))
            history_errs.append(valid_err)

            # the first time validation or better model
            if uidx // validFreq == 0 or valid_err <= np.array(history_errs).min():
                bad_counter = 0
                print('Saving model params ... ')
                if multi_gpu_flag:
                    torch.save(TAP_model.module.state_dict(), saveto)
                else:
                    torch.save(TAP_model.state_dict(), saveto)

            # worse model
            if uidx / validFreq != 0 and valid_err > np.array(history_errs).min():
                bad_counter += 1
                if bad_counter > patience:
                    if halfLrFlag == 2:
                        print('Early Stop!')
                        estop = True
                        break
                    else:
                        print('Lr decay and retrain!')
                        bad_counter = 0
                        my_eps = my_eps / 10.
                        for param_group in optimizer.param_groups:
                            param_group['eps'] = my_eps
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = start_lr
                        myeidx = 0
                        halfLrFlag += 1
            print('Valid WER: %.2f%%, ExpRate: %.2f%%' % (valid_err, valid_sacc))

        # finish after these many updates
        if uidx >= finish_after:
            print('Finishing after %d iterations!' % uidx)
            estop = True
            break

    print('Seen %d samples' % n_samples)

    # early stop
    if estop:
        break
