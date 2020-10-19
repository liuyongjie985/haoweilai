import torch
import torch.nn as nn
from decoder import Tap_gru_cond_layer


def getReverse(a):
    # a = a.cpu().numpy()[::-1]
    # return torch.from_numpy(a.copy()).cuda()
    return a[range(len(a))[::-1]]


# Embedding
class My_Tap_Embedding(nn.Module):
    def __init__(self, params):
        super(My_Tap_Embedding, self).__init__()
        self.embedding = nn.Embedding(params['num_target'], params['word_dim'])

    def forward(self, params, y, is_train):
        if y.sum() < 0.:
            emb = torch.zeros(1, params['word_dim']).cuda()
        else:
            if is_train:  # only for training stage
                emb = self.embedding(y)
                emb_shifted = torch.zeros([emb.shape[0], emb.shape[1], params['word_dim']], dtype=torch.float32).cuda()
                emb_shifted[1:] = emb[:-1]
                emb = emb_shifted
            else:
                # emb = torch.zeros([y.shape[0], params["word_dim"]]).cuda()
                emb = self.embedding(y)
        return emb


class Encoder_Decoder(nn.Module):
    def __init__(self, params):
        super(Encoder_Decoder, self).__init__()
        self.tap_encoder_0_1 = nn.GRU(params["dim_feature"], params["tap_gru_Wx"][1], 2,
                                      bidirectional=True)  # ->(input_size,hidden_size,num_layers)
        self.tap_encoder_2 = nn.GRU(params["tap_gru_W"][0], params["tap_gru_Wx"][1], 1,
                                    bidirectional=True)  # ->(input_size,hidden_size,num_layers)

        self.tap_encoder_3 = nn.GRU(params["tap_gru_W"][0], params["tap_gru_Wx"][1], 1,
                                    bidirectional=True)  # ->(input_size,hidden_size,num_layers)

        self.ff_state_W = nn.Linear(params['ff_state_W'][0], params['ff_state_W'][1])
        self.tap_emb_model = My_Tap_Embedding(params)
        self.tap_decoder = Tap_gru_cond_layer(params)
        self.ff_logit_lstm = nn.Linear(params["ff_logit_lstm"][0], params["ff_logit_lstm"][1])
        self.ff_logit_prev = nn.Linear(params["ff_logit_prev"][0], params["ff_logit_prev"][1])
        self.ff_logit_ctx = nn.Linear(params["ff_logit_ctx"][0], params["ff_logit_ctx"][1])
        self.ff_logit = nn.Linear(params["ff_logit"][0], params["ff_logit"][1])

    def forward(self, params, tap_x, tap_x_mask, tap_y, tap_y_mask, maxlen, is_train=True):
        self.tap_encoder_0_1.flatten_parameters()
        self.tap_encoder_2.flatten_parameters()
        self.tap_encoder_3.flatten_parameters()

        # print("\tIn Train Model before: input size", tap_y.size())

        # 传入时 batch * seq_x * dim => seq_x * batch * dim
        tap_x = tap_x.permute([1, 0, 2])
        tap_x_mask = tap_x_mask.permute([1, 0])
        tap_y = tap_y.permute([1, 0])
        tap_y_mask = tap_y_mask.permute([1, 0])
        # print("\tIn Train Model after: tap_x", tap_x)
        # print("\tIn Train Model after: tap_y", tap_y)
        h, (hn) = self.tap_encoder_0_1(tap_x)
        first_layer = h
        h, (hn) = self.tap_encoder_2(h)
        h = h[0::2]
        tap_x_mask = tap_x_mask[0::2]
        h, (hn) = self.tap_encoder_3(h)
        h = h[0::2]
        tap_x_mask = tap_x_mask[0::2]

        tap_ctx = h

        # x_mask[:, :, None] = seq_x × batch_size × 1
        # (ctx * x_mask[:, :, None]).sum(0) = batch_size × 500
        # x_mask.sum(0)[:, None] = batch_size ×1
        tap_ctx_mean = (tap_ctx * tap_x_mask[:, :, None]).sum(0) / tap_x_mask.sum(0)[:, None]
        # init_state = batch_size × 256
        tap_init_state = torch.tanh(self.ff_state_W(tap_ctx_mean))
        # tparams['Wemb_dec'] = 111 × 256
        # y.flatten = y_length * batch_size
        # tparams['Wemb_dec'][y.flatten()] = y.flatten × 256

        # print("\tIn Train Model: input size", tap_emb.size())
        if is_train:
            tap_emb = self.tap_emb_model(params, tap_y, True)
            tap_proj = self.tap_decoder(params, tap_emb, mask=tap_y_mask, context=tap_ctx, context_mask=tap_x_mask,
                                        one_step=False, init_state=tap_init_state)
            tap_proj_h = tap_proj[0]
            tap_ctxs = tap_proj[1]
            logit_lstm = self.ff_logit_lstm(tap_proj_h)
            logit_prev = self.ff_logit_prev(tap_emb)
            logit_ctx = self.ff_logit_ctx(tap_ctxs)

            logit = logit_lstm + logit_prev + logit_ctx

            shape = logit.shape
            shape2 = int(shape[2] / 2)
            shape3 = 2
            logit = torch.reshape(logit, [shape[0], shape[1], shape2, shape3])
            logit = logit.max(3)[0]  # seq*batch*128
            last_logit = self.ff_logit(logit)
            return last_logit.permute([1, 0, 2])
        else:
            result_list = []
            # tap decoder batch_size * dim
            next_w = torch.zeros([tap_ctx.shape[1], params["word_dim"]], dtype=torch.float32).cuda()
            # batch_size * seq_x
            next_alpha_past = torch.zeros([tap_ctx.shape[1], tap_ctx.shape[0]]).cuda()  # start position
            # batch_size * 256
            next_state = tap_init_state

            for ii in range(maxlen):
                tap_proj = self.tap_decoder(params, next_w, context=tap_ctx, context_mask=tap_x_mask,
                                            one_step=True, init_state=next_state, alpha_past=next_alpha_past)
                next_state = tap_proj[0]
                # batch_size * dim
                tap_ctxs = tap_proj[1]
                next_alpha_past = tap_proj[3]
                # tap_next_state:batch_size * dim
                logit_lstm = self.ff_logit_lstm(next_state)
                logit_prev = self.ff_logit_prev(next_w)
                logit_ctx = self.ff_logit_ctx(tap_ctxs)

                logit = logit_lstm + logit_prev + logit_ctx

                shape = logit.shape
                # dim /2
                shape1 = int(shape[1] / 2)
                shape2 = 2
                # batch_size * dim / 2 * 2
                logit = torch.reshape(logit, [shape[0], shape1, shape2])
                # batch_size * dim / 2
                logit = logit.max(2)[0]  # batch*128
                # batch_size * target
                logit = self.ff_logit(logit)
                # batch_size * target
                next_probs = torch.softmax(logit, logit.ndim - 1)
                # batch_size *
                next_w = torch.argmax(next_probs, next_probs.ndim - 1)
                result_list.append(next_w)
                next_w = self.tap_emb_model(params, next_w, False)

            return result_list

    # decoding: encoder part

    def tap_f_init(self, params, tap_x):
        # print("\tIn Valid Model: input size", tap_x.size())
        self.tap_encoder_0_1.flatten_parameters()
        self.tap_encoder_2.flatten_parameters()
        self.tap_encoder_3.flatten_parameters()
        h, (hn) = self.tap_encoder_0_1(tap_x)

        h, (hn) = self.tap_encoder_2(h)
        h = h[0::2]
        h, (hn) = self.tap_encoder_3(h)
        h = h[0::2]
        tap_ctx = h
        tap_ctx_mean = tap_ctx.mean(0)
        tap_init_state = torch.tanh(self.ff_state_W(tap_ctx_mean))
        return tap_init_state, tap_ctx

    # decoding: decoder part
    def tap_f_next(self, params, y, tap_ctx, init_state, alpha_past):
        tap_emb = self.tap_emb_model(params, y, False)
        tap_proj = self.tap_decoder(params, tap_emb, context=tap_ctx,
                                    one_step=True,
                                    init_state=init_state, alpha_past=alpha_past)
        tap_next_state = tap_proj[0]
        tap_ctxs = tap_proj[1]
        next_alpha_past = tap_proj[3]
        logit_lstm = self.ff_logit_lstm(tap_next_state)
        logit_prev = self.ff_logit_prev(tap_emb)
        logit_ctx = self.ff_logit_ctx(tap_ctxs)

        logit = logit_lstm + logit_prev + logit_ctx
        shape = logit.shape
        shape1 = int(shape[1] / 2)
        shape2 = 2
        logit = torch.reshape(logit, [shape[0], shape1, shape2])
        logit = logit.max(2)[0]  # seq * batch * 128
        logit = self.ff_logit(logit)
        next_probs = torch.softmax(logit, logit.ndim - 1)
        return next_probs, tap_next_state, next_alpha_past
