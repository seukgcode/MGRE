import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
import numpy as np
import math
from torch.nn import init
from torch.nn.utils import rnn
from pytorch_transformers import *
from torch.nn.utils.rnn import pad_sequence


class MRGE_bert(nn.Module):
    def __init__(self, config):
        super(MRGE_bert, self).__init__()
        self.config = config
        self.use_entity_type = True
        self.use_coreference = True
        self.use_distance = True

        word_vec_size = config.data_word_vec.shape[0]
        self.word_emb = nn.Embedding(word_vec_size, config.data_word_vec.shape[1])
        self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))
        self.word_emb.weight.requires_grad = False

        self.hidden_size = 128
        word_size = config.data_word_vec.shape[1]

        if self.use_entity_type:
            self.ner_emb = nn.Embedding(7, config.entity_type_size, padding_idx=0)

        if self.use_coreference:
            self.entity_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)


        self.sent_cnn = nn.Conv2d(1, 300, kernel_size=(3,3), padding=1)
        self.sent_W = nn.Linear(word_size+config.coref_size+config.entity_type_size, 300, bias=False)
        self.sent_v = nn.Linear(300, 1, bias=False)
        self.sent_att = nn.Softmax(dim=1)
        self.linear_doc = nn.Linear(word_size+config.coref_size+config.entity_type_size, config.relation_num)

        self.tree_RNN = DepST_RNN(word_size+config.coref_size+config.entity_type_size, config.dep_size, config.dep_rel_num)
        self.tree_LSTM = EncoderLSTM(word_size+config.coref_size+config.entity_type_size+config.dep_size, self.hidden_size, 1, True, False, 1 - config.keep_prob, False)

        self.linear_ent = nn.Linear(self.hidden_size, config.relation_num)


    def forward(self, context_idxs, pos, context_ner, context_char_idxs, context_lens, h_mapping, t_mapping,
                relation_mask, dis_h_2_t, dis_t_2_h, sent_idxs, sent_lengths, reverse_sent_idxs, context_masks, context_starts, context_tree, context_SDP, context_SDP_len):

        context_output = self.word_emb(context_idxs)

        if self.use_coreference:
            context_output = torch.cat([context_output, self.entity_embed(pos)], dim=-1)

        if self.use_entity_type:
            context_output = torch.cat([context_output, self.ner_emb(context_ner)], dim=-1)


        sent = self.sent_cnn(context_output.unsqueeze(1))
        sent,_ = torch.max(sent,2)
        sent = torch.tanh(sent)

        att_sent = torch.tanh(self.sent_W(sent))
        att_sent = self.sent_v(att_sent)
        att_sent = self.sent_att(att_sent.squeeze(2))

        doc = torch.sum(torch.mul(sent.squeeze(2), att_sent.unsqueeze(2)), dim=1)
        doc = self.linear_doc(doc)

        ent = self.tree_RNN(context_output, context_tree)

        ent_head_output = torch.zeros(context_SDP.size(0), context_SDP.size(1), self.hidden_size).cuda()#(batch_size,pre_num,128)
        ent_tail_output = torch.zeros(context_SDP.size(0), context_SDP.size(1), self.hidden_size).cuda()


        for batch in range(ent.size(0)):
            len = int(sum(context_SDP_len[batch, :, 0] != 0))

            ent_head = torch.zeros(len, self.config.SDP_deep, ent.size(2)).cuda()
            ent_tail = torch.zeros(len, self.config.SDP_deep, ent.size(2)).cuda()

            for pre_num in range(len):
                ent_head[pre_num] = torch.index_select(ent[batch], 0, context_SDP[batch, pre_num, 0])
                ent_tail[pre_num] = torch.index_select(ent[batch], 0, context_SDP[batch, pre_num, 1])

                ent_head_o = self.tree_LSTM(ent_head[pre_num:pre_num+1, :context_SDP_len[batch, pre_num:pre_num+1, 0]])
                ent_tail_o = self.tree_LSTM(ent_tail[pre_num:pre_num+1, :context_SDP_len[batch, pre_num:pre_num+1, 1]])
                print(ent_head_o.size())

                ent_head_output[batch, pre_num] = ent_head_o[0, -1]
                ent_tail_output[batch, pre_num] = ent_tail_o[0, -1]

        ent = ent_head_output - ent_tail_output
        ent = self.linear_ent(ent)

        predict_re = 0.5 * ent + (1-0.5) * doc.unsqueeze(1).expand(ent.size())

        return predict_re



class DepST_RNN(nn.Module):
    def __init__(self, node_size, dep_size, dep_rel_num):
        super().__init__()
        self.node_size = node_size
        self.dep_size = dep_size
        self.dep_rel_num = dep_rel_num

        self.dep_W = nn.ParameterList([nn.Parameter(torch.rand(self.dep_size, self.node_size+self.dep_size), requires_grad=True) for _ in range(self.dep_rel_num)])

    def forward(self, context, tree):
        child = torch.zeros(context.size(0), context.size(1), self.dep_size).cuda()

        batch_size = context.size(0)
        for batch in range(batch_size):
            for layer in range(len(tree[batch]), 0, -1):
                child_num = dict()
                for item in tree[batch][str(layer)]:
                    child[batch,item['head']] = torch.sum(torch.mul(self.dep_W[item['rel']], torch.cat([context[batch,item['tail']], child[batch,item['tail']]])), dim = 1)
                    if item['head'] not in child_num:
                        child_num[item['head']] = 0
                    child_num[item['head']] += 1
                for i in child_num.keys():
                    child[batch, int(i)] /= child_num[i]

        context = torch.cat([context, child], dim=-1)

        return context



class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.LSTM(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)

        self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.init_c = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])

        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last


    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous(), self.init_c[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()

        for i in range(self.nlayers):
            hidden, c = self.get_init(bsz, i)

            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)

            output, (hidden, c) = self.rnns[i](output, (hidden, c))


            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen: # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=-1)
        return outputs[-1]