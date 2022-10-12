# import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel
# from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput, BertAttention, BertSelfAttention
from bert_model import *

class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        # self.hidden_activation = activations.get(hidden_activation) keras中activation为linear时，返回原tensor,unchanged
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        """
            如果是条件Layer Norm，则cond不是None
        """
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs



class ConvolutionLayer(nn.Module):
    def __init__(self, input_size, channels, dilation, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU(),
        )

        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d) for d in dilation])


    def forward(self, x, y, dis_emb):
        B, M, _ = x.size()
        B, N, _ = y.size()

        fea_map = dis_emb.permute(0, 3, 1, 2).contiguous()
        fea_map = self.base(fea_map)

        fea_maps = []
        for conv in self.convs:
            x = conv(fea_map)
            x = F.gelu(x)
            fea_maps.append(x)
        fea_map = torch.cat(fea_maps, dim=1)
        fea_map = fea_map.permute(0, 2, 3, 1).contiguous()
        return fea_map


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1)

        return s


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class CoPredictor(nn.Module):
    def __init__(self, cls_num, hid_size, biaffine_size, channels, ffnn_hid_size, dropout=0):
        super().__init__()
        self.mlp_sub = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.mlp_obj = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.biaffine = Biaffine(n_in=biaffine_size, n_out=cls_num, bias_x=True, bias_y=True)
        self.mlp_rel = MLP(channels, ffnn_hid_size, dropout=dropout)
        self.linear = nn.Linear(ffnn_hid_size, cls_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, fea_maps):
        ent_sub = self.dropout(self.mlp_sub(x))
        ent_obj = self.dropout(self.mlp_obj(y))

        rel_outputs1 = self.biaffine(ent_sub, ent_obj)

        fea_maps = self.dropout(self.mlp_rel(fea_maps))
        rel_outputs2 = self.linear(fea_maps)
        return rel_outputs1 + rel_outputs2

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,  #B num_generated_triples H
        encoder_hidden_states,
        encoder_attention_mask
    ):
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0] #hidden_states.shape
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :] # B 1 1 H
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    encoder_hidden_shape, encoder_attention_mask.shape
                )
            )
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0 #1 1 0 0 -> 0 0 -1000 -1000
        
        cross_attention_outputs = self.crossattention(
            hidden_states=attention_output, encoder_hidden_states=encoder_hidden_states,  encoder_attention_mask=encoder_extended_attention_mask
        )
        attention_output = cross_attention_outputs[0] #B m H
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output) #B m H
        outputs = (layer_output,) + outputs
        return outputs


class Model(nn.Module):
    def __init__(self, config, bert_config):
        super(Model, self).__init__()
        self.use_bert_last_4_layers = config.use_bert_last_4_layers

        self.lstm_hid_size = config.lstm_hid_size
        self.conv_hid_size = config.conv_hid_size

        lstm_input_size = 0

        self.bert = AutoModel.from_pretrained(config.bert_name, output_hidden_states=True)
        lstm_input_size += config.bert_hid_size

        self.dis_embs = nn.Embedding(20, config.dist_emb_size)
        self.cls_embs = nn.Embedding(3, config.type_emb_size)

        self.encoder = nn.LSTM(lstm_input_size, config.lstm_hid_size // 2, num_layers=1, batch_first=True,
                               bidirectional=True)

        conv_input_size = config.lstm_hid_size + config.dist_emb_size + config.type_emb_size

        self.convLayer = ConvolutionLayer(conv_input_size, config.conv_hid_size, config.dilation, config.conv_dropout)
        self.dropout = nn.Dropout(config.emb_dropout)
        self.predictor_old = CoPredictor(config.old_label_num, config.lstm_hid_size, config.biaffine_size,
                                     config.conv_hid_size * len(config.dilation), config.ffnn_hid_size, config.out_dropout)

        self.cln = LayerNorm(config.lstm_hid_size, config.lstm_hid_size, conditional=True)
        
        self.Cr = nn.Linear(config.cr*2, config.label_num*config.label_num)

        self.Lr_e1_rev=nn.Linear(config.label_num*config.label_num, config.lstm_hid_size)
        self.rounds=config.rounds
        self.e_layer=DecoderLayer(bert_config)
        self.label_num = config.label_num
        
        torch.nn.init.orthogonal_(self.Cr.weight, gain=1)
        torch.nn.init.orthogonal_(self.Lr_e1_rev.weight, gain=1)
        self.Lr_e1=nn.Linear(config.lstm_hid_size, config.lstm_hid_size)
        torch.nn.init.orthogonal_(self.Lr_e1.weight, gain=1)
        self.Lr_e2=nn.Linear(bert_config.hidden_size,bert_config.hidden_size)
        torch.nn.init.orthogonal_(self.Lr_e2.weight, gain=1)

        self.Lr_e2_rev=nn.Linear(config.label_num*config.label_num, config.lstm_hid_size)

        torch.nn.init.orthogonal_(self.Lr_e2_rev.weight, gain=1)
        self.output_old=nn.Linear(config.old_label_num, config.cr*2)
        torch.nn.init.orthogonal_(self.output_old.weight, gain=1)
        self.output_new=nn.Linear(config.new_label_num, config.cr*2)
        torch.nn.init.orthogonal_(self.output_new.weight, gain=1)
        self.output_logit=nn.Linear(config.label_num*config.label_num, config.label_num)
        torch.nn.init.orthogonal_(self.output_logit.weight, gain=1)

        self.predictor_new = CoPredictor(config.new_label_num, config.lstm_hid_size, config.biaffine_size,
                                     config.conv_hid_size * len(config.dilation), config.ffnn_hid_size, config.out_dropout)

        self.old_label_num = config.old_label_num
        self.new_label_num = config.new_label_num

    def forward(self, word_inputs, bert_inputs, char_inputs, grid_mask2d, dist_inputs, pieces2word, word_mask2d):
        word_length = word_inputs.ne(0).sum(dim=-1)

        bert_embs = self.bert(input_ids=bert_inputs, attention_mask=bert_inputs.ne(0).float())
        if self.use_bert_last_4_layers:
            bert_embs = torch.stack(bert_embs[2][-4:], dim=-1).mean(-1)
        else:
            bert_embs = bert_embs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)
        word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, word_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_reps, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=word_length.max()) #batch_size, sent_len, 512
        word_reps_1 = self.Lr_e1(word_reps)
        word_reps_2 = self.Lr_e2(word_reps)
        for i in range(self.rounds):
            cln = self.cln(word_reps_1.unsqueeze(2), word_reps_2)

            dis_emb = self.dis_embs(dist_inputs)


            tril_mask = torch.tril(grid_mask2d.clone().long())
            cls_inputs = tril_mask + grid_mask2d.clone().long()
            cls_emb = self.cls_embs(cls_inputs)

            dis_emb = torch.cat([dis_emb, cls_emb, cln], dim=-1)

            dis_emb = torch.masked_fill(dis_emb, grid_mask2d.eq(0).unsqueeze(-1), 0.0)

            fea_maps = self.convLayer(word_reps_1, word_reps_2, dis_emb)

            fea_maps = torch.masked_fill(fea_maps, grid_mask2d.eq(0).unsqueeze(-1), 0.0) # batch_size, sent_len, snet_len, 240

            B, L = fea_maps.shape[0], fea_maps.shape[1]
            table_logist = self.Cr(fea_maps) # batch_size, sent_len, snet_len, 9
            if i!=self.rounds-1:
                table_e1 = table_logist.max(dim=2).values # batch_size, sent_len, 9
                table_e2 = table_logist.max(dim=1).values

                e1_ = self.Lr_e1_rev(table_e1) # batch_size, sent_len, 512
                e2_ = self.Lr_e2_rev(table_e2)

                word_reps_1 = word_reps_1 + self.e_layer(e1_, word_reps, word_mask2d)[0] # word_mask2d: B,L,L
                word_reps_2 = word_reps_2 + self.e_layer(e2_, word_reps, word_mask2d)[0]
        
        table_logist = self.output_logit(table_logist)
        old_table_logist = table_logist[:,:,:,:self.old_label_num]
        new_table_logist = table_logist[:,:,:,self.old_label_num:]
        old_fea_maps = self.output_old(old_table_logist)
        new_fea_maps = self.output_new(new_table_logist)
        outputs_old = self.predictor_old(word_reps, word_reps, old_fea_maps)
        outputs_new = self.predictor_new(word_reps, word_reps, new_fea_maps)
        
        return outputs_old, outputs_new
