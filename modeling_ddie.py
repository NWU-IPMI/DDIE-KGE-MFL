# -*- coding: utf-8 -*-
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, BertModel, BertConfig
from MultiFocalLoss import MultiFocalLoss

# GELU激活函数
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True, activation="relu"):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        activations = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'leakyrelu': nn.LeakyReLU(), 'prelu': nn.PReLU(),
                       'relu6': nn.ReLU6, 'rrelu': nn.RReLU(), 'selu': nn.SELU(), 'celu': nn.CELU(), 'gelu': GELU(),
                       'tanh': nn.Tanh()}
        self.activation = activations[activation]

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.activation(x)
        return self.linear(x)


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, args, config, tokenizer):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.tokenizer = tokenizer
        self.args = args
        self.dropout = nn.Dropout(args.dropout_prob)
        activations = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'leakyrelu': nn.LeakyReLU(), 'prelu': nn.PReLU(),
                       'relu6': nn.ReLU6, 'rrelu': nn.RReLU(), 'selu': nn.SELU(), 'celu': nn.CELU(), 'gelu': GELU()}
        self.activation = activations[args.activation]

        if args.use_cnn:
            self.conv_list = nn.ModuleList(
                [nn.Conv1d(config.hidden_size + 2 * args.pos_emb_dim, config.hidden_size, w, padding=(w - 1) // 2) for w
                 in args.conv_window_size])
            self.pos_emb = nn.Embedding(2 * args.max_seq_length, args.pos_emb_dim, padding_idx=0)
        self.semantic_pos_emb = nn.Embedding(2 * args.max_seq_length, args.pos_emb_dim, padding_idx=0)

        if args.middle_layer_size == 0:
            self.classifier = nn.Linear(len(args.conv_window_size) * 768, config.num_labels)
        else:
            self.middle_classifier = nn.Linear(len(args.conv_window_size) * 768,
                                               args.middle_layer_size)
            self.classifier = nn.Linear(args.middle_layer_size, config.num_labels)
        self.init_weights()

        if args.use_cnn:
            self.pos_emb.weight.data.uniform_(-1e-3, 1e-3)
            self.semantic_pos_emb.weight.data.uniform_(-1e-3, 1e-3)

        self.config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=self.num_labels)
        self.config.output_hidden_states = True
        self.bert = BertModel.from_pretrained(args.model_name_or_path, config=self.config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.bert_layer_weights = nn.Parameter(torch.rand(13, 1))
        self.use_cnn = args.use_cnn
        self.middle_layer_size = args.middle_layer_size

        self.MLP1 = FCLayer(
            768+200*2,
            768,
            0.3,
            use_activation=True,
        )

        self.linear = nn.Linear(2000, 200)
        self.classifier2 = nn.Linear(200, 5)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, relative_dist1=None,
                relative_dist2=None, all_dep_mask0=None, all_dep_mask1=None, all_dep_mask2=None, all_dep_mask3=None,
                labels=None, drug_a_ids=None, drug_b_ids=None):
        dep_mask = all_dep_mask1
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # position feature
        if self.use_cnn:
            relative_dist1 *= attention_mask
            relative_dist2 *= attention_mask
            pos_embs1 = self.pos_emb(relative_dist1)
            pos_embs2 = self.pos_emb(relative_dist2)
            conv_input = torch.cat((outputs[0], pos_embs1, pos_embs2), 2)
            conv_outputs = []
            for c in self.conv_list:
                conv_output1 = self.activation(c(conv_input.transpose(1, 2)))
                conv_output, _ = torch.max(conv_output1, -1)
                conv_outputs.append(conv_output)
            position_feature = torch.cat(conv_outputs, 1)

        # key_path feature
        batch_size = input_ids.shape[0]
        all_hidden_states = outputs[2]
        ht_cls = torch.cat(all_hidden_states)[:, :1, :].view(13, batch_size, 1, 768)
        atten = torch.sum(ht_cls * self.bert_layer_weights.view(13, 1, 1, 1), dim=[1, 3])
        atten = F.softmax(atten.view(-1), dim=0)
        attention_feature = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])

        # semantic feature
        pos_semantic = self.semantic_pos_emb(dep_mask)
        conv_input_semantic = torch.cat((outputs[0], pos_semantic), 2)
        conv_output_semantic = []
        c_conv = nn.Conv1d(self.config.hidden_size + self.args.pos_emb_dim, self.config.hidden_size, 3, padding=1)
        c_conv.to(self.args.device)
        conv_output_t = self.activation(c_conv(conv_input_semantic.transpose(1, 2)))
        conv_output_t1, _ = torch.max(conv_output_t, -1)
        conv_output_semantic.append(conv_output_t1)
        key_path_feature = torch.cat(conv_output_semantic, 1)

        # synthetical feature
        pooled_output = (position_feature + attention_feature + key_path_feature) / 3

        # knowledge graph embedding
        drug_a_ids = torch.tensor([d for d in drug_a_ids], dtype=torch.float).to(self.args.device)
        drug_b_ids = torch.tensor([d for d in drug_b_ids], dtype=torch.float).to(self.args.device)
        drug_a_id = self.linear(drug_a_ids)
        drug_b_id = self.linear(drug_b_ids)
        # final feature
        pooled_output = torch.cat((pooled_output, drug_a_id, drug_b_id), 1)
        pooled_output = self.MLP1(pooled_output)
        if self.middle_layer_size == 0:
            logits = self.classifier(pooled_output)
        else:
            middle_output = self.activation(self.middle_classifier(pooled_output))
            logits = self.classifier(middle_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # MultiFocal loss
                loss_fct = MultiFocalLoss(self.num_labels, [0.8, 0.07, 0.08, 0.04, 0.01])
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                # kge loss
                labels_kg = drug_b_id - drug_a_id  # 分别对应每一条数据的label
                logits2 = self.classifier2(labels_kg)
                loss2 = loss_fct(logits2.view(-1, self.num_labels), labels.view(-1))

                # KGE-MFL loss
                loss = 0.6 * loss + 0.4 * loss2

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def zero_init_params(self):
        self.update_cnt = 0
        for x in self.parameters():
            x.data *= 0

    def accumulate_params(self, model):
        self.update_cnt += 1
        for x, y in zip(self.parameters(), model.parameters()):
            x.data += y.data

    def average_params(self):
        for x in self.parameters():
            x.data /= self.update_cnt

    def restore_params(self):
        for x in self.parameters():
            x.data *= self.update_cnt
