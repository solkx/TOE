import argparse
import random
import json
import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import torch.optim as optim
import transformers
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
import data_loader
import utils
from model import Model
import os
import torch.nn.functional as F

def compute_kl_loss(p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = LabelSmoothSoftmaxCEV1(lb_smooth=config.smoothing)

        if config.use_bert:
            bert_params = set(self.model.bert.parameters())
            other_params = list(set(self.model.parameters()) - bert_params)
            no_decay = ['bias', 'LayerNorm.weight']
            params = [
                {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                 'lr': config.bert_learning_rate,
                 'weight_decay': config.bert_weight_decay},
                {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
                 'lr': config.bert_learning_rate,
                 'weight_decay': 0.0},
                {'params': other_params,
                 'lr': config.learning_rate,
                 'weight_decay': config.weight_decay},
            ]
        else:
            params = model.parameters()

        self.optimizer = optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.7)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=config.warm_factor * updates_total,
                                                                      num_training_steps=updates_total)

    @staticmethod
    def multilabel_categorical_crossentropy(y_pred, y_true):
        """
        This function is a loss function for multi-label learning
        ref: https://kexue.fm/archives/7359

        y_pred: (batch_size_train, ... , type_size)
        y_true: (batch_size_train, ... , type_size)
        y_true and y_pred have the same shape，elements in y_true are either 0 or 1，
             1 tags positive classes，0 tags negtive classes(means tok-pair does not have this type of link).
        """
        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
        y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])# st - st
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return (neg_loss + pos_loss).mean()

    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        pred_result = []
        label_result = []
        pred_result_new = []
        label_result_new = []
        for i, data_batch in enumerate(tqdm(data_loader)):
            data_batch = [data.cuda() for data in data_batch[:-1]]

            word_inputs, bert_inputs, char_inputs, grid_labels, grid_labels_new, grid_mask2d, pieces2word, dist_inputs, word_mask2d = data_batch
            
            outputs, outputs_new = model(word_inputs, bert_inputs, char_inputs, grid_mask2d, dist_inputs, pieces2word, word_mask2d)

            grid_mask2d = grid_mask2d.clone()

            loss = config.alpha * self.multilabel_categorical_crossentropy(outputs[grid_mask2d], grid_labels[grid_mask2d]) + (1 - config.alpha) * self.multilabel_categorical_crossentropy(outputs_new[grid_mask2d], grid_labels_new[grid_mask2d])
      
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_list.append(loss.cpu().item())

            # outputs = torch.argmax(outputs, -1)
            grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
            outputs = outputs[grid_mask2d].contiguous().view(-1)
            grid_labels_new = grid_labels_new[grid_mask2d].contiguous().view(-1)
            outputs_new = outputs_new[grid_mask2d].contiguous().view(-1)


            label_result.append(grid_labels)
            pred_result.append(outputs)
            label_result_new.append(grid_labels_new)
            pred_result_new.append(outputs_new)

            self.scheduler.step()

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)
        label_result_new = torch.cat(label_result_new)
        pred_result_new = torch.cat(pred_result_new)
        p, r, f1, _ = precision_recall_fscore_support(label_result.cpu().numpy(),
                                                      np.int64(pred_result.cpu().detach().numpy()>0),
                                                      average="macro")

        p_new, r_new, f1_new, _ = precision_recall_fscore_support(label_result_new.cpu().numpy(),
                                                      np.int64(pred_result_new.cpu().detach().numpy()>0),
                                                      average="macro")

        table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
        table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] +
                      ["{:3.4f}".format(x) for x in [(f1+f1_new)/2, (p+p_new)/2, (r+r_new)/2]])
        logger.info("\n{}".format(table))
        return (f1+f1_new)/2

    def eval(self, epoch, data_loader, is_test=False):
        self.model.eval()

        pred_result = []
        label_result = []
        pred_result_new = []
        label_result_new = []
        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0

        with torch.no_grad():
            for i, data_batch in enumerate(tqdm(data_loader)):
                entity_text = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                word_inputs, bert_inputs, char_inputs, grid_labels, grid_labels_new, grid_mask2d, pieces2word, dist_inputs, word_mask2d = data_batch

                outputs, outputs_new = model(word_inputs, bert_inputs, char_inputs, grid_mask2d, dist_inputs, pieces2word, word_mask2d)
                length = word_inputs.ne(0).sum(dim=-1)

                grid_mask2d = grid_mask2d.clone()

                if i == 0:
                    ent_r, ent_p, ent_c = 0, 0, 0
                else:
                    ent_r, ent_p, ent_c = utils.decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy())
                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c
                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)
                grid_labels_new = grid_labels_new[grid_mask2d].contiguous().view(-1)
                outputs_new = outputs_new[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels)
                pred_result.append(outputs)
                label_result_new.append(grid_labels_new)
                pred_result_new.append(outputs_new)

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)
        label_result_new = torch.cat(label_result_new)
        pred_result_new = torch.cat(pred_result_new)

        p, r, f1, _ = precision_recall_fscore_support(label_result.cpu().numpy(),
                                                      np.int64(pred_result.cpu().detach().numpy()>0),
                                                      average="macro")

        p_new, r_new, f1_new, _ = precision_recall_fscore_support(label_result_new.cpu().numpy(),
                                                      np.int64(pred_result_new.cpu().detach().numpy()>0),
                                                      average="macro")
        if total_ent_r == 0 or total_ent_p == 0:
            e_f1, e_p, e_r = 0, 0, 0
        else:
            e_r = total_ent_c / total_ent_r
            e_p = total_ent_c / total_ent_p
            if total_ent_c == 0 or total_ent_r == 0:
                e_f1 = 0
            else:
                e_f1 = (2 * e_p * e_r / (e_p + e_r))
 
        title = "EVAL" if not is_test else "TEST"
        logger.info('{} Label F1 {}'.format(title, 0.5 * (f1_score(label_result.cpu().numpy(),
                                                            np.int64(pred_result.cpu().detach().numpy()>0),
                                                            average=None) + f1_score(label_result_new.cpu().numpy(),
                                                            np.int64(pred_result_new.cpu().detach().numpy()>0),
                                                            average=None))))

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [(f1+f1_new)/2, (p+p_new)/2, (r+r_new)/2]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])
        logger.info("\n{}".format(table))
        return e_f1

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

def seed_torch(seed=3306):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

import time
import json
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/cadec.json')
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--word_emb_size', type=int)
    parser.add_argument('--char_emb_size', type=int)
    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--type_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--conv_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--ffnn_hid_size', type=int)
    parser.add_argument('--biaffine_size', type=int)

    parser.add_argument('--dilation', type=str, help="e.g. 1,2,3")

    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--conv_dropout', type=float)
    parser.add_argument('--out_dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--bert_weight_decay', type=float)
    parser.add_argument('--warm_factor', type=float)

    parser.add_argument('--use_bert', type=int, help="1: true, 0: false")
    parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")

    parser.add_argument('--seed', type=int)
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--cr', type=int)


    args = parser.parse_args()
    
    bert_config = config.Config(args, is_bert=True)
    config = config.Config(args)
    

    seed_torch(config.seed)
    logger = utils.get_logger(config)
    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    logger.info("Loading Data")
    datasets = data_loader.load_data_bert(config)

    train_loader, dev_loader, test_loader = (
        DataLoader(dataset=dataset,
                   batch_size=config.batch_size,
                   collate_fn=data_loader.collate_fn,
                   shuffle=i == 0,
                   num_workers=4,
                   drop_last=i == 0)
        for i, dataset in enumerate(datasets)
    )

    updates_total = len(datasets[0]) // config.batch_size * config.epochs

    logger.info("Building Model")
    model = Model(config, bert_config)

    model = model.cuda()

    trainer = Trainer(model)

    best_f1 = 0
    best_test_f1 = 0
    for i in range(config.epochs):
        logger.info("Epoch: {}".format(i))
        trainer.train(i, train_loader)
        f1 = trainer.eval(i, dev_loader)
        test_f1 = trainer.eval(i, test_loader, is_test=True)
        if f1 > best_f1:
            best_f1 = f1
            best_test_f1 = test_f1
            trainer.save("model.pt")
    logger.info("Best DEV F1: {:3.4f}".format(best_f1))
    logger.info("Best TEST F1: {:3.4f}".format(best_test_f1))
    trainer.load("model.pt")
    trainer.eval("Final", test_loader, True)
