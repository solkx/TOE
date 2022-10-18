
import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import prettytable as pt
from gensim.models import KeyedVectors
from transformers import AutoTokenizer
import os
import utils
import requests
import copy
os.environ["TOKENIZERS_PARALLELISM"] = "false"

dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'
    PRE = '<pre>'
    NSUC = '<nsuc>'
    DSUC = '<dsuc>'
    CON = '<con>'



    def __init__(self, frequency=0):
        self.token2id = {self.PAD: 0, self.UNK: 1}
        self.id2token = {0: self.PAD, 1: self.UNK}
        self.token2count = {self.PAD: 1000, self.UNK: 1000}
        self.frequency = frequency

        self.char2id = {self.PAD: 0, self.UNK: 1}
        self.id2char = {0: self.PAD, 1: self.UNK}

        # self.label2id = {self.PAD: 0, self.SUC: 1}
        # self.id2label = {0: self.PAD, 1: self.SUC}
        self.old_label_list = [self.SUC, self.PRE, "ht", "th"]
        self.new_label_list = [self.NSUC, self.DSUC, self.CON]
        self.label2id = {self.NSUC: 0, self.DSUC: 1, self.CON: 2, self.SUC: 3, "ht": 4, self.PRE: 5, "th": 6}
        self.id2label = {0: self.NSUC, 1: self.DSUC, 2: self.CON, 3: self.SUC, 4: "ht", 5: self.PRE, 6: "th"}

    def add_token(self, token):
        token = token.lower()
        if token in self.token2id:
            self.token2count[token] += 1
        else:
            self.token2id[token] = len(self.token2id)
            self.id2token[self.token2id[token]] = token
            self.token2count[token] = 1

        assert token == self.id2token[self.token2id[token]]

    def add_char(self, char):
        char = char.lower()
        if char not in self.char2id:
            self.char2id[char] = len(self.char2id)
            self.id2char[self.char2id[char]] = char

        assert char == self.id2char[self.char2id[char]]

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label

        assert label == self.id2label[self.label2id[label]]

    def remove_low_frequency_token(self):
        new_token2id = {self.PAD: 0, self.UNK: 1}
        new_id2token = {0: self.PAD, 1: self.UNK}

        for token in self.token2id:
            if self.token2count[token] > self.frequency and token not in new_token2id:
                new_token2id[token] = len(new_token2id)
                new_id2token[new_token2id[token]] = token

        self.token2id = new_token2id
        self.id2token = new_id2token

    def __len__(self):
        return len(self.token2id)

    # def label_to_id(self, label):
    #     label = label.lower()
    #     return self.label2id[label]

    def encode(self, text):
        return [self.token2id.get(x.lower(), 1) for x in text]

    def encode_char(self, text):
        return [self.char2id.get(x, 1) for x in text]

    def decode(self, ids):
        return [self.id2token.get(x) for x in ids]


def collate_fn(data):
    word_inputs, bert_inputs, char_inputs, grid_labels_old, grid_labels_new, grid_mask2d, pieces2word, dist_inputs, word_mask2d, entity_text = map(list, zip(*data))

    batch_size = len(word_inputs)
    max_tok = np.max([x.shape[0] for x in word_inputs])
    max_pie = np.max([x.shape[0] for x in bert_inputs])
    max_char = np.max([len(w) for x in char_inputs for w in x])

    word_inputs = pad_sequence(word_inputs, True)
    bert_inputs = pad_sequence(bert_inputs, True)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    char_mat = torch.zeros((batch_size, max_tok, max_char), dtype=torch.long)
    char_inputs = fill(char_inputs, char_mat)
    dis_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    dist_inputs = fill(dist_inputs, dis_mat)
    word_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    word_mask2d = fill(word_mask2d, word_mat)
    labels_mat_old = torch.zeros((batch_size, max_tok, max_tok, grid_labels_old[0].shape[-1]), dtype=torch.long)
    grid_labels_old = fill(grid_labels_old, labels_mat_old)
    labels_mat_new = torch.zeros((batch_size, max_tok, max_tok, grid_labels_new[0].shape[-1]), dtype=torch.long)
    grid_labels_new = fill(grid_labels_new, labels_mat_new)
    mask2d_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.bool)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)
    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)

    return word_inputs, bert_inputs, char_inputs, grid_labels_old, grid_labels_new, grid_mask2d, pieces2word, dist_inputs, word_mask2d, entity_text


class RelationDataset(Dataset):
    def __init__(self, word_inputs, bert_inputs, char_inputs, grid_labels_old, grid_labels_new, grid_mask2d, pieces2word, dist_inputs, word_mask2d, entity_text):
        self.word_inputs = word_inputs
        self.bert_inputs = bert_inputs
        self.char_inputs = char_inputs
        self.grid_labels_old = grid_labels_old
        self.grid_labels_new = grid_labels_new
        self.grid_mask2d = grid_mask2d
        self.pieces2word = pieces2word
        self.dist_inputs = dist_inputs
        self.word_mask2d = word_mask2d
        self.entity_text = entity_text

    def __getitem__(self, item):
        return torch.LongTensor(self.word_inputs[item]), \
               torch.LongTensor(self.bert_inputs[item]), \
               torch.LongTensor(self.char_inputs[item]), \
               torch.LongTensor(self.grid_labels_old[item]), \
               torch.LongTensor(self.grid_labels_new[item]), \
               torch.LongTensor(self.grid_mask2d[item]), \
               torch.LongTensor(self.pieces2word[item]), \
               torch.LongTensor(self.dist_inputs[item]), \
               torch.LongTensor(self.word_mask2d[item]), \
               self.entity_text[item]

    def __len__(self):
        return len(self.word_inputs)


def process_bert(data, tokenizer, vocab):
    word_inputs = []
    bert_inputs = []
    char_inputs = []
    grid_labels_old = []
    grid_labels_new = []
    grid_mask2d = []
    dist_inputs = []
    entity_text = []
    pieces2word = []
    word_mask2d = []
    for index, instance in enumerate(data):
        if len(instance['sentence']) == 0:
            continue
        _word_inputs = np.array(vocab.encode([word for word in instance['sentence']]))

        if tokenizer is not None:
            tokens = [tokenizer.tokenize(word) for word in instance['sentence']]
            pieces = [piece for pieces in tokens for piece in pieces]
            _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
            _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])
        else:
            _bert_inputs = _word_inputs
            pieces = _word_inputs

        length = len(_word_inputs)
        max_char_length = np.max([len(x) for x in instance['sentence']])
        _char_inputs = np.zeros((length, max_char_length), dtype=np.int)
        _grid_labels_old = np.zeros((length, length, len(vocab.old_label_list)), dtype=np.int)
        _grid_labels_new = np.zeros((length, length, len(vocab.new_label_list)), dtype=np.int)
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool)
        _dist_inputs = np.zeros((length, length), dtype=np.int)
        _grid_mask2d = np.ones((length, length), dtype=np.bool)
        _word_mask2d = np.ones((length, length), dtype=np.int)

        for i, token in enumerate(instance['sentence']):
            _char_inputs[i, :len(token)] = np.array(vocab.encode_char(token))

        if "word" in instance:
            for word_idx in instance["word"]:
                s, e = word_idx[0], word_idx[-1] + 1
                _word_mask2d[s:e, s:e] = 2

        if tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)

        for k in range(length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k

        for i in range(length):
            for j in range(length):
                if _dist_inputs[i, j] < 0:
                    _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                else:
                    _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
        _dist_inputs[_dist_inputs == 0] = 19
        
        for entity in instance["ner"]:
            index = entity["index"]
            for i in range(len(index)):
                if i == 0:
                    continue
                _grid_labels_old[index[i], index[i - 1], 1] = 1
            new_index = []
            for i in range(len(index)):
                if i == 0:
                    ind = [index[i]]
                else:
                    if index[i-1] == index[i]-1:
                        ind.append(index[i])
                    else:
                        new_index.append(copy.deepcopy(ind))
                        ind = [index[i]]
                if i + 1 >= len(index):
                    break
                _grid_labels_old[index[i], index[i + 1], 0] = 1
            _grid_labels_old[index[-1], index[0], 3] = 1
            _grid_labels_old[index[0], index[-1], 2] = 1
            if len(ind) != 0:
                new_index.append(copy.deepcopy(ind))
            if len(new_index) == 1:
                for new_item in new_index:
                    _grid_labels_new[new_item[-1], new_item[0], 1] = 1
            else:
                for i, new_item in enumerate(new_index):
                    _grid_labels_new[new_item[-1], new_item[0], 2] = 1
                    if i == 0:
                        continue
                    else:
                        _grid_labels_new[new_index[i-1][-1], new_index[i][0], 0] = 1
        _entity_text = set([utils.convert_index_to_text(e["index"])
                            for e in instance["ner"]])

        word_inputs.append(_word_inputs)
        bert_inputs.append(_bert_inputs)
        char_inputs.append(_char_inputs)
        grid_labels_old.append(_grid_labels_old)
        grid_labels_new.append(_grid_labels_new)
        grid_mask2d.append(_grid_mask2d)
        dist_inputs.append(_dist_inputs)
        pieces2word.append(_pieces2word)
        entity_text.append(_entity_text)
        word_mask2d.append(_word_mask2d)

    return word_inputs, bert_inputs, char_inputs, grid_labels_old, grid_labels_new, grid_mask2d, pieces2word, dist_inputs, word_mask2d, entity_text


def fill_vocab(vocab, dataset):
    entity_num = 0
    for instance in dataset:

        for token in instance['sentence']:
            vocab.add_token(token)
            for char in token:
                vocab.add_char(char)

        # for entity in instance["ner"]:
        #     vocab.add_label(entity["type"])
        entity_num += len(instance["ner"])
    return entity_num


def load_data_bert(config):
    with open('./data/{}/train.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('./data/{}/dev.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    with open('./data/{}/test.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    # train_data = train_data + dev_data
    # dev_data = test_data



    tokenizer = None
    while tokenizer is None:
        try:
            # tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache/")
            tokenizer = AutoTokenizer.from_pretrained(config.bert_name)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            continue


    vocab = Vocabulary()
    train_ent_num = fill_vocab(vocab, train_data)
    dev_ent_num = fill_vocab(vocab, dev_data)
    test_ent_num = fill_vocab(vocab, test_data)

    table = pt.PrettyTable([config.dataset, 'sentences', 'entities'])
    table.add_row(['train', len(train_data), train_ent_num])
    table.add_row(['dev', len(dev_data), dev_ent_num])
    table.add_row(['test', len(test_data), test_ent_num])
    config.logger.info("\n{}".format(table))

    config.word_num = len(vocab.token2id)
    config.char_num = len(vocab.char2id)
    config.label_num = len(vocab.label2id)
    config.old_label_num = len(vocab.old_label_list)
    config.new_label_num = len(vocab.new_label_list)
    config.vocab = vocab

    train_dataset = RelationDataset(*process_bert(train_data, tokenizer, vocab))
    dev_dataset = RelationDataset(*process_bert(dev_data, tokenizer, vocab))
    test_dataset = RelationDataset(*process_bert(test_data, tokenizer, vocab))
    return train_dataset, dev_dataset, test_dataset


def load_embedding(config):
    vocab = config.vocab
    wvmodel = KeyedVectors.load_word2vec_format(config.embedding_path, binary=True)
    embed_size = config.word_emb_size
    embedding = np.random.uniform(-0.01, 0.01, (len(vocab), embed_size))
    hit = 0
    for token, i in vocab.token2id.items():
        if token in wvmodel:
            hit += 1
            embedding[i, :] = wvmodel[token]
    print("Total hit: {} rate {:.4f}".format(hit, hit / len(vocab)))
    embedding[0] = np.zeros(embed_size)
    embedding = torch.FloatTensor(embedding)
    return embedding
