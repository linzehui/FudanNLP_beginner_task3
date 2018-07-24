import os
import torch
from sklearn.model_selection import train_test_split
from config import Config
from collections import Counter
from torch.autograd import Variable
from main import config

class Dictionary():

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.__vocab_size = 0
        self.add_word('<unk>')

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = self.__vocab_size
            self.__vocab_size += 1

    def __len__(self):
        return self.__vocab_size

    def get_index(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx['<unk>']

    def get_word(self, idx):
        return self.idx2word[idx]


def get_iter_dataset(txt_file):  # return generator
    with open(txt_file, 'r') as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            sentence = list(line.strip()) + ['<eos>']
            yield sentence


def get_dataset(txt_file):
    iter_dataset=  get_iter_dataset(txt_file)
    return list(iter_dataset)


def split_data(dataset, train_proportion=0.7):
    if type(dataset) != list:
        dataset = list(dataset)

    train_dataset, test_dataset = train_test_split(dataset, train_size=train_proportion,
                                                       test_size=1.0 - train_proportion,
                                                       shuffle=True)
    return train_dataset, test_dataset


def build_dict(dataset):
    dict = Dictionary()
    for sentence in dataset:
        for word in sentence:
            dict.add_word(word)

    return dict


def tokenize(dataset, dict):
    cnt = 0  # 计数
    for sentence in dataset:
        for word in sentence:
            cnt += 1
    token = 0
    ids = torch.LongTensor(cnt)
    for sentence in dataset:
        for word in sentence:
            ids[token] = dict.get_index(word)
            token += 1

    return ids


def batchify(token, batch_size):
    num_batch = len(token) // batch_size
    batch = token.narrow(0, 0, num_batch * batch_size)  # 丢掉后面的部分
    batch = batch.view(batch_size, -1).t().contiguous()  # batch= num_batch*batch_size 必须要先(batch_size,-1)再转置，否则数据不对；contiguous() 表示内存的存储区域是连续的

    return batch


def get_batch(batch_source, i, seq_len):
    seq_len = min(seq_len, len(batch_source) - 1 - i)  # len(source)==num_batch
    x = Variable(batch_source[i:i + seq_len])
    y = Variable(batch_source[i + 1:i + 1 + seq_len].view(-1))  # 展平
    if config.use_cuda:
        x,y=x.cuda(),y.cuda()
    return x,y


if __name__=="__main__":
    pass