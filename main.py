from config import Config

config = Config()

import argparse
import os
import math
import time
from datetime import timedelta
import torch.optim as optim
from torch.nn.functional import softmax
import torch
import torch.nn as nn
from torch.autograd import Variable

from model import RNNModel
import data


file_path = 'poetryFromTang.txt'

save_dir = 'checkpoints/' + file_path.split('.')[0]
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_name = file_path.split('.')[0] + '_{}.pt'


def repackage_hidden(h):
    """重新包装hidden_state，否则计算梯度会追溯到开头"""

    if type(h) == torch.Tensor:  # rnn/gru,在pytorch 0.4过后variable和Tensor不分了
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def train(model, batch_source):
    train_len=len(batch_source)
    seq_len=config.seq_len

    if config.use_cuda:
        model.cuda()

    criterion=nn.CrossEntropyLoss()

    start_time=time.time()

    print("starting to train...")
    for epoch in range(1,config.epoch+1):
        total_loss=0.0
        model.train()
        hidden= model.init_hidden(config.batch_size)
        for ibatch, i in enumerate(range(0, train_len - 1, seq_len)):
            x, y = data.get_batch(batch_source, i, seq_len)
            hidden = repackage_hidden(hidden)   #这边一定要repackage，否则会出错，'Trying to backward through the graph a second time, but the buffers have already been freed.'
            model.zero_grad()

            output, hidden = model(x, hidden)
            loss = criterion(output.view(-1,config.vocab_size), y)
            loss.backward()
            optimizer = optim.Adam(model.parameters())
            # 手动clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            optimizer.step()
            total_loss += loss.data

            if ibatch % config.log_interval == 0 and ibatch > 0:  # 每隔多少个批次输出一次状态
                cur_loss = total_loss.item() / config.log_interval
                elapsed = get_time_dif(start_time)
                print("Epoch {:3d}, {:5d}/{:5d} batches, loss {:5.2f}, ppl {:8.2f}, time {}".format(
                    epoch, ibatch, train_len // seq_len, cur_loss, math.exp(cur_loss), elapsed))
                # 为什么这里使用e作为指数,因为log也是e为底数，cur_loss为什么不需要1/N取平均，因为y展开，相当于已经取了平均
                evaluate(model,test_dataset,dict)
                total_loss = 0.0

        if epoch % config.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, model_name.format(epoch)))

def evaluate_iter(model,sentence,dict):
    if type(sentence) == str:
        sentence = str.split()
    sentence_len = len(sentence)
    sentence_idx = [dict.get_index(word) for word in sentence]
    model.eval()
    hidden = model.init_hidden(1)
    input=torch.LongTensor(1,1)
    total_p=0
    for i in range(1,sentence_len):
        current_idx = sentence_idx[i]
        input.data.fill_(current_idx)
        output, hidden = model(input, hidden)
        output = output.squeeze()
        output = softmax(output, dim=0)
        p = output[current_idx].data  # 概率
        total_p+=math.log(p)  #e为底
    return math.exp(-total_p*(1/sentence_len))

def evaluate(model,test_dataset,dict):
    ppl=0
    for sentence in test_dataset:
        ppl+=evaluate_iter(model,sentence,dict)
    ppl=ppl/len(test_dataset)
    print("evaluation ppl:",ppl)
    return ppl

if __name__ == '__main__':
    dataset = data.get_dataset(file_path)
    dict = data.build_dict(dataset)
    config.vocab_size=len(dict)
    train_dataset, test_dataset = data.split_data(dataset, train_proportion=config.train_proportion)
    train_tokens = data.tokenize(train_dataset, dict)
    model=RNNModel(config)
    train_batch_source=data.batchify(train_tokens,config.batch_size)  #传入batchify好的数据直接训练
    train(model,batch_source=train_batch_source)

    #test
    evaluate(model,test_dataset,dict)