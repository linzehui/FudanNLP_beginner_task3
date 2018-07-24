
import torch

class Config():
    embedding_dim = 64
    rnn_type = 'GRU'
    hidden_dim = 64
    num_layers = 2
    vocab_size=0

    dropout = 0.5  # take effect only when num_layers>1
    tie_weights = True  # share weights between encoder and decoder

    train_proportion = 0.9
    test_proportion = 0.3
    batch_size = 16
    seq_len = 10  # seq_len

    clip = 0.25
    lr = 0.01  # learning rate

    epoch = 100
    log_interval = 50  # 隔多少批次输出一次状态
    save_interval = 10  # 多少轮次保存一次参数

    use_cuda = torch.cuda.is_available()

