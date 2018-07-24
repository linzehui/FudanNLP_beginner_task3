import torch.nn as nn
from torch.autograd import Variable
import torch


class RNNModel(nn.Module):
    def __init__(self, config):
        super(RNNModel, self).__init__()
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.rnn_type = config.rnn_type
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers

        self.dropout = nn.Dropout(config.dropout)
        self.tie_weights = config.tie_weights
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)  # encoder

        if self.rnn_type in ['RNN', 'LSTM', 'GRU']:
            self.rnn = getattr(nn, self.rnn_type)(self.embedding_dim, self.hidden_dim,
                                                  self.num_layers, dropout=config.dropout)
        else:
            raise ValueError("rnn_type error,options must be ['RNN','LSTM','GRU']")

        self.decoder = nn.Linear(self.hidden_dim, self.vocab_size)  # decoder

        if self.tie_weights:
            if self.hidden_dim != self.embedding_dim:
                raise ValueError('hidden_dim==embedding_dim when tie_weight==True')
            self.decoder.weight = self.embedding.weight

    def forward(self, inputs, hidden_state):
        embedding = self.dropout(self.embedding(inputs))
        output, hidden_state = self.rnn(embedding,
                                        hidden_state)  # output shape:(seq_len,batch_size,hidden_dim)
        decoded = self.decoder(output.view(-1, output.size(2)))  # 先展平做好映射
        return decoded.view(output.size(0), output.size(1), -1), hidden_state  # 复原

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.fill_(0)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        # print(weight)
        #
        # if self.rnn_type=='LSTM':  #lstm:(h0,c0)
        #     return (Variable(torch.zeros(self.num_layers,batch_size,self.hidden_dim)),
        #             Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim)))
        # else:
        #     return Variable(torch.zeros(self.num_layers,batch_size,self.hidden_dim))
        if self.rnn_type == 'LSTM':  # lstm：(h0, c0)
            return (Variable(weight.new(self.num_layers, batch_size, self.hidden_dim).zero_()),
                    Variable(weight.new(self.num_layers, batch_size, self.hidden_dim).zero_()))
        else:  # gru 和 rnn：h0
            return Variable(weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())
