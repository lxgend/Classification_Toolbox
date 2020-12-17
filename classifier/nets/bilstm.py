# coding=utf-8
import torch
import torch.nn as nn

# "[START]", "[END]"
START_TAG = 'START'
STOP_TAG = 'STOP'
START_TAG = '<START>'
STOP_TAG = '<STOP>'
embedding_dim = 100,
hidden_dim = 128 // 2
dropout = 1.0,
num_rnn_layers = 1

torch.manual_seed(123)  # 保证每次运行初始化的随机数相同
vocab_size = 5000  # 词表大小
embedding_size = 100  # 词向量维度
num_classes = 2  # 二分类
sentence_max_len = 64  # 单个句子的长度
hidden_size = 16

num_layers = 1  # 一层lstm
num_directions = 2  # 双向lstm
lr = 1e-3
batch_size = 16
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiLSTM(nn.Module):
    def __init__(
            self,
            embedding_dim,
            hidden_dim,
            num_rnn_layers,
            num_directions,
            dropout,
            num_classes,

            vocab_size,
            batch_size,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_rnn_layers = num_rnn_layers
        self.num_directions = num_directions
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=self.num_rnn_layers,
                            batch_first=True,
                            dropout=0,    # num_layers=1时是无效的
                            bidirectional=(num_directions == 2))

        self.liner = nn.Linear(hidden_dim * num_directions * num_rnn_layers, num_classes)
        self.act_func = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)  # 随机初始化

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tag_size)

        # 转移矩阵，随机初始化
        # Matrix of transition parameters. Entry i,j is the score of transitioning *to* i *from* j
        self.transitions = nn.Parameter(torch.randn(self.tag_size, self.tag_size))
        # self.transitions.data[:, self.tag_map[START_TAG]] = -1000.
        # self.transitions.data[self.tag_map[STOP_TAG], :] = -1000.

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        # self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        # self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

    def init_hidden(self, batch_size):
        """
        random initialize hidden variable 随机初始化
        num_layers * num_directions
        :return: 两个tensor: h0, c0
        """
        h0 = torch.randn(self.num_rnn_layers * self.num_directions, batch_size, self.hidden_dim).to(device)
        c0 = torch.randn(self.num_rnn_layers * self.num_directions, batch_size, self.hidden_dim).to(device)
        return h0, c0

    def forward(self, sentence):
        # sentence.shape: torch.Size([16, 42]), 分别是: batch大小，seq长度
        batch_size = sentence.size(0)
        seq_length = sentence.size(1)

        # 由于数据集不一定是预先设置的batch_size的整数倍，所以用size(1)获取当前数据实际的batch
        batch_size = x.size(1)

        # lstm最初的前项输出
        h0, c0 = self.init_hidden(batch_size)

        # out[seq_len, batch_size, num_directions * hidden_size]。多层lstm，out只保存最后一层每个时间步t的输出h_t
        # h_n, c_n [num_layers * num_directions, batch_size, hidden_size]
        out, (h_n, c_n) = self.lstm(sentence, (h0, c0))

        x = self.liner(x)
        x = self.act_func(x)

        return x



    def forwaddd(self, x):
        # lstm的输入维度为 [seq_len, batch_size, input_size]
        # x [batch_size, sentence_length, embedding_size]
        x = x.permute(1, 0, 2)  # [sentence_length, batch_size, embedding_size]

        # 由于数据集不一定是预先设置的batch_size的整数倍，所以用size(1)获取当前数据实际的batch
        batch_size = x.size(1)

        # 设置lstm最初的前项输出
        h_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)

        # out[seq_len, batch_size, num_directions * hidden_size]。多层lstm，out只保存最后一层每个时间步t的输出h_t
        # h_n, c_n [num_layers * num_directions, batch_size, hidden_size]
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))

        x = h_n  # [num_layers*num_directions, batch_size, hidden_size]
        x = x.permute(1, 0, 2)  # [batch_size, num_layers*num_directions, hidden_size]
        x = x.contiguous().view(batch_size,
                                self.num_layers * self.num_directions * self.hidden_size)  # [batch_size, num_layers*num_directions*hidden_size]
        x = self.liner(x)
        x = self.act_func(x)
        return x

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    

if __name__ == '__main__':
    pass
