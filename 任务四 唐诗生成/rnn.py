# 导入模块
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


# 初始化权重，初始化神经网络中的线性层权重和偏置。
def weights_init(m):
    classname = m.__class__.__name__  # 获取模块m的类名（此处应为Linear）
    # 检查类名中是否为线性层
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())  # 获得权重张量的形状
        fan_in = weight_shape[1]  # 输入维度
        fan_out = weight_shape[0]  # 输出维度
        w_bound = np.sqrt(6. / (fan_in + fan_out))  # 计算边界值（w_bound）
        m.weight.data.uniform_(-w_bound, w_bound)  # 使用均匀分布初始化权重，范围在[-w_bound, w_bound]
        m.bias.data.fill_(0)  # 将偏置初始化为0
        print("初始化线性层权重。")  # 输出提示信息


# 自定义词嵌入层。
class Word_embedding(nn.Module):
    def __init__(self, vocab_length, embedding_dim):
        """
        :param vocab_length:词汇表的大小。
        :param embedding_dim:每个单词被映射到连续向量空间的向量长度。（例如，embedding_dim是100，则每个单词将被表示为一个100维的向量。）
        """
        super(Word_embedding, self).__init__()
        w_embeding_random_intial = np.random.uniform(-1, 1, size=(vocab_length, embedding_dim))  # 生成均匀分布的随机初始化权重
        self.word_embedding = nn.Embedding(vocab_length, embedding_dim)  # 定义一个nn.Embedding层
        self.word_embedding.weight.data.copy_(torch.from_numpy(w_embeding_random_intial))  # 将生成的权重复制到nn.Embedding层的权重中

    def forward(self, input_sentence):
        """
        定义前向传播方法。
        :param input_sentence: 输入到模型中的一组单词索引。（每个整数代表一个单词词汇表中的索引）
        """
        sen_embed = self.word_embedding(input_sentence)  # 进行前向传播，将输入的词索引转换为词嵌入。
        return sen_embed


# 定义基于LSTM的模型，用于处理序列数据。
class RNN_model(nn.Module):
    def __init__(self, batch_sz, vocab_len, word_embedding, embedding_dim, lstm_hidden_dim):
        """
        定义初始化方法。
        :param batch_sz:批处理的大小。
        :param vocab_len:词汇表长度。
        :param word_embedding:词嵌入层。
        :param embedding_dim:词嵌入的维度。
        :param lstm_hidden_dim:LSTM隐藏层的维度。
        """
        super(RNN_model, self).__init__()
        self.batch_size = batch_sz
        self.vocab_length = vocab_len
        self.word_embedding_lookup = word_embedding
        self.word_embedding_dim = embedding_dim
        self.lstm_dim = lstm_hidden_dim
        #########################################
        # here you need to define the "self.rnn_lstm"  the input size is "embedding_dim" and the output size is "lstm_hidden_dim"
        # the lstm should have two layers, and the  input and output tensors are provided as (batch, seq, feature)
        # 定义LSTM层，输入embedding_dim，输出lstm_hidden_dim，两层LSTM，输入张量形状为(batch,seq,feature)
        self.rnn_lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, num_layers=2, batch_first=True)
        ##########################################
        self.fc = nn.Linear(lstm_hidden_dim, vocab_len)  # 定义全连接层，将LSTM的输出映射到词汇表大小，用于预测下一个词的概率。
        self.apply(weights_init)  # 应用自定义的函数初始化权重。
        self.softmax = nn.LogSoftmax(dim=1)  # 将输出转换为对数概率，在维度1上操作。

    def forward(self, sentence, is_test=False):
        """
        定义前向传播方法。
        :param sentence:输入句子。
        :param is_test:检测训练阶段还是测试阶段。
        """
        # 通过词嵌入层将输入的句子转换为嵌入向量，并调整输入张量的形状为(1,sequence_length,embedding_dim)，1表示批次大小，sequence_length是句子中单词数量，embedding_dim是嵌入向量的维度。
        batch_input = self.word_embedding_lookup(sentence).view(1, -1, self.word_embedding_dim)
        ################################################
        # LSTM有两层，批次大小为1（每次输入一句）。
        h0 = torch.zeros(2, 1, self.lstm_dim)  # 初始LSTM的隐藏状态
        c0 = torch.zeros(2, 1, self.lstm_dim)  # 初始LSTM的细胞状态
        output, (hn, cn) = self.rnn_lstm(batch_input, (h0, c0))  # 进行前向传播，获取每个时间步的输出和最后一个时间步的隐藏状态和细胞状态。
        ################################################
        out = output.contiguous().view(-1, self.lstm_dim)  # 将LSTM的输出展平
        out = F.relu(self.fc(out))  # 通过全连接层将LSTM的输出映射到词汇表大小
        out = self.softmax(out)  # 将输出转换为对数概率
        # 如果是测试模式，返回最后一个时间步的输出，用于预测下一个单词。否则，返回所有时间步的输出。
        if is_test:
            prediction = out[-1, :].view(1, -1)
            output = prediction
        else:
            output = out
        return output
