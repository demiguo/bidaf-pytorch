import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MyLinear(nn.Module):
    def __init__(self, input_size, output_size, dropout=1.0):
        super(MyLinear, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = dropout

    def forward(self, x, is_train):
        if dropout < 1.0:
            assert is_train is not None
            if is_train:
                x = nn.Dropout(p=dropout)(x)
            x = self.linear(x)
        return x


class CharEmbeddingLayer(nn.Module):
    def __init__(self, char_single_embedding_dim, char_embedding_dim, filter_height, dropout, char_vocab_size):
        super(CharEmbeddingLayer, self).__init__()
        self.char_single_embedding_dim = char_single_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.filter_height = filter_height
        self.dropout = dropout
        self.char_vocab_size = char_vocab_size

        self.embedding_lookup = nn.Embedding(self.char_vocab_size, char_single_embedding_dim)
        self.cnn = nn.Conv1d(char_single_embedding_dim, char_embedding_dim, filter_height, padding=0)

    def forward(self, text_char, is_train):
        batch_size, max_length, max_word_length = text_char.size()

        # embedding look up
        text_char = text_char.contiguous().view(batch_size * max_length, max_word_length)
        text_char = self.embedding_lookup(text_char)
        assert text_char.size() == (batch_size * max_length, max_word_length, self.char_single_embedding_dim)
        
        # dropout + cnn
        if self.dropout < 1.0:
            assert is_train is not None
            if is_train:
                text_char = nn.Dropout(p=self.dropout)(text_char)
        text_char = torch.transpose(text_char, 1, 2)
        assert text_char.size() == (batch_size * max_length, self.char_single_embedding_dim, max_word_length)
        text_char = self.cnn(text_char)
        assert text_char.size() == (batch_size * max_length, self.char_embedding_dim, -1)
        text_char = text_char.contiguous().view(batch_size * max_length * self.char_embedding_dim, -1)
        text_char = nn.functional.relu(text_char)

        # maxpool 
        text_char = torch.max(text_char, 1)
        assert text_char.size() == (batch_size * max_length * self.char_embedding_dim, 1)
        text_char = text_char.contiguous().view(batch_size, max_length, self.char_embedding_dim)
        return text_char


class HighwayNetwork(nn.Module):
    def __init__(self, size, num_layers, dropout):
        super(HighwayNetwork, self).__init__()

        self.num_layers = num_layers
        self.size = size

        self.trans = nn.ModuleList([MyLinear(size, size, dropout) for _ in range(num_layers)])
        self.gate = nn.ModuleList([MyLinear(size, size, dropout) for _ in range(num_layers)])

    def forward(self, x, is_train):
        assert x.size() == (-1, size)
        for layer in range(self.num_layers):
            gate = nn.functional.sigmoid(self.gate[layer](x, is_train))
            trans = nn.functional.relu(self.trans[layer](x, is_train))
            x = gate * trans + (1 - gate) * x
        return x

# QS(demi): only softmax per sentence not per paragraph???

class AttentionLayer(nn.Module):
    def __init__(self, config):
        self.config = config

    def bi_attention(self, is_train, h, u):
        print "not implemented"

    def forward(self, h, u, is_train):
        # assume we always use q2c and c2q attention
        batch_size, max_p_length, contextual_dim = h.size()
        # by default, contextual_dim = 2d (d is hidden_dim)
        u_a, h_a = self.bi_attention(is_train, h, u)
        assert u_a.size() == h_a.size()
        assert h_a.size() == (batch_size, max_p_length, contextual_dim)

        g = torch.cat((h, u_a, h * u_a, h * h_a), 2)
        assert g.size() == (batch_size, max_p_length, 4 * contextual_dim)
        return g
