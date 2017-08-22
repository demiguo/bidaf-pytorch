import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import exp_mask

class MyLinear(nn.Module):
    def __init__(self, input_size, output_size, dropout=1.0):
        super(MyLinear, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = dropout

    def forward(self, x, is_train):
        if self.dropout < 1.0:
            assert is_train is not None
            if is_train:
                x = nn.Dropout(p=self.dropout)(x)
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
        assert text_char.size() == (batch_size * max_length, self.char_single_embedding_dim, max_word_length), "text_char.size()=%s"%(text_char.size())
        text_char = self.cnn(text_char)
        assert text_char.size() == (batch_size * max_length, self.char_embedding_dim, text_char.size(2))
        text_char = text_char.contiguous().view(batch_size * max_length * self.char_embedding_dim, -1)
        text_char = nn.functional.relu(text_char)

        # maxpool
        text_char = torch.max(text_char, 1)[0]
        assert text_char.size() == (batch_size * max_length * self.char_embedding_dim, )
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
        assert len(x.size()) == 2 and x.size(1) == self.size
        for layer in range(self.num_layers):
            gate = nn.functional.sigmoid(self.gate[layer](x, is_train))
            trans = nn.functional.relu(self.trans[layer](x, is_train))
            x = gate * trans + (1 - gate) * x
        return x

# QS(demi): only softmax per sentence not per paragraph???

class AttentionLayer(nn.Module):
    def __init__(self, config):
        super(AttentionLayer, self).__init__()
        self.config = config
        self.attention_linear = MyLinear(config.args["contextual_dim"] * 3, 1, config.args["dropout"])

    def bi_attention(self, is_train, h, u, h_mask, u_mask):
        # NB(demi): assume we always have mask
        assert h_mask is not None and u_mask is not None

        # reformat to (batch_size, max_num_sent, max_p_length, max_q_length, contextual_dim)
        h_aug = h.unsqueeze(3).repeat(1, 1, 1, self.max_q_length, 1)

        u_aug = u.unsqueeze(1).unsqueeze(2).repeat(1, self.max_num_sent, self.max_p_length, 1, 1)
        h_mask_aug = h_mask.unsqueeze(3).repeat(1, 1, 1, self.max_q_length)
        u_mask_aug = u_mask.unsqueeze(1).unsqueeze(2).repeat(1, self.max_num_sent, self.max_p_length, 1)

        assert h_mask_aug.size() == u_mask_aug.size()
        # NB(demi): perform dot product (equivalent to and operator)
        hu_mask_aug = h_mask_aug * u_mask_aug
        assert hu_mask_aug.size() == h_mask_aug.size()

        # sanity check
        check_size = (self.batch_size, self.max_num_sent, self.max_p_length, self.max_q_length, self.contextual_dim)
        assert h_aug.size() == check_size and u_aug.size() == check_size
        assert h_mask_aug.size() == check_size[:-1] and u_mask_aug.size() == check_size[:-1]

        # get attention matrix
        # NB(demi): assume it's always tri-linear
        hu_aug = h_aug * u_aug
        hu_concat_aug = torch.cat((h_aug, u_aug, hu_aug), 4)
        hu_concat_aug = hu_concat_aug.view(self.batch_size * self.max_num_sent * self.max_p_length * self.max_q_length, self.contextual_dim * 3)
        logits = self.attention_linear(hu_concat_aug, is_train).view(self.batch_size, self.max_num_sent, self.max_p_length, self.max_q_length)
        logits = exp_mask(logits, hu_mask_aug)

        # get c2q attention
        # (batch_size, max_num_sent, max_p_length, max_q_length)
        # -> (batch_size, max_num_sent, max_p_length, contextual_dim)
        c2q_logits = logits.view(self.batch_size * self.max_num_sent * self.max_p_length, self.max_q_length)
        c2q_logits = nn.Softmax()(c2q_logits)
        c2q_logits = c2q_logits.view(self.batch_size, self.max_num_sent, self.max_p_length, self.max_q_length).unsqueeze(4).repeat(1,1,1,1,self.contextual_dim)
        u_a = c2q_logits * u_aug
        u_a = torch.sum(u_a, 3).view(self.batch_size, self.max_num_sent, self.max_p_length, self.contextual_dim)

        # get q2c attention
        # (batch_size, max_num_sent, max_p_length)
        # -> (batch_size, max_num_sent, contextual_dim)
        logits_maxq = torch.max(logits, 3)[0].view(self.batch_size, self.max_num_sent, self.max_p_length)
        q2c_logits = logits_maxq.view(self.batch_size * self.max_num_sent, self.max_p_length)
        q2c_logits = nn.Softmax()(q2c_logits)
        q2c_logits = q2c_logits.view(self.batch_size, self.max_num_sent, self.max_p_length).unsqueeze(3).repeat(1,1,1,self.contextual_dim)
        h_a = q2c_logits * h
        assert h_a.size() == (self.batch_size, self.max_num_sent, self.max_p_length, self.contextual_dim)
        h_a = torch.sum(h_a, 2)
        if len(h_a.size()) == 3:
            h_a = h_a.unsqueeze(2)
        #print "after torch.sum(2) : h_a.size()=", h_a.size()
        assert h_a.size() == (self.batch_size, self.max_num_sent, 1, self.contextual_dim)
        h_a = h_a.repeat(1, 1, self.max_p_length, 1)
        #print "h_a.size()=", h_a.size()
        #print "self.batch_size=%d, self.max_num_sent=%d, self.max_p_length=%d, self.contextual_dim=%d" % (self.batch_size, self.max_num_sent, self.max_p_length, self.contextual_dim)
        assert h_a.size() == (self.batch_size, self.max_num_sent, self.max_p_length, self.contextual_dim)

        return u_a, h_a

    def forward(self, h, u, h_mask, u_mask, is_train):
        # assume we always use q2c and c2q attention
        self.batch_size, self.max_num_sent, self.max_p_length, self.contextual_dim = h.size()
        assert u.size(0) == self.batch_size and u.size(2) == self.contextual_dim
        self.max_q_length = u.size(1)

        # sanity check
        assert h_mask.size() == (self.batch_size, self.max_num_sent, self.max_p_length)
        assert u_mask.size() == (self.batch_size, self.max_q_length)

        # by default, contextual_dim = 2d (d is hidden_dim)
        u_a, h_a = self.bi_attention(is_train, h, u, h_mask, u_mask)
        assert u_a.size() == h_a.size()
        assert h_a.size() == (self.batch_size, self.max_num_sent, self.max_p_length, self.contextual_dim)

        g = torch.cat((h, u_a, h * u_a, h * h_a), 3)
        assert g.size() == (self.batch_size, self.max_num_sent, self.max_p_length, 4 * self.contextual_dim)
        return g
