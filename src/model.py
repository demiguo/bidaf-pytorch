
import torch
import torch.nn as nn 

import layers
from utils import exp_mask 

import nltk
import tqdm
import pandas as pd
import numpy as np
from tqdm import tqdm 

eps = 1e-10
class BiDAF(nn.module):
    def __init__(self, config, i2w):
        self.config = config 
        self.i2w = i2w
        self.vocab_size = self.config.args["vocab_size"]
        self.hidden_dim = self.config.args["hidden_dim"]

        # embedding layer
        self.char_embedding_dim = self.config.args["char_embedding_dim"]
        self.word_embedding_dim = self.config.args["glove_dim"]

        # QS(demi): is this correct? in this case, self.embedding_dim may not equal to self.hidden_dim
        self.embedding_dim = self.char_embedding_dim + self.word_embedding_dim
        self.char_embedding_layer = nn.layers.CharEmbeddingLayer(\
            self.config.args["char_single_embedding_dim"],
            self.char_embedding_dim, # out_channel_dim
            self.config.args["filter_height"],
            self.config.args["dropout"],
            self.config.args["char_vocab_size"])
        self.word_embedding_layer = nn.Embedding(self.vocab_size, self.word_embedding_dim)
        self.init_word_embedding()
        self.highway_network = layers.HighwayNetowrk(self.embedding_dim, self.config.args["highway_num_layers"], self.config.args["dropout"])

        # contextual layer
        self.contextual_dim = self.hidden_dim * 2  
        self.context_biLSTM = nn.LSTM(self.embedding_dim, self.contextual_dim // 2, num_layers=1, bidirectional=True, dropout=0.5, batch_first=True)

        # attention layer
        self.attention_dim = self.hidden_dim * 8
        self.attention_layer = nn.layers.AttentionLayer(self.config)

        # model layer
        self.model_dim = self.hidden_dim * 2
        self.model_biLSTM = nn.LSTM(self.attention_dim, self.model_dim // 2, num_layers=2, bidirectional=True, dropout=0.5, batch_first=True)

        # output layer 
        self.w_p1 = nn.Linear(self.attention_dim + self.model_dim, 1, bias=False)
        self.w_p2 = nn.Linear(self.attention_dim + self.model_dim, 1, bias=False)
        self.output_biLSTM = nn.LSTM(self.model_dim, self.model_dim // 2, num_layers=1, bidirectional=True, dropout=0.5, batch_first=True)


    def init_config(passages, questions, passages_char, questions_char, answers):
        # NB(demi): max_num_sent, max_p_length, max_word_size, max_q_length should be fixed, and in config
        #           batch_size may vary [depends on implementation]
        self.batch_size, self.max_num_sent, self.max_p_length, self.max_word_size = passages_char.size()
        assert passages.size() == (self.batch_size, self.max_num_sent, self.max_p_length, self.max_word_size)
        self.max_q_length = questions_char.size(1)
        assert questions_char.size() == (self.batch_size, self.max_q_length, self.max_word_size)
        assert questions.size() == (self.batch_size, self.max_q_length)

    def init_hidden(hidden_size, num_layer = 1):
        if self.config.args["use_cuda"]:
            h0 = autograd.Variable(torch.randn(2 * num_layer, self.batch_size, hidden_size // 2).cuda())
            c0 = autograd.Variable(torch.randn(2 * num_layer, self.batch_size, hidden_size // 2).cuda())
        else:
            h0 = autograd.Variable(torch.randn(2 * num_layer, self.batch_size, hidden_size // 2))
            c0 = autograd.Variable(torch.randn(2 * num_layer, self.batch_size, hidden_size // 2)
        return (h0, c0)

    def init_word_embedding(self):
        glove_weight = np.loadtxt(self.config.args["glove_file"])
        if self.config.args["use_cuda"]:
            self.word_embedding_layer.weight.data.copy_(torch.FloatTensor(glove_weight).cuda())
        else:
            self.word_embedding_layer.weight.data.copy_(torch.FloatTensor(glove_weight))
    
        # NB(demi): for unknown words, we set it to be all 0s. so we now use trainable glove. 
        #           we may change this later.
        # self.word_embedding_layer.weight.requires_grad = False

    def get_train_parameters(self):
        params = []
        for param in self.parameters():
            if param.requires_grad == True:
                params.append(param)
        return params
        
    def get_loss(p_1, p_2, answer_starts, answer_ends):
        loss_criterion = nn.CrossEntropyLoss()  # log softmax + nll_loss
        answer_starts = answer_starts.contiguous().view(self.batch_size, self.max_num_sent * self.max_p_length)        
        answer_ends = answer_ends.contiguous().view(self.batch_size, self.max_num_sent * self.max_p_length)        
        assert p_1.size() == (self.batch_size)  # 0-based index in flat [max_num_sent*max_p_length] array
        assert p_2.size() == (self.batch_size) # 0-based index in flat [max_num_sent*max_p_length] array
        loss = loss_criterion(p_1, answer_starts) + loss_criterion(p_2, answer_ends)
        return loss

    def get_answer(p_1_softmax, p_2_softmax, passages, passages_mask, ids):
        # NB/TODO(demi): now, we only consider spans in a single sentence
        i2answer_dict = {}
        p_1 = p_1_softmax.contiguous().view(self.batch_size, self.max_num_sent, self.max_p_length)
        p_2 = p_2_softmax.contiguous().view(self.batch_size, self.max_num_sent, self.max_p_length)
        for batch_id in tqdm(range(self.batch_size), desc="In model: get answer"):
            best_answer = (-1, -1, -1)  # sent_id, start_id, end_id
            best_prob = -1e30
            for sent_id in range(self.max_num_sent):
                start_idx = -1
                start_prob = -1e30

                for idx in range(self.max_p_length):
                    if passages_mask[batch_id][sent_id][idx] == 0:
                        # out of range
                        break
                    cur_prob = p_1[batch_idx][sent_id][idx]
                    if cur_prob > start_prob + eps:
                        start_prob = p_1[batch_idx][sent_id][idx]
                        start_idx = idx

                    if cur_prob * start_prob > best_prob + eps:
                        best_answer = (sent_id, start_idx, idx)
                        best_prob = cur_prob * start_prob 
            if -1 in best_answer: 
                # invalid
                self.config.log.warning("model -> get answer: can't find best answer span")
                best_answer_text = ""
            else:
                word_ids = passages[batch_id, best_answer[0], best_answer[1]:beset_answer[2]+1]
                word_ids = word_ids.contiguous().view(-1).cpu().numpy()
                words = map(lambda idx: i2w[idx], word_ids)
                best_answer_text = " ".join(words)
            i2answer_dict[ids[batch_id]] = best_answer_text
        return i2answer_dict

    def forward(ids, passages, questions, passages_char, questions_char, passages_mask, questions_mask, answer_starts, answer_ends, need_get_answer=True):
        init_config(passages, questions, passages_char, questions_char, answers)
        # NB(demi): let's first don't consider optimization, and consider (passage, question, answer) triples
        #            the format of answers are not finalized
        #            let's first consider all words in passages are concatenated, and all words in a question is concatenated
        #            let's first ignore masking, but manually padded to same length (max_p_length, max_q_length)
        # passages: batch_size * max_num_sent * max_p_length (in word_indices)
        # passages_char: batch_size * max_p_length * max_word_size (in character indices)
        # questions are similar format

        # QS(demi): max_num_sent * max_p_length OR max_num_sent, max_p_length? 
        # QS(demi): is it ok to store all different variables? will it increase the overall memory?
        # QS(demi): does 'viewing' very costly?

        ### EMBEDDING LAYER ###

        # QS(demi): is it shared? 
        passages_char = passages_char.contiguous().view(self.batch_size, self.max_num_sent * self.max_p_length, self.max_word_size)
        p_char_embed = self.char_embedding_layer(passages_char, self.training)
        q_char_embed = self.char_embedding_layer(questions_char, self.training)
        assert p_char_embed.size() == (self.batch_size, self.max_num_sent * self.max_p_length, self.char_embedding_dim)
        assert q_char_embed.size() == (self.batch_size, self.max_q_length, self.char_embedding_dim)

        passages = passages.contiguous().view(self.batch_size, self.max_num_sent * self.max_p_length)
        p_word_embed = self.word_embedding_layer(passages)
        q_word_embed = self.word_embedding_layer(questions)
        assert p_word_embed.size() == (self.batch_size, self.max_num_sent * self.max_p_length, self.word_embedding_dim)
        assert q_word_embed.size() == (self.batch_size, self.max_q_length, self.word_embedding_dim)

        # combine embedding
        p_concat_embed = torch.cat((p_char_embed, p_word_embed), 2)
        q_concat_embed = torch.cat((q_char_embed, q_word_embed), 2)
        p_embed_X = self.highway_network(p_concat_embed, self.training)
        q_embed_Q = self.highway_netowrk(q_concat_embed, self.training)
        assert p_concat_embed.size() == (self.batch_size, self.max_num_sent * self.max_p_length, self.embedding_dim)
        assert q_concat_embed.size() == (self.batch_size, self.max_q_length, self.embedding_dim)
        assert p_embed_X.size() == (self.batch_size, self.max_num_sent * self.max_p_length, self.embedding_dim)
        assert q_embed_Q.size() == (self.batch_size, self.max_q_length, self.embedding_dim)

        # now reshape back
        p_embed_X = p_embed_X.contiguous().view(self.batch_size, self.max_num_sent, self.max_p_length, self.embedding_dim)
        

        ### CONTEXTUAL EMBEDDING LAYER ###
        self.p_context_hidden = self.init_hidden(self.contextual_dim)
        self.q_context_hidden = self.init_hidden(self.contextual_dim)

        p_embed_X = p_embed_X.contiguous().view(self.batch_size * self.max_num_sent, self.max_p_lengh, self.embedding_dim)
        p_context_H, self.p_context_hidden = self.context_biLSTM(p_embed_X, self.p_context_hidden)
        q_context_U, self.q_context_hidden = self.context_biLSTM(q_embed_Q, self.q_context_hidden)
        assert p_context_H.size() == (self.batch_size * self.max_num_sent, self.max_p_length, self.contextual_dim)
        assert q_context_U.size() == (self.batch_size, self.max_q_length, self.contextual_dim)
        p_context_H = p_context_H.contiguous().view(self.batch_size, self.max_num_sent, self.max_p_length. self.contextual_dim)
        # TODO(demi): zero mask on p_context_H and q_context_U

        ### ATTENTION LAYER ###
        G = self.attention_layer(p_context_H, q_context_U, passages_mask, questions_mask, self.training)
        assert G.size() == (self.batch_size, self.max_num_sent, self.max_p_length, self.attention_dim)

        ### MODELING LAYER ###
        G_patched = G.contiguous().view(self.batch_size * self.max_num_sent, self.max_p_length, self.attention_dim)
        self.model_hidden = self.init_hidden(self.model_dim, 2)
        M, self.model_hidden = self.model_biLSTM(G_patched, self.model_hidden)
        assert M.size() == (self.batch_size * self.max_num_sent, self.max_p_length, self.model_dim)
        M = M.contiguous().view(self.batch_size, self.max_num_sent, self.max_p_length, self.model_dim)
        # TODO(demi): zero mask on M

        ### OUTPUT LAYER ###
        p_1 = torch.cat((G, M), -1) # dim 3
        p_1 = p_1.contiguous().view(self.batch_size * self.max_num_sent * self.max_p_length, self.model_dim + self.attention_dim)
        # TODO(demi): add dropout layer if necessary
        p_1 = self.w_p1(p_1)
        assert p_1.size() == (self.batch_size * self.max_num_sent * self.max_p_length, 1)
        p_1 = p_1.contiguous().view(self.batch_size, self.max_num_sent, self.max_p_length)
        p_1 = exp_mask(p_1, passages_mask)
        p_1 = p_1.contiguous().view(self.batch_size, self.max_num_sent * self.max_p_length)
        p_1_softmax = nn.Softmax()(p_1)

        self.output_hidden = self.init_hidden(self.model_dim)
        M_patched = M.contiguous().view(self.batch_size * self.max_num_sent, self.max_p_length, self.model_dim)
        M_2, self.output_hidden = self.output_biLSTM(M_patched, self.output_hidden)
        M_2 = M_2.contiguous().view(self.batch_size, self.max_num_sent, self.max_p_length, self.model_dim)

        p_2 = torch.cat((G, M_2), -1) # dim 3
        p_2 = p_2.contiguous().view(self.batch_size * self.max_num_sent * self.max_p_length, self.attention_dim + self.model_dim)
        # TODO(demi): add dropout layer if necessary
        p_2 = self.w_p2(p_2)
        assert p_2.size() == (self.batch_size * self.max_num_sent * self.max_p_length, 1)
        p_2 = p_2.contiguous().view(self.batch_size, self.max_num_sent, self.max_p_length)
        p_2 = exp_mask(p_2, passages_mask)
        p_2 = p_2.view(self.batch_size, self.max_num_sent * self.max_p_length)
        p_2_softmax = nn.Softmax()(p_2)


        loss = get_loss(p_1, p_2, answer_starts, answer_ends)
        if need_get_answer:
            answers = get_answer(p_1_softmax, p_2_softmax, passages, passages_mask, ids)
        else:
            answers = {}
        return loss, answers
