import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sets import Set 
import json
import unicodedata
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm 
import re
import io
import os

import random

#################
### for model ###
#################
kVeryBigNumber = 1e30
kVerySmallNumber = 1e-30
kVeryPositiveNumber = kVeryBigNumber
kVeryNegativeNumber = -kVeryBigNumber

# assuming val is cuda FloatTensor
def zero_mask(val, mask):
    return val * mask.float()

def exp_mask(val, mask):
    return val + (1 - mask.float()) * kVeryNegativeNumber



################
### for data ###
################
""" normalize """
def normalize(text):
    return re.sub('\s+', ' ', unicodedata.normalize('NFD', text))

""" flatten json for train data """
def squad_flatten_json_train(config, filename):
    save_filename = filename + ".flatten"
    if os.path.isfile(save_filename+".npz"):
        config.log.info("squad flatten json train: already flattened - skip")
        datafile = np.load(save_filename+".npz")
        config.log.info("squad flatten json train: finish loading")
        return (datafile['data_id'], datafile['data_passage'], datafile['data_question'],
                datafile['data_answer']), Set(datafile['vocab_list']), Set(datafile['chars_list'])
    vocab = Set([])
    chars = Set([])
    f = json.loads(open(filename).read())
    data_passage = []
    data_question = []
    data_answer = []
    data_id = []
    counter = 0
    for passage in tqdm(f['data'], desc="squad flatten json train data"):
        counter += 1
        for paragraph in tqdm(passage['paragraphs'], desc="passage %d"%counter):
            context = paragraph['context']
            context = normalize(context)
            context_final = context.split('.')
            context_final = [nltk.word_tokenize(sent) for sent in context_final]
            for sent in context_final:
                for word in context:
                    vocab.add(word) 
                    for c in word:
                        chars.add(c)
            for qa in paragraph['qas']:
                id_, question, answers = qa['id'], qa['question'], qa['answers']
                answer = answers[0]['text']  # only 1 answer in training data
                #answer_start = answers[0]['answer_start']
                #answer_end = answer_start + len(answer)
                answer_final = [normalize(answer)]

                question = normalize(question)
                question_final = nltk.word_tokenize(question)
                for word in question_final:
                    vocab.add(word)
                    for c in word:
                        chars.add(c)
                data_passage.append(context_final)
                data_question.append(question_final)
                data_answer.append(answer_final)
                data_id.append(id_)
    assert len(data_passage) == len(data_question) and len(data_question) == len(data_answer)
    np.savez(save_filename, data_id=data_id, data_passage=data_passage, data_question=data_question, data_answer=data_answer, vocab_list=list(vocab), chars_list=list(chars))
    return (data_id, data_passage, data_question, data_answer), vocab, chars


""" flatten json for dev data """
def squad_flatten_json_dev(config, filename):
    save_filename = filename + ".flatten"
    if os.path.isfile(save_filename+".npz"):
        config.log.info("squad flatten json dev: already flattened - skip")
        datafile = np.load(save_filename+".npz")
        config.log.info("squad flatten json dev: finish loading")
        return (datafile['data_id'], datafile['data_passage'], datafile['data_question'],
                datafile['data_answer']), Set(datafile['vocab_list']), Set(datafile['chars_list'])
    vocab = Set([])
    chars = Set([])
    f = json.loads(open(filename).read())
    data_passage = []
    data_question = []
    data_answer = []
    data_id = []
    counter_paragraph = 0
    counter = 0
    for passage in tqdm(f['data'], desc="squad flatten json dev data"):
        counter += 1
        for paragraph in tqdm(passage['paragraphs'], desc="passage %d"%counter):
            context = paragraph['context']
            context = normalize(context)
            context_final = context.split('.')
            context_final = [nltk.word_tokenize(sent) for sent in context_final]
            for sent in context_final:
                for word in context:
                    vocab.add(word) 
                    for c in word:
                        chars.add(c)
            counter_paragraph += 1

            for qa in paragraph['qas']:
                id_, question, answers = qa['id'], qa['question'], qa['answers']
                answers = [normalize(a['text']) for a in answers]               
                answer_final = answers

                question = normalize(question)
                question_final = nltk.word_tokenize(question)
                for word in question_final:
                    vocab.add(word)
                    for c in word:
                        chars.add(c)

                data_passage.append(context_final)
                data_question.append(question_final)
                data_answer.append(answer_final)
                data_id.append(id_)

    assert len(data_passage) == len(data_question) and len(data_question) == len(data_answer)
    np.savez(save_filename, data_id=data_id, data_passage=data_passage, data_question=data_question, data_answer=data_answer, vocab_list=list(vocab), chars_list=list(chars))
    return (data_id, data_passage, data_question, data_answer), vocab, chars



""" Read all (train and test combined) data, return lists of (passage, question, answer) tuples and word dictionaries."""
def squad_read_data(config, train_file, dev_file, old_glove_file, new_glove_file):
    train_data, train_vocab, train_chars = squad_flatten_json_train(config, train_file)
    dev_data, dev_vocab, dev_chars = squad_flatten_json_dev(config, dev_file)

    config.log.info("squad read data: finish flatten json ......")
    # now build vocab and dictionaries
    chars = train_chars | dev_chars
    i2c, c2i = {0:'.'}, {'.':0}  # unknown character = 0
    counter = 1
    for c in chars:
        if c == '.':
            continue
        i2c[counter] = c
        c2i[c] = counter
        counter += 1

    vocab = train_vocab | dev_vocab

    # QS(demi): do i need to convert them into unicode
    i2w, w2i = {0:'<UNK>'}, {'<UNK>':0} # unknown word = 0
    counter = 1
    for word in vocab:
        if word == "<UNK>":
            continue
        i2w[counter] = word
        w2i[word] = counter
        counter += 1

    counter = len(w2i)


    # new glove file
    f = io.open(old_glove_file, encoding="utf-8")
    out = open(new_glove_file, "w")
    lines = f.readlines()
    assert len(lines) >= 1
    glove_dim = len(lines[0].split(" ")) - 1

    all_zero = " ".join(["0"] * glove_dim)
    glove_matrix = [all_zero] * counter  # for unknown words, we set them to all zeros
    print "glove_matrix.len = ", len(glove_matrix)
    for line in lines:
        line = line.split(" ")
        # TODO(demi): figure out the right thing to do
        word = normalize(line[0])
        vec_str = " ".join(line[1:])
        if word in vocab:
            if w2i[word] >= counter:
                print "word={} w2i={}, len(w2i)={} len(vocab)={} counter={}".format(word.encode("utf-8"),w2i[word],len(w2i),len(vocab),counter)
            glove_matrix[w2i[word]] = vec_str
        else:
            print "not found word {}".format(word.encode("utf-8"))
    for glove_line in glove_matrix:
        out.write(glove_line + "\n")
    f.close()
    out.close()

    return train_data, dev_data, i2w, w2i, i2c, c2i, new_glove_file, glove_dim, len(w2i), len(c2i)

""" get exact answer span index in passage based on answer text """
def get_answer_span(passage, answer, config):
    num_sent = len(passage)
    answer_len = len(answer.split(" "))
    # TODO(demi): not very efficient right now, we may need to speed it up

    max_p_length = config.args["max_p_length"]
    words = []
    positions = []
    for i in range(len(passage)):
        for j in range(len(passage[i])):
            words.append(passage[i][j])
            positions.append((i,j))
    for i in range(len(words)):
        j = i + answer_len # [i, j)
        if j > len(words):
            break # overflow
        if " ".join(words[i:j]) == answer:
            # get start_id
            x, y = positions[i]
            start_id =  x * max_p_length + y
            # get end_id
            x, y = positions[j-1]
            end_id = x * max_p_length + y
            return start_id, end_id
    config.log.warning("can't find answer for {}".format(answer))
    return 0, 0  

""" create mask and align data converted into indices. return new dataset & mask (numpy) """
def make_dataset(config, data, w2i, c2i):
    data_id, data_passage, data_question, data_answer = data

    batch_size = len(data)
    new_data_id = np.array(data_id)
    new_data_passage = np.zeros((batch_size, config.max_num_sent, config.max_p_length))
    new_data_passage_mask = np.zeros((batch_size, config.max_num_sent, config.max_p_length))
    new_data_passage_char = np.zeros((batch_size, config.max_num_sent, config.max_p_length, config.max_word_size))
    new_data_question = np.zeros((batch_size, config.max_q_length))
    new_data_question_mask = np.zeros((batch_size, config.max_q_length))
    new_data_question_char = np.zeros((batch_size, config.max_q_length, config.max_word_size))
    new_answer_start = np.zeros((batch_size))
    new_answer_end = np.zeros((batch_size))


    # new_answer_start - new_answer_end (happy to know you)
    for batch_id in tqdm(range(batch_size), desc="finding exact answer span"):
        cur_answer_start = []
        cur_answer_end = []
        for answer in data_answer[batch_id]:
            astart, aend = find_answer_span(data_passage[batch_id], answer, config.args["max_num_sent"], config.args["max_p_length"])
            cur_answer_start.append(astart)
            cur_answer_end.append(aend)

        # NB(demi): currently, even for dev, we only select on answer
        random.seed(config.args["seed"])
        select_id = random.choice(range(len(data_answer[batch_id])))
        new_answer_start[batch_id] = cur_answer_start[select_id]
        new_answer_end[batch_id] = cur_answer_end[select_id]

    # process passaged
    for i in range(batch_size):
        passage = data_passage[i]
        for j in range(len(passage)):
            sent = passage[j]
            for z in range(len(sent)):
                word = unicode(sent[z])  # sent[z] should already be unicode. add "unicode" for sanity
                assert word in w2i, "word {} not found in w2i dictionary".format(word)
                new_data_passage[i][j][z] = w2i(word)
                new_data_passage_mask[i][j][z] = 1
                for l in range(len(word)):
                    ch = word[l]
                    assert ch in c2i, "char {} not found in c2i dictionary".format(ch)
                    new_data_passage_char[i][j][z][l] = c2i[ch]


    # process questions
    for i in range(batch_size):
        question = data_question[i]
        for j in range(len(question)):
            word = question[j]
            assert word in w2i, "word {} not found in w2i dictionary".format(word)
            new_data_question[i][j] = w2i(word)
            new_data_question_mask[i][j] = 1
            for z in range(len(word)):
                ch = word[z]
                assert ch in c2i, "char {} not found in c2i dictionary".format(ch)
                new_data_question_char[i][j][z] = c2i[ch]

    return (new_data_id, new_data_passage, new_data_question, new_data_passage_char, new_data_question_char, new_data_passage_mask, new_data_question_mask,\
            new_answer_start, new_answer_end)

################
### datasets ###
################
""" Dataset. Each item is a batched dataset. """
class QnADataset(torch.utils.data.Dataset):
    def __init__(self, data, config):
        data_id, data_passage, data_question, data_passage_char, data_question_char, data_passage_mask, data_question_mask, answer_start, answer_end = data
        
        self.data_size = len(data_passage)
        indices = range(self.data_size)

        # NB(demi): assume we always don't shuffle data order in batches

        self.passages = []
        self.questions = []
        self.passages_char = []
        self.questions_char = []
        self.passages_mask = []
        self.questions_mask = []
        self.ids = []
        self.answer_starts = []
        self.answer_ends = []
        for i in range(0, self.data_size, config.args["batch_size"]):
            j = min(config.args["batch_size"] + i, self.data_size)
            assert i != j
            # batch: [i, j)
            self.ids.append(data_id[i:j])
            self.passages.append(data_passage[i:j])
            self.questions.append(data_question[i:j])
            self.passages_char.append(data_passage_char[i:j])
            self.questions_char.append(data_question_char[i:j])
            self.passages_mask.append(data_passage_mask[i:j])
            self.questions_mask.append(data_question_mask[i:j])
            self.answer_starts.append(answer_start[i:j])
            self.answer_ends.append(answer_ends[i:j])

    def __getitem__(self, index):
        # NB(demi): assume it's already numpy arrays
        return self.ids[index],\
               self.passages[index],\
               self.questions[index],\
               self.passages_char[index],\
               self.questions_char[index],\
               self.passages_mask[index],\
               self.questions_mask[index],\
               self.answer_starts[index],\
               self.answer_ends[index]

    def __len__(self):
        return self.data_size