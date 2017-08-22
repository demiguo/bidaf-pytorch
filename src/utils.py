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
    text = unicodedata.normalize('NFD', text).replace("''", "\"").replace("``", "\"")
    return text

""" word tokenize """
def tokenize_word(text):
    text = text.encode("ascii", errors="replace")
    text = " ".join(text.split("-"))
    text = nltk.word_tokenize(text)
    return text

""" flatten json for train data """
def squad_flatten_json_train(config, filename, overwrite=False):
    config.log.info("squad flatten json train: %s" % filename)

    max_num_sent, max_p_length, max_q_length, max_word_size = 0,0,0,0
    save_filename = filename + ".flatten"
    if (not overwrite) and os.path.isfile(save_filename+".npz"):
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
    counter_paragraph = 0
    counter = 0
    for passage in tqdm(f['data'], desc="squad flatten json train data"):
        counter += 1
        for paragraph in passage['paragraphs']:
            context = paragraph['context']
            context = normalize(context)


            #config.log.info("new paragraph={}".format(context.encode("utf-8")))
            #config.log.info("split into # sentences = %d" % len(context_final))

            context_final = [tokenize_word(context)]
            max_num_sent = max(max_num_sent, len(context_final))
            for sent in context_final:
                max_p_length = max(max_p_length, len(sent))
                for word in sent:
                    max_word_size = max(max_word_size, len(word))
                    vocab.add(word)
                    for c in word:
                        chars.add(c)
            counter_paragraph += 1

            for qa in paragraph['qas']:
                id_, question, answers = qa['id'], qa['question'], qa['answers']
                #config.log.info("new qa: {}\n{}\n{}\n".format(id_, question.encode("utf-8"), answers))
                answers = [normalize(a['text']) for a in answers]
                answer_final = answers

                question = normalize(question)
                question_final = tokenize_word(question)
                max_q_length = max(max_q_length, len(question_final))
                for word in question_final:
                    max_word_size = max(max_word_size, len(word))
                    vocab.add(word)
                    for c in word:
                        chars.add(c)

                data_passage.append(context_final)
                data_question.append(question_final)
                data_answer.append(answer_final)
                data_id.append(id_)

    assert len(data_passage) == len(data_question) and len(data_question) == len(data_answer)
    np.savez(save_filename, data_id=data_id, data_passage=data_passage, data_question=data_question, data_answer=data_answer, vocab_list=list(vocab), chars_list=list(chars))
    print "max_num_sent={}, max_p_length={}, max_q_length={}, max_word_size={}\n".format(\
        max_num_sent, max_p_length, max_q_length, max_word_size)
    return (data_id, data_passage, data_question, data_answer), vocab, chars


""" flatten json for dev data """
def squad_flatten_json_dev(config, filename, overwrite=False):
    config.log.info("squad flatten json dev: %s" % filename)
    max_num_sent, max_p_length, max_q_length, max_word_size = 0,0,0,0
    save_filename = filename + ".flatten"
    if (not overwrite) and os.path.isfile(save_filename+".npz"):
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
        for paragraph in passage['paragraphs']:
            context = paragraph['context']
            context = normalize(context)
            context_final = [tokenize_word(context)]
            max_num_sent = max(max_num_sent, len(context_final))
            for sent in context_final:
                max_p_length = max(max_p_length, len(sent))
                for word in sent:
                    max_word_size = max(max_word_size, len(word))
                    vocab.add(word)
                    for c in word:
                        chars.add(c)
            counter_paragraph += 1

            for qa in paragraph['qas']:
                id_, question, answers = qa['id'], qa['question'], qa['answers']
                answers = [normalize(a['text']) for a in answers]
                answer_final = answers

                question = normalize(question)
                question_final = tokenize_word(question)
                max_q_length = max(max_q_length, len(question_final))
                for word in question_final:
                    max_word_size = max(max_word_size, len(word))
                    vocab.add(word)
                    for c in word:
                        chars.add(c)

                data_passage.append(context_final)
                data_question.append(question_final)
                data_answer.append(answer_final)
                data_id.append(id_)

    assert len(data_passage) == len(data_question) and len(data_question) == len(data_answer)
    np.savez(save_filename, data_id=data_id, data_passage=data_passage, data_question=data_question, data_answer=data_answer, vocab_list=list(vocab), chars_list=list(chars))
    print "max_num_sent={}, max_p_length={}, max_q_length={}, max_word_size={}\n".format(\
        max_num_sent, max_p_length, max_q_length, max_word_size)
    return (data_id, data_passage, data_question, data_answer), vocab, chars



""" Read all (train and test combined) data, return lists of (passage, question, answer) tuples and word dictionaries."""
def squad_read_data(config, train_file, dev_file, old_glove_file, new_glove_file, overwrite=False):
    train_data, train_vocab, train_chars = squad_flatten_json_train(config, train_file, overwrite)
    dev_data, dev_vocab, dev_chars = squad_flatten_json_dev(config, dev_file, overwrite)


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
    config.log.info("glove_matrix.len = %d" % len(glove_matrix))

    if config.args["mode"] == "fake":
        config.log.info("fake mode: skipped loading glove")
    else:
        for line in tqdm(lines, desc="generate subset & reordered glove"):
            line = line.split(" ")
            # TODO(demi): figure out the right thing to do
            word = normalize(line[0])
            vec_str = " ".join(line[1:])
            if word in vocab:
                assert w2i[word] < counter
                glove_matrix[w2i[word]] = vec_str
            else:
                config.log.debug("not found word {}".format(word.encode("utf-8")))

    for glove_line in glove_matrix:
        out.write(glove_line + "\n")
    f.close()
    out.close()

    return train_data, dev_data, i2w, w2i, i2c, c2i, new_glove_file, glove_dim, len(w2i), len(c2i)

""" get exact answer span index in passage based on answer text """
def get_answer_span(passage, answer, config):
    if len(answer) == 0:
        config.log.debug("answer is null - skipped")
        return 0, 0

    answer = tokenize_word(answer)
    answer = " ".join(answer)
    #???
    num_sent = len(passage)
    answer_len = len(answer.split(" "))
    # TODO(demi): not very efficient right now, we may need to speed it up

    max_p_length = config.args["max_p_length"]
    #print "get_answer_span: max_p_length=", max_p_length

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
    config.log.warning("can't find answer for {}".format(answer.encode("utf-8")))
    print "answer=", answer.encode("utf-8")
    print "passage:"
    for sent in passage:
        sent_str = " ".join(sent)
        print sent_str.encode("utf-8")
    print "\n"
    print "answer: %s" % answer
    return 0, 0

""" create mask and align data converted into indices. return new dataset & mask (numpy) """
def make_dataset(config, data, w2i, c2i):
    data_id, data_passage, data_question, data_answer = data

    batch_size = len(data_id)
    new_data_passage = np.zeros((batch_size, config.args["max_num_sent"], config.args["max_p_length"]), dtype=int)
    new_data_passage_mask = np.zeros((batch_size, config.args["max_num_sent"], config.args["max_p_length"]), dtype=int)
    new_data_passage_char = np.zeros((batch_size, config.args["max_num_sent"], config.args["max_p_length"], config.args["max_word_size"]), dtype=int)
    new_data_question = np.zeros((batch_size, config.args["max_q_length"]), dtype=int)
    new_data_question_mask = np.zeros((batch_size, config.args["max_q_length"]), dtype=int)
    new_data_question_char = np.zeros((batch_size, config.args["max_q_length"], config.args["max_word_size"]), dtype=int)
    new_answer_start = np.zeros((batch_size), dtype=int)
    new_answer_end = np.zeros((batch_size), dtype=int)

    # id conversion map
    new2old, old2new = {}, {}
    new_data_id = []
    for batch_id in tqdm(range(batch_size), desc="creating QA ID conversion map"):
        old_id = data_id[batch_id]
        new_id = batch_id
        new_data_id.append(new_id)
        new2old[new_id] = old_id
        old2new[old_id] = new_id

    # new_answer_start - new_answer_end (happy to know you)
    config.log.info("make dataset: num batch=%d" % batch_size)
    bad_answers = 0
    for batch_id in tqdm(range(batch_size), desc="finding exact answer span"):
        cur_answer_start = []
        cur_answer_end = []
        for answer in data_answer[batch_id]:
            astart, aend = get_answer_span(data_passage[batch_id], answer, config)
            if astart == 0 and aend == 0:
                continue
            cur_answer_start.append(astart)
            cur_answer_end.append(aend)

        #assert len(cur_answer_start) >= 1, "found 0 answers for batch_id=%d" % batch_id
        if len(cur_answer_start) == 0:
            cur_answer_start.append(0)
            cur_answer_end.append(0)
            bad_answers += 1
        # NB(demi): currently, even for dev, we only select on answer
        random.seed(config.args["seed"])
        select_id = random.choice(range(len(cur_answer_start)))
        new_answer_start[batch_id] = cur_answer_start[select_id]
        new_answer_end[batch_id] = cur_answer_end[select_id]
    print "bad ansers : %d out of %d " % (bad_answers, batch_size)

    # process passaged
    for i in range(batch_size):
        passage = data_passage[i]
        for j in range(len(passage)):
            sent = passage[j]
            for z in range(len(sent)):
                word = unicode(sent[z])  # sent[z] should already be unicode. add "unicode" for sanity
                assert word in w2i, "word {} not found in w2i dictionary".format(word)
                new_data_passage[i][j][z] = w2i[word]
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
            new_data_question[i][j] = w2i[word]
            new_data_question_mask[i][j] = 1
            for z in range(len(word)):
                ch = word[z]
                assert ch in c2i, "char {} not found in c2i dictionary".format(ch)
                new_data_question_char[i][j][z] = c2i[ch]

    print "make_dataset: new_answer_end=", new_answer_end
    return (new2old, old2new), (new_data_id, new_data_passage, new_data_question, new_data_passage_char, new_data_question_char, new_data_passage_mask, new_data_question_mask,\
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
        self.num_batch = 0
        for i in range(0, self.data_size, config.args["batch_size"]):
            self.num_batch += 1
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
            self.answer_ends.append(answer_end[i:j])

        # sanity check
        assert len(self.ids) == self.num_batch
        assert len(self.passages) == self.num_batch
        assert len(self.passages_char) == self.num_batch
        assert len(self.questions) == self.num_batch
        assert len(self.questions_char) == self.num_batch
        assert len(self.passages_mask) == self.num_batch
        assert len(self.questions_mask) == self.num_batch
        assert len(self.answer_starts) == self.num_batch
        assert len(self.answer_ends) == self.num_batch

        #config.log.info("in dataset: passages_char each batch size={}".format(self.passages_char[0].shape))
        #print "DATASET answer_ends=", self.answer_ends
    def __getitem__(self, index):
        # NB(demi): assume it's already numpy arrays
        return np.array(self.ids[index]),\
               np.array(self.passages[index]),\
               np.array(self.questions[index]),\
               np.array(self.passages_char[index]),\
               np.array(self.questions_char[index]),\
               np.array(self.passages_mask[index]),\
               np.array(self.questions_mask[index]),\
               np.array(self.answer_starts[index]),\
               np.array(self.answer_ends[index])

    def __len__(self):
        return self.num_batch
