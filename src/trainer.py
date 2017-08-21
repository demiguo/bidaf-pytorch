import sys
import torch 
import torch.autograd as autograd 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np
import argparse
import torch.utils.data
import datetime
from tqdm import tqdm


class Trainer:

    def __init__(self,model_config):
        self.config = model_config
        self.debug = False

    def run(self, model, id_new2old, data_loader, optimizer, mode="train"):
        # return model, optimizer, avg_loss, answer_dict
        assert mode == "train" or mode == "dev", "[ERROR]Try to execute Trainer in unknown mode: %s" % mode

        if mode == "train":
            model.train()
        else:
            model.eval()

        answer_dict = {}
        total_loss = torch.FloatTensor([0.0])
        total_batch = 0
        self.config.log.info("len(data_loader)= %d" % len(data_loader))
        for data in tqdm(data_loader, desc="running {} mode".format(mode)):
            data_variable = []
            for data_item in data:
                data_item = torch.squeeze(data_item, 0)
                if self.config.args["use_cuda"]:
                    data_item = data_item.cuda()
                data_variable.append(autograd.Variable(data_item))
            assert len(data_variable) == 9, "data loader error in Trainer: data_variable length {} not equal to 9".format(len(data_variable))

            #ids, passages, questions, 
            #passages_char, questions_char, 
            #passages_mask, questions_mask, 
            #answer_starts, answer_ends = data_variable

            total_batch += data_variable[0].size(0)

            if mode == "train":
                optimizer.zero_grad()
            cur_loss, cur_answer_dict = model(id_new2old, *data_variable)
            if self.config.args["mode"] == "fake":
                self.config.log.info("trainer: finish running model")
            assert not np.isnan(cur_loss.data[0]), "[WARNING]Model Diverge with Loss = NaN"
            total_loss += cur_loss.data
            answer_dict.update(cur_answer_dict)
            if self.config.args["mode"] == "fake":
                self.config.log.info("trainer: finish updating answer dict")

            if mode == "train":
                cur_loss /= data_variable[0].size(0)   # normalize loss
                print "cur_loss=", cur_loss
                cur_loss.backward()
                torch.nn.utils.clip_grad_norm(model.get_train_parameters(), self.config.args.get("clip", 5.0))
                optimizer.step()
            
            if self.config.args["mode"] == "fake":
                self.config.log.info("trainer: finish updating model")

        assert total_batch != 0, "Trainer: no data found (total_batch = 0)"
        avg_loss = total_loss / total_batch
        return model, optimizer, avg_loss, answer_dict
