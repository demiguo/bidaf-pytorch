import sys
import random
import torch 
import torch.autograd as autograd 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np
import argparse
import torch.utils.data
import datetime
import logging

class Config:

    """ load constant config (constants & information) """
    def load_constant_config():
        info = {}
        # NB(demi): right now, let's only support train for simplicity
        info["modes"] = {"train" : "[default] train a new model"}
        return info

    info = load_constant_config()


    def __init__(self):
        self.args = {}
        self.epoch = -1


    """ generate model name from user config """
    def generate_model_name(self):
        print "not implemented"

    """ Save config """
    def save_config(self):
        print "not implemented"

    """ load user config (hyper-parameters) """
    def load_user_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-train_file", "--train_file", help="Training Data", required=False, default="../data/squad/train-v1.1.json")
        parser.add_argument("-dev_file", "--dev_file", help="Dev Data", required=False, default="../data/squad/dev-v1.1.json")
        parser.add_argument("-log_file", "--log_file", help="Logging File", required=False, default="../models/sample.log")
        parser.add_argument("-glove_file", "--glove_file", help="Glove File", required=False, default="../data/glove/glove.6B.100d.txt")
        parser.add_argument("-model_dir", "--model_dir", help="Model Directory", required=False, default="../models/")
        parser.add_argument("-model_name", "--model_name", help="Model Name", required=False, default="tmp")
        
        parser.add_argument("-mode", "--mode", help="Mode of the program", required=False, default="train")
        parser.add_argument("-use_cuda", "--use_cuda", help="Whether to use cuda or not", type=bool, required=False, default=True)
        parser.add_argument("-default_device", "--default_device", help="Cuda default GPU #", type=int, required=False, default=0)
        parser.add_argument("-seed", "--seed", help="Random seed for all random generator", type=int, required=False, default=1)

        parser.add_argument("-max_epoch", "--max_epoch", help="Max Epoch to Run", type=int, required=False, default=100)
        parser.add_argument("-batch_size", "--batch_size", help="Batch Size", type=int, required=False, default=60)
        parser.add_argument("-optimizer", "--optimizer", help="Optimizer", required=False, default="Adadelta")
        parser.add_argument("-lr", "--lr", help="Learning Rate", type=float, required=False, default=0.5)
        parser.add_argument("-dropout", "--dropout", help="Drop Out Rate", type=float, required=False, default=0.2)
        parser.add_argument("-?", "?")

        parser.add_argument("-hidden_dim", "--hidden_dim", help="Hidden Size of the model", type=int, required=False, default=100)
        parser.add_argument("-char_embedding_dim", "--char_embedding_dim", help="Char Embedding dimension (output channel dimension of CNN)", type=int, required=False, default=100)
        parser.add_argument("-char_single_embedding_dim", "--char_single_embedding_dim", help="Char Embedding dimension for single character", type=int, required=False, default=8)
        parser.add_argument("-filter_height", "--filter_height", help="Filter Height for Char Embedding CNN layer", type=int, required=False, default=5)

        self.args = vars(parser.parse_args())

        # NB(demi): These following parameters will be automatically generated in main
        self.args["vocab_size"] = 0
        self.args["char_vocab_size"] = 0
        self.args["glove_dim"] = 100 


        # set up logging
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)
        self.fh = logging.FileHandler(self.args["log_file"])
        self.fh.setLevel(logging.DEBUG)
        self.ch = logging.StreamHandler(sys.stdout)
        self.ch.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S")
        self.fh.setFormatter(formatter)
        self.ch.setFormatter(formatter)
        self.log.addHandler(self.fh)
        self.log.addHandler(self.ch)


        self.log.info("log to %s" % self.args["log_file"])
        np.random.seed(self.args["seed"])
        random.seed(self.args["seed"])
        torch.manual_seed(self.args["seed"])
        if torch.args["use_cuda"]:
            if not torch.cuda.is_available():
                self.config.log.warning("cuda: your cuda is not available. force to switch to non-cuda mode...")
                torch.args["use_cuda"] = False
            else:
                torch.cuda.manual_seed(self.args["seed"])
                torch.cuda.set_device(self.args["default_device"])
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.args["use_cuda"] else {}

        # model name
        self.generate_model_name()


        self.save_config(self.args["model_dir"] + self.args["model_name"] + ".config")

    """ update user config (hyper-parameters) - single"""
    def update_single(self, name, val):
        self.args[name] = val

    """ update user config (hyper-parameters) - batch"""
    def update_batch(self, name_val_pairs):
        for pr in name_val_pairs:
            name,val=pr
            self.update_single(name, val)

