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
    def init():
        args = {}
    def get_config_from_user(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-log_file', '--log_file', help="Logging File", required=False, default="../models/sample.log")
      
        
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

