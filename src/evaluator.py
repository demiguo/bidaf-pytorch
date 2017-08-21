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
import json 
import pandas as pd
import config
import os

class Evaluator:
	def __init__(self, config):
		self.config = config

	def official_eval(self, target_file, predict_dict, predict_file):
		f = open(predict_file, "w")
		json.dump(predict_dict, f)
		f.close()

		# NB(demi): very hacky for now
		cmd = "python official_evaluator.py {} {}".format(target_file, predict_file)

		self.config.log.info("running cmd: %s" % cmd)
		os.system(cmd)

		# json.dump(predict_dict)??
	def eval(self, eval_type, target_file, predict_dict, predict_file=""):
		if eval_type == "official":
			self.official_eval(target_file, predict_dict, predict_file)
		else:
			self.config.log.warning("eval: type {} not supported yet.".format(eval_type))
