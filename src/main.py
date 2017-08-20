#import spacy
import sys
import torch 
import torch.autograd as autograd 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np
import argparse
import torch.utils.data
import datetime
import copy


from utils import squad_read_data, make_dataset, QnADataset
from config import Config
from model import BiDAF

from trainer import Trainer
from evaluator import Evaluator

def main(argv):
    config = Config()
    config.load_user_config()
    config.log.info("finish loading user config")

    train_file = config.args["train_file"]
    dev_file = config.args["dev_file"]
    old_glove_file = config.args["glove_file"]
    new_glove_file = config.args["glove_file"] + ".subset"

    # TODO(demi): switch "overwrite" to False
    train_data_raw, dev_data_raw, i2w, w2i, i2c, c2i, new_glove_file, glove_dim, vocab_size, char_vocab_size\
         = squad_read_data(config, train_file, dev_file, old_glove_file, new_glove_file, overwrite=True)  
    config.log.info("finish reading squad data in raw formats")

    config.update_batch([("glove_file", new_glove_file), 
                   ("glove_dim", glove_dim), 
                   ("vocab_size", vocab_size),
                   ("char_vocab_size", char_vocab_size)])


    config.log.warning("reminder: now we only support train/fake mode")
    assert config.args["mode"] in ["trian", "fake"], "mode not found"

    train_id_conversion, train_data = make_dataset(config, train_data_raw, w2i, c2i)
    dev_id_conversion, dev_data = make_dataset(config, dev_data_raw, w2i, c2i)
    config.log.info("finish making datasets: reformatting raw data")

    train_data = QnADataset(train_data, config)
    dev_data = QnADataset(dev_data, config)
    config.log.info("finish generating datasets")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, **config.kwargs)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=1, **config.kwargs)
    config.log.info("finish generating data loader")


    model = BiDAF(config, i2w)
    config.log.info("finish creating model")
    if config.args["use_cuda"]:
        model.cuda()

    # log config and model
    config.log.info(config.format_string())
    config.log.info("model:{}".format(model))

    if config.args['optimizer'] == "Adam":
        optimizer = optim.Adam(model.get_train_parameters(), lr=config.args['lr'], weight_decay=config.args['weight_decay'])
    if config.args['optimizer'] == "Adamax":
        optimizer = optim.Adamax(model.get_train_parameters(), lr=config.args['lr'], weight_decay=config.args['weight_decay'])
    if config.args['optimizer'] == "SGD":
        optimizer = torch.optim.SGD(model.get_train_parameters(), lr=config.args['lr'], momentum=0.9, weight_decay=config.args['weight_decay'])
    if config.args['optimizer'] == "Adadelta":
        optimizer = torch.optim.Adadelta(model.get_train_parameters(), lr=config.args["lr"])
    #if config.args['optimizer'] == "Adagrad":



    config.log.info("model = %s" % model)
    config.log.info("config = %s" % config.format_string())

    trainer = Trainer(config)
    evaluator = Evaluator(config)

    """ save model checkpoint """
    def save_checkpoint(epoch):
        checkpoint = {"model_state_dict": model.state_dict(),
                      "config_args" : config.args}
        if config.args["optimizer"] != "YF":  # YF can't save state dict right now
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        checkpoint_file = config.args["model_dir"] + config.args["model_name"] + "-EPOCH%d" % epoch
        troch.save(checkpoint, checkpoint_file)
        config.log.info("saving checkpoint: {}".format(checkpoint_file))


    for epoch in range(1, config.args["max_epoch"] + 1):
        config.log.info("training: epoch %d" % epoch)
        # QS(demi): do i need to return model & optimizer?
        model, optimizer, train_avg_loss, train_answer_dict = trainer.run(model, train_id_conversion[0], train_loader, optimizer, mode="train")
        model, optimizer, dev_avg_loss, dev_answer_dict = trainer.run(model, dev_id_conversion[0], dev_loader, optimizer, mode="dev")

        # loss is a float tensor with size 1
        config.log.info("[EPOCH %d] LOSS = (train)%.5lf | (dev)%.5lf" % (epoch, train_avg_loss[0], test_avg_loss[0]))
        
        answer_filename = "{}/{}-EPOCH{}".format(config.args["model_dir"], config.args["model_name"], epoch)
        config.log.info("[EVAUATION] TRAIN EVAL")
        evaluator.eval("official", train_file, train_answer_dict, "{}.answer.train".format(answer_filename))
        config.log.info("[EVAUATION] DEV EVAL")
        evaluator.eval("official", dev_file, dev_answer_dict, epoch, "{}.answer.dev".format(answer_filename))

        save_checkpoint(epoch)

if __name__ == "__main__":
    main(sys.argv)
