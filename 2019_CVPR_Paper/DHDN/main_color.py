import argparse, os, random, h5py, time, math

import numpy as np

parser = argparse.ArgumentParser(description="PyTorch Densely Connected Hierarchical Network for Image Denoising")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size. Default: 16")
parser.add_argument("--nEpochs", type=int, default=100, help="Number of epochs to train for. Default: 100")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=3,
                    help="Halves the learning rate for every n epochs. Default: n=3")
parser.add_argument("--cuda", action="store_true", help="Use cuda? Default: True")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint for resume. Default: None")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts). Default: 1")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 0")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model. Default: None")
parser.add_argument("--train", default="./data/gaus_train_c_50.h5", type=str, help="training set path.")
parser.add_argument("--valid", default="./data/gaus_val_c_50.h5", type=str, help="validation set path.")
parser.add_argument("--gpu", default='0', help="GPU number to use when training. ex) 0,1 Default: 0")
parser.add_argument("--checkpoint", default="./checkpoint", type=str,
                    help="Checkpoint path. Default: ./checkpoint ")


def main():
    global opt, model
    
    opt = parser.parse_args()
    print(opt)
    
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)

if __name__ == "__main__":
    main()