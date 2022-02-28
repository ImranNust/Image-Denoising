import argparse, os, random, h5py, time, math
import tensorflow as tf
import numpy as np


from DHDN import DHDN_color


parser = argparse.ArgumentParser(description="PyTorch Densely Connected Hierarchical Network for Image Denoising")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size. Default: 16")
parser.add_argument("--nEpochs", type=int, default=100, help="Number of epochs to train for. Default: 100")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=3,
                    help="Halves the learning rate for every n epochs. Default: n=3")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint for resume. Default: None")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs for training. Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model. Default: None")
parser.add_argument("--train", default="../data_processing/gaus_train_c_50.h5", type=str, help="training set path.")
parser.add_argument("--valid", default="./data_processing/gaus_val_c_50.h5", type=str, help="validation set path.")
parser.add_argument("--gpu", default='0', help="GPU number to use when training. ex) 0,1 Default: 0")
parser.add_argument("--checkpoint", default="./checkpoint", type=str,
                    help="Checkpoint path. Default: ./checkpoint ")
parser.add_argument('--input_shape', nargs = '+', default=[64, 64, 3], type=int, help='the size of the input image') 



def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    global opt, model
    
    opt = parser.parse_args()
    print(opt)
    
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    
    print('Creating the Model.......................\n')
    model = DHDN_color.model_creation(input_shape = opt.input_shape)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=opt.lr),
                  loss = tf.keras.losses.MeanAbsoluteError())
    
    print('Reading and Preparing the Training Data...\n')
    f = h5py.File(opt.valid, 'r')
    input_train = f['data'][...].transpose(0,2,3,1)
    label_train = f['label'][...].transpose(0,2,3,1)
    f.close()
    print('There are {} training images of sizes: {}\n'.format(input_train.shape[0], input_train.shape[1:]))

    print('Commencing the Training of the Model...... \n')
    model.fit(x = input_train, y = label_train, batch_size = opt.batchSize,
             epochs = opt.epochs, shuffle = True)
    
if __name__ == "__main__":
    main()