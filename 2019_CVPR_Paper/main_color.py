import argparse, os, random, h5py, time, math
import tensorflow as tf
import numpy as np


from DHDN import DHDN_color


parser = argparse.ArgumentParser(description="PyTorch Densely Connected Hierarchical Network for Image Denoising")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size. Default: 16")
parser.add_argument("--nEpochs", type=int, default=100, help="Number of epochs to train for. Default: 100")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint for resume. Default: None")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs for training. Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model. Default: None")
parser.add_argument("--train", default="./data_processing/gaus_train_c_50_toy.h5", type=str, help="training set path.")
parser.add_argument("--valid", default="./data_processing/gaus_val_c_50_toy.h5", type=str, help="validation set path.")
parser.add_argument("--gpu", default='0', help="GPU number to use when training. ex) 0,1 Default: 0")
parser.add_argument('--input_shape', nargs = '+', default=[64, 64, 3], type=int, help='the size of the input image') 
parser.add_argument('--model_save_path', default = "./saved_models/", type = str, help = 'Path for saving the model.' )
parser.add_argument('--NL', default = 50, type = int, help = 'Noise level: like 30 or 10 or 50' )

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
    
    print('Reading and Preparing the Valiation Data...\n')
    f = h5py.File(opt.valid, 'r')
    input_valid = f['data'][...].transpose(0,2,3,1)
    label_valid = f['label'][...].transpose(0,2,3,1)
    f.close()
    
    print('There are {} training images of sizes: {}\n'.format(input_train.shape[0], input_train.shape[1:]))
    print('There are {} validation images of sizes: {}\n'.format(input_valid.shape[0], input_valid.shape[1:]))

    print('Commencing the Training of the Model...... \n')
    save_path = opt.model_save_path + 'Model_for_NL_' + str(opt.NL)

    save_model_callback = tf.keras.callbacks.ModelCheckpoint(filepath = save_path,
                                                        monitor = 'val_loss',
                                                        verbose = 1,
                                                        save_best_only = True,
                                                        save_weights_only = False,
                                                        save_freq = 'epoch')
    model.fit(x = input_train, y = label_train, 
              validation_data = (input_valid, label_valid),
              batch_size = opt.batchSize,
              epochs = opt.epochs, shuffle = True, callbacks = [save_model_callback])
    
if __name__ == "__main__":
    main()