import argparse
import logging
import os, glob
import tensorflow as tf

from source_files import create_model, preparing_data, train, test, plotting_loss


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=8, type=int, help='batch size') 
parser.add_argument('--input_shape', nargs = '+', default=[128, 128, 3], type=int, help='the size of the input image') 
parser.add_argument('--train_dir', default='C:\\Users\\Imran Qureshi\\Desktop\\DeepLearning\\ImageDenoising\\data\\DIV2K\\DIV2K_train_HR\\', type=str, help='path of train data') 
parser.add_argument('--test_dir', default='C:\\Users\\Imran Qureshi\\Desktop\\DeepLearning\\ImageDenoising\\kodak\\', type=str, help='directory of test dataset')
parser.add_argument('--val_dir', default = 'C:\\Users\\Imran Qureshi\\Desktop\\DeepLearning\\ImageDenoising\\data\\DIV2K\\DIV2K_valid_HR\\', type=str, help='directory of the validation data')
parser.add_argument('--std',default = 0.01, type = float,help='standard deviation for Gaussian noise [0, 1]')
parser.add_argument('--mean',default = 0, type = float,help='Mean for Gaussian noise [0, 1]')
parser.add_argument('--noise_factor',default = 1, type = float,help='Scaling factor for noise')
parser.add_argument('--plot_figure',default = False, type = bool,help='If you want to see original and noisy image, then make it TRUE, otherwise False')
parser.add_argument('--epochs', default=50, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_frequency', default=5, type=int, help='save model at every x epoches') #every x epoches save model
parser.add_argument('--save_model', default = True, type = str, help = 'Model is saved if TRUE')
parser.add_argument('--pretrain', default=False, type=bool, help='Resume train if Ture; otherwise, start the fresh training')
parser.add_argument('--only_test', default=False, type=bool, help='train and test or only test')
parser.add_argument('--plot_loss', default=True, type = bool, help = "If you want to see loss vs epochs then keep it True; otherwise, false.")
args = parser.parse_args()




def start_training():
    
    print('\nPreparing the Training Data\n')
    X_train, Y_train = preparing_data.preprocessing(args.train_dir, args.input_shape, args.noise_factor, 
                                                    args.mean, args.std,args.plot_figure)
    print('\nThere are {} training images...\n'.format(X_train.shape[0]))
    print('\nTraining Data is Prepared...\n')
    # print('Training Data is Prepared, now preparing the test data')
    # X_test, Y_test = preparing_data.preprocessing(args.test_dir, args.input_shape, args.noise_factor, 
                                                  # args.mean, args.std, args.plot_figure)
    # print('Test Data is Prepared, now preparing the validation data')
    print('\nPreparing the Validation Data\n')
    X_val, Y_val = preparing_data.preprocessing(args.val_dir, args.input_shape, args.noise_factor, 
                                                args.mean, args.std, args.plot_figure)
    print('\nThere are {} validation images...\n'.format(X_val.shape[0]))
    print('\nValidation Data is Prepared, now preparing and compiling the model\n')
    model = create_model.BRDNet(args.input_shape, args.lr)
    
    print('\nModel is created and compiled, now starting the training...\n')
    
    
    model, history = train.train(model=model, X_train = X_train, Y_train = Y_train, 
                           X_val = X_val, Y_val = Y_val, save_model = args.save_model,
                           save_frequency = args.save_frequency, batch_size = args.batch_size,
                           epochs = args.epochs, std = args.std)
    
    return model, history

def start_testing():
    model_path = 'C:\\Users\\Imran Qureshi\\Desktop\\DeepLearning\\ImageDenoising\\2020_BRDN' + '\\saved_model'+'_STD_'+str(args.std)
    model = tf.keras.models.load_model(filepath = model_path, compile = False)
    test.testing(model, args.test_dir, std = args.std, mean = args.mean, input_shape = args.input_shape,
                noise_factor = args.noise_factor)
    return


if __name__ == "__main__":
    if not args.only_test:
        model, history = start_training()
        if args.plot_loss:
            plotting_loss.plot_loss(history)
        start_testing()
    else:
        start_testing()

