import argparse, os, random, h5py, time, math, logging
import tensorflow as tf
import numpy as np
# import glob
# import scipy.io as sio
# from skimage import io
# from skimage.transform import resize
# from skimage import metrics
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from skimage.transform import resize
# from PIL import Image
# import pandas as pd

from utilities import model, data_processing, plot_loss, test, train


######################################################################################################
parser = argparse.ArgumentParser(description="Residual Based Image Denoising Technique in TensorFlow")
parser.add_argument("--BatchSize", type=int, default=16 , help="Training batch size. Default: 16")
parser.add_argument("--lr", type=float, default=0.0001 , help="Learning Rate. Default=0.001")
parser.add_argument("--step", type=int, default=5 , help="Halves the learning rate for every n epochs. Default: n=5")
# parser.add_argument("--resume", default="", type=str, 
#                     help="Path to checkpoint for resume. Default: None")
parser.add_argument("--epochs", default=50 , type=int, help="Number of epochs for training. Default: 100")
# parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
# parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
# parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model. Default: None")
parser.add_argument("--train_data_path", default='/home/ubuntu/work/Image_Denoising/OurPaper/data/' , type=str, help="Path where the training data is stored.")
parser.add_argument("--valid_data_path", default='/home/ubuntu/work/Image_Denoising/OurPaper/data/' , type=str, help="Path where the validation data is stored.")
parser.add_argument("--test_data_path", default='/home/ubuntu/work/Image_Denoising/OurPaper/data/' , type=str, help="Path where the testing data is stored.")
parser.add_argument("--result_path", default = './results/' , type = str, help = "Path where results will be stored.")
# parser.add_argument("--gpu", default='0', help="GPU number to use when training. ex) 0,1 Default: 0")
parser.add_argument('--input_shape', nargs = '+', default=(64 , 64 , 3) , type=int, help='the size of the input image') 
parser.add_argument('--model_save_path', default = "./saved_models/", type = str, help = 'Path for saving the model.' )
parser.add_argument('--NL', default = 30 , type = int, help = 'Noise level: like 30 or 10 or 50' )
parser.add_argument('--only_test', default=False , type=bool,help='If you have trained your model, and you want to test it on test images, then keep this parameter TRUE.')
parser.add_argument('--plot_loss', default=True , type = bool, help = "If you want to see loss vs epochs then keep it True; otherwise, false.")
parser.add_argument('--BatchReNormalization', default = True, type = bool, help ='True if you want BatchReNormalization and False if you want to use simple BatchNormalization')
parser.add_argument('--train_data_name', default = 'gaus_train_c_30.h5', type = str, help = 'Define the type of the data. For instance, in our case "gaus_train_c_50.h5" contains images corrupted with noise level 10. You can replace 10 with 30 or 50.')
parser.add_argument('--valid_data_name', default = 'gaus_val_c_30.h5', type = str, help = 'Define the type of the data. For instance, in our case "gaus_val_c_10.h5" contains images corrupted with noise level 10. You can replace 10 with 30 or 50.')
parser.add_argument('--test_data_name', default = 'Kodak_Test_C_NL30.h5', type = str, help = 'Define the type of the data. For instance, in our case "Kodak_Test_C_NL10.h5" contains images corrupted with noise level 10. You can replace 10 with 30 or 50.')
######################################################################################################
args = parser.parse_args()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(args)

######################################################################################################
# Finding the number of GPUS
No_of_GPUs = len(tf.config.list_physical_devices('GPU'))
print('We have total {} GPUs'.format(No_of_GPUs))


######################################################################################################    
if __name__ == '__main__':   
    if not args.only_test:
        history = train.train(train_path = args.train_data_path, 
                              valid_path = args.valid_data_path,
                              model_path = args.model_save_path, 
                              train_data_name = args.train_data_name, 
                              valid_data_name = args.valid_data_name,
                              No_of_GPUs = No_of_GPUs, 
                              BatchReNormalization = args.BatchReNormalization, 
                              lr = args.lr, 
                              input_shape = args.input_shape, 
                              BatchSize = args.BatchSize,
                              epochs = args.epochs, 
                              NL = args.NL, 
                              step = args.step)
        if args.plot_loss:
            plot_loss.plot_loss(history)
        
        avg_psnr, avg_ssim = test.test(test_data_path = args.test_data_path, 
                                       NL = args.NL, model_path = args.model_save_path, 
                                       result_path = args.result_path, data_name = args.test_data_name)
    elif args.only_test:
        avg_psnr, avg_ssim = test.test(test_data_path = args.test_data_path, 
                                       NL = args.NL, model_path = args.model_save_path, 
                                       result_path = args.result_path,
                                       data_name = args.test_data_name)
    else:
        print('There is something wrong in your code!!!')