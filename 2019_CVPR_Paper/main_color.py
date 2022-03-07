import argparse, os, random, h5py, time, math
import tensorflow as tf
import numpy as np
import glob
import scipy.io as sio
from skimage import io
from skimage.transform import resize
from skimage import metrics
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.transform import resize
from PIL import Image
import pandas as pd

from DHDN import DHDN_color


parser = argparse.ArgumentParser(
    description="PyTorch Densely Connected Hierarchical Network for Image Denoising")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size. Default: 16")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=3,
                    help="Halves the learning rate for every n epochs. Default: n=3")
parser.add_argument("--resume", default="", type=str, 
                    help="Path to checkpoint for resume. Default: None")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs for training. Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, 
                    help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, 
                    help="path to pretrained model. Default: None")
parser.add_argument("--train", default="./data_processing/gaus_train_c_10.h5", type=str, 
                    help="training set path.")
parser.add_argument("--valid", default="./data_processing/gaus_val_c_10.h5", type=str, 
                    help="validation set path.")

parser.add_argument("--result_path", default = './Results/', type = str, 
                    help = "Path where results will be stored.")
parser.add_argument("--gpu", default='0', help="GPU number to use when training. ex) 0,1 Default: 0")
parser.add_argument('--input_shape', nargs = '+', default=[64, 64, 3], type=int, 
                    help='the size of the input image') 
parser.add_argument('--model_save_path', default = "./saved_models/", type = str, 
                    help = 'Path for saving the model.' )
parser.add_argument('--NL', default = 10, type = int, help = 'Noise level: like 30 or 10 or 50' )
parser.add_argument('--only_test', default=False, type=bool, 
                    help='If you have trained your model, and you want to test it on test images, then keep this parameter TRUE.')
parser.add_argument('--plot_loss', default=True, type = bool, 
                    help = "If you want to see loss vs epochs then keep it True; otherwise, false.")

######################################################################################################
opt = parser.parse_args()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(opt)

######################################################################################################
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def adjust_learning_rate(epoch):
    lr = opt.lr
    for i in range(epoch // opt.step):
        lr = lr / 2
    return lr

######################################################################################################
def train():
    
    global opt, model
    
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    
    print('Creating the Model.......................\n')
    model = DHDN_color.model_creation(input_shape = opt.input_shape)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=opt.lr),
                  loss = tf.keras.losses.MeanAbsoluteError())
    
    print('Reading and Preparing the Training Data...\n')
    f = h5py.File(opt.train, 'r')
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
    lr_callback = tf.keras.callbacks.LearningRateScheduler(adjust_learning_rate)
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                         patience = 10,
                                                         verbose = 1)
    history = model.fit(x = input_train, y = label_train, 
                        validation_data = (input_valid, label_valid),
                        batch_size = opt.batchSize,
                        epochs = opt.epochs, 
                        shuffle = True, 
                        callbacks = [save_model_callback, lr_callback, earlystop_callback])
    
    return history
    
######################################################################################################   
def test(test_data_dir, test_data_type):
    print('\nStarting the Test... for Noise Level: {}\n'.format(opt.NL))
    path = opt.model_save_path + 'Model_for_NL_' + str(opt.NL)
    test_dir_path =  test_data_dir + test_data_type
    results_path = opt.result_path + test_data_type + 'NL_'+ str(opt.NL) + '/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    print('Loading the Model for NL {} ...\n'.format(opt.NL))
    model = tf.keras.models.load_model(path)
    print('Model for NL {} is LOADED SUCCESSFULLY...\n'.format(opt.NL))
    name = []
    psnr = []
    ssim = []
    print('There are {} images in the given test directory.'.format(len(os.listdir(test_dir_path))))
    images_path = glob.glob('{}/*'.format(test_dir_path))
    for img_num in range(0, len(images_path)):
        img = io.imread(images_path[img_num])
        img = resize(img, (opt.input_shape[0], opt.input_shape[1], opt.input_shape[2])).astype('float32')

        img_noisy = (img + opt.NL/255.0 * np.random.rand(img.shape[0], img.shape[1], img.shape[2]))
        img_noisy = np.clip(img_noisy, 0., 1.)
        img_noisy = np.expand_dims(img_noisy,axis=0)
        img_predict = model.predict(img_noisy)
        img_predict = np.clip(img_predict, 0, 1)
        psnr_noise = metrics.peak_signal_noise_ratio(img, img_noisy[0], data_range=1)
        psnr_denoised = metrics.peak_signal_noise_ratio(img, img_predict[0], data_range=1)
        ssim_noise = metrics.structural_similarity(img, img_noisy[0], multichannel=True, data_range=1)
        ssim_denoised = metrics.structural_similarity(img, img_predict[0], multichannel=True, 
                                                      data_range=1)
        print('Processed Image {} .... '.format(img_num))
        psnr.append(psnr_denoised)
        ssim.append(ssim_denoised)
        name.append(images_path[img_num].split('/')[-1].split('\\')[-1]) 
        im = Image.fromarray((img_noisy[0]*255).astype('uint8'))
        im.save(results_path + 'denoised_image_'+ name[img_num])
    
    average_psnr = np.average(psnr)
    average_ssim = np.average(ssim)
    name.append('Average')
    psnr.append(average_psnr)
    ssim.append(average_ssim)
    print('Average PSNR = {0:.2f}, SSIM = {1:.2f}'.format(average_psnr, average_ssim))
    pd.DataFrame({'name':np.array(name), 'psnr':np.array(psnr), 
                  'ssim':np.array(ssim)}).to_csv(results_path+'/metrics.csv', index=True)

    
######################################################################################################

def plot_loss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(15,7))
    plt.plot(loss, color =  'red', linewidth = 2, visible = True)
    plt.plot(val_loss, color =  'blue', linewidth = 2, visible = True)
    plt.grid(visible = True, which = 'both', color='green', ls = ':' )
    plt.xticks(np.arange(0, len(loss),2), fontweight = 'bold',fontsize = 18, color = 'darkred')
    plt.yticks(fontsize = 18, fontweight = 'bold',color = 'darkred')
    plt.title('Loss vs Epochs', fontsize = 18, fontweight='bold', color='m')
    plt.legend(['loss', 'val_loss'], fontsize = 18)
    plt.show()

######################################################################################################
if __name__ == "__main__":
    if not opt.only_test:
        history = train()
        if opt.plot_loss:
            plot_loss(history)
    else:
        test_data_dir =  '../data/test_data/'
        test_data_type = 'kodak'
        print('Commencing the testing of {} dataset'.format(test_data_type))
        test(test_data_dir, test_data_type)
        print('\nTesting for {} dataset is completed succesfully...\n'.format(test_data_type))
        test_data_type = 'BSDS300'
        print('Commencing the testing of {} dataset'.format(test_data_type))
        test(test_data_dir, test_data_type)
        print('\nTesting for {} dataset is completed succesfully...\n'.format(test_data_type))  