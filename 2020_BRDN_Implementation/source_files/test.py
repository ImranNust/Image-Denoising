import tensorflow as tf
import glob
import pandas as pd
import os
import numpy as np
from skimage import io
from skimage.transform import resize
from skimage import metrics
from PIL import Image



def testing(model, test_dir, std, mean, noise_factor, input_shape):
    results_path = os.path.join(os.getcwd(), 'results_for_std_'+str(std)+'\\')
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    print('\nStarting the Test... for Standard Deviation: {}\n'.format(std))
    name = []
    psnr = []
    ssim = []
    images_path = glob.glob('{}/*'.format(test_dir))
    for img_num in range(0, len(images_path)):
        img = io.imread(images_path[img_num])
        img = resize(img, (input_shape[0], input_shape[1], input_shape[2])).astype('float32')
        np.random.seed(0)
        img_noisy = (img + noise_factor * np.random.normal(loc=mean, scale = std, 
                                                           size = img.shape)).astype('float32')
        img_noisy = np.clip(img_noisy, 0., 1.)
        img_noisy = np.expand_dims(img_noisy,axis=0)
        img_predict = model.predict(img_noisy)
        img_predict = np.clip(img_predict, 0, 1)
        psnr_noise = metrics.peak_signal_noise_ratio(img, img_noisy[0], data_range=1)
        psnr_denoised = metrics.peak_signal_noise_ratio(img, img_predict[0], data_range=1)
        ssim_noise = metrics.structural_similarity(img, img_noisy[0], multichannel=True, data_range=1)
        ssim_denoised = metrics.structural_similarity(img, img_predict[0], multichannel=True, 
                                                      data_range=1)
        psnr.append(psnr_denoised)
        ssim.append(ssim_denoised)
        name.append(images_path[img_num].split('/')[-1].split('\\')[-1]) 
        im = Image.fromarray((img_noisy[0]*255).astype('uint8'))
        im.save(results_path+'test_image_'+ name[img_num])
        im = Image.fromarray((img_predict[0]*255).astype('uint8'))
        im.save(results_path + 'denoised_image_'+ name[img_num])
    
    average_psnr = np.average(psnr)
    average_ssim = np.average(ssim)
    name.append('Average')
    psnr.append(average_psnr)
    ssim.append(average_ssim)
    print('Average PSNR = {0:.2f}, SSIM = {1:.2f}'.format(average_psnr, average_ssim))
    pd.DataFrame({'name':np.array(name), 
                  'psnr':np.array(psnr), 
                  'ssim':np.array(ssim)}).to_csv(results_path+'/metrics.csv',
                                                 index=True)
    return