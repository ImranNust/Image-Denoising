import tensorflow as tf
import os
import pandas as pd
import numpy as np
from skimage import metrics
from utilities.data_processing import data_processing

def test(test_data_path, NL, model_path, result_path, data_name):
    print('\nStarting the Test... for Noise Level: {}\n'.format(NL))
    path = model_path + 'Model_for_NL_' + str(NL)
    results_path = result_path + 'NL_'+ str(NL) + '/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    ###############################################################################################
    # Reading the Test Data
    input_test, label_test = data_processing(data_path=test_data_path, data_name=data_name)
    print('There are {} testing images of sizes: {}\n'.format(input_test.shape[0], input_test.shape[1:]))
    ###############################################################################################
    print('Loading the Model for NL {} ...\n'.format(NL))
    model = tf.keras.models.load_model(path)
    print('Model for NL {} is LOADED SUCCESSFULLY...\n'.format(NL))
    ###############################################################################################
    img_predict = model.predict(input_test)
    img_predict = np.clip(img_predict, 0, 1)
    psnr_noise = metrics.peak_signal_noise_ratio(input_test, label_test, data_range=1)
    psnr_denoised = metrics.peak_signal_noise_ratio(img_predict, label_test, data_range=1)
    ssim_noise = metrics.structural_similarity(input_test, label_test, multichannel=True, data_range=1)
    ssim_denoised = metrics.structural_similarity(img_predict, label_test, multichannel=True, data_range=1)
    
    print('Average PSNR = {0:.2f}, SSIM = {1:.2f}'.format(psnr_denoised, ssim_denoised))
    result = [psnr_denoised, ssim_denoised]
    pd.DataFrame({'Results':result},
                 index = ['Avg PSNR', 'Avg SSIM']).to_csv(results_path + '/metrics.csv',
                                                          index = True)
    return psnr_denoised, ssim_denoised