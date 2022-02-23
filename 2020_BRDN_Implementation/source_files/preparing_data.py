import numpy as np
import matplotlib.pyplot as plt
import os
import random
from skimage import io
from skimage.transform import resize

def preprocessing(data, input_shape, noise_factor, mean, std, plot_figure = False):
    
    # making a list containing the path of each image in the data directory
    images_list = os.listdir(data)
    
    # Intialization for original and noisy data (batch_size, rows, cols, channels)
    original_data = np.full(shape = (len(images_list), input_shape[0], input_shape[1], input_shape[2]), fill_value=0, dtype='float32')
    noisy_data = np.full(shape = (len(images_list), input_shape[0], input_shape[1], input_shape[2]), fill_value=0, dtype='float32')

    for i in range(0, len(images_list)):
        path = os.path.join(data, images_list[i])
        img = io.imread(path)
        img = resize(img, (input_shape[0], input_shape[1], input_shape[2])).astype('float32')
        img_noisy = (img + noise_factor * np.random.normal(loc=mean, scale = std, size = img.shape)).astype('float32')
        img_noisy = np.clip(img_noisy, 0., 1.)
        original_data[i,...] = img
        noisy_data[i,...] = img_noisy
    img_num = random.randint(0, len(images_list)-1)
    if plot_figure:
        plt.figure(figsize=(6, 6))
        plt.subplot(121); plt.imshow(original_data[img_num,...]); plt.axis('off'); plt.title('Original Image');
        plt.subplot(122); plt.imshow(noisy_data[img_num,...]);plt.axis('off'); plt.title('Noisy Image');
        plt.tight_layout(); plt.show()
    return original_data, noisy_data


if __name__ == "__main___":
    print('Please import the module and then run')
