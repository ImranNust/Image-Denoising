clear; close all; clc

% path of the testing directory
test_folder_path = 'C:\Users\Imran Qureshi\Desktop\DeepLearning\Image-Denosing\data\test_data\kodak\';

% save_paths
kodak_GD = 'Kodak_GD_C.mat';
kodak_noisy = 'Noisy_kodak_NL50_C.mat';

NL = 50; % Noise Level

images_list = dir(test_folder_path);
[rows, cols, channels] = size(imread(strcat(test_folder_path,...
    images_list(3).name)));
data_val = zeros(rows, cols, 3, 24, 'single');
label_val = zeros(rows, cols, 3, 24, 'single');

count1 = 0;
no_of_images = length(images_list)-2;
for i = 3:no_of_images+2
    file_origin = strcat(test_folder_path, images_list(i).name);
    origin = imread(file_origin);
    origin = single(origin)/255;
    count1 = count1 + 1;
    if size(origin,1)>size(origin,2)
        origin=imrotate(origin,90);
    end
    label_val(:,:,:, count1) = origin;
    data_val(:,:,:,count1) = origin + NL/255*randn(size(origin));

    disp(['processing image ', num2str(i-2), ' out of ', ...
        num2str(no_of_images), ' images...'])
end

% Clipping the values between 0 and 1
data_val(data_val<0)=0;
data_val(data_val>1)=1;