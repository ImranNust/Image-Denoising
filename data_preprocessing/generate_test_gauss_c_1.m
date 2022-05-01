clear; close all; clc

% path of the testing directory
test_folder_path = 'C:\Users\Imran Qureshi\Desktop\DeepLearning\Image-Denosing\data\test_data\BSDS300\';
savepath = 'BSDS300_Test_C_NL50.h5';
% save_paths

NL = 50; % Noise Level
patch_size = 64;
stride = 64;
noiselevel = 50;


batch_size = 64;


images_list = dir(test_folder_path);
[rows, cols, channels] = size(imread(strcat(test_folder_path,...
    images_list(3).name)));

total_images = floor(cols/batch_size)*floor(rows/batch_size)*length(images_list);
data = zeros(patch_size, patch_size, 3, total_images, 'single');
label = zeros(patch_size, patch_size, 3, total_images, 'single');

count = 0;
count1 = 0;
glob = 0;
n = length(images_list);
%% generate data
for i = 3 : n
    file_origin = strcat(test_folder_path, '\', images_list(i).name);    
    
    origin = imread(file_origin);    
    origin = single(origin)/255;
    [hei, wid, ~] = size(origin);    
    
    for x = 1  : stride : hei-patch_size+1
        for y = 1  :stride : wid-patch_size+1
            
            count=count+1;
            subim_origin = origin(x : x+patch_size-1, y : y+patch_size-1, :);
            if glob
%                 subim_noisy = subim_origin + 5*randi(10)/255*randn(size(subim_origin));
            else
                subim_noisy = subim_origin + noiselevel/255*randn(size(subim_origin));
            end            
            label(:, :, :, count) = subim_origin;
            data(:, :, :, count) = subim_noisy;
        end
    end
    display(100*(i-2)/(n-2));disp('percent complete(training)');
end

data = data(:,:,:,1:total_images);
label = label(:,:,:,1:total_images);


order = randperm(total_images);
data = data(:, :, :, order);
label = label(:, :, :, order);
data(data<0)=0;
data(data>1)=1;

%% writing to HDF5 (Train)
chunksz = 16;
created_flag = false;
totalct = 0;

for batchno = 1:floor(total_images/chunksz)
    batchno;
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,:,last_read+1:last_read+chunksz);
    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz, 'single'); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end

h5disp(savepath);