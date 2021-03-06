 clear; close all; clc
%% Train Set
train_folder_origin = 'C:\Users\Imran Qureshi\Desktop\DeepLearning\Image-Denosing\data\train_data\DIV2K\DIV2K_train_HR';
val_folder_origin = 'C:\Users\Imran Qureshi\Desktop\DeepLearning\Image-Denosing\data\train_data\DIV2K\DIV2K_valid_HR';
savepath = 'gaus_train_c_30.h5';
savepath_val = 'gaus_val_c_30.h5';

patch_size = 64;
stride = 64;
noiselevel = 30;

patch_size_v = 64;
stride_v = 64;
batch_size = 64;

% Number of images you want to retain for training
train_imgs = 300000;
valid_imgs = round((train_imgs*12.5)/100);

data = zeros(patch_size, patch_size, 3, train_imgs, 'single');
label = zeros(patch_size, patch_size, 3, train_imgs, 'single');
data_val = zeros(patch_size_v, patch_size_v, 3, valid_imgs, 'single');
label_val = zeros(patch_size_v, patch_size_v, 3, valid_imgs, 'single');

count = 0;
count1 = 0;
glob = 0;

list_origin = dir(train_folder_origin);
n = length(list_origin);

%% generate data
for i = 3 : n
    file_origin = strcat(train_folder_origin, '\', list_origin(i).name);    
    
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

%% validation data
list_origin = dir(val_folder_origin);
n = length(list_origin);
i=randi(n-2)+2;
file_origin = strcat(val_folder_origin, '\', list_origin(i).name);

origin = imread(file_origin);
origin = single(origin)/255;
[hei, wid, c] = size(origin);

for x = 1  : stride_v : hei-patch_size_v+1
    for y = 1  :stride_v : wid-patch_size_v+1
        
        count1=count1+1;
        subim_origin = origin(x : x+patch_size_v-1, y : y+patch_size_v-1, :);
        if glob
%             subim_noisy = subim_origin + 5*randi(10)/255*randn(size(subim_origin));
        else
            subim_noisy = subim_origin + noiselevel/255*randn(size(subim_origin));
        end
        label_val(:, :, :, count1) = subim_origin;
        data_val(:, :, :, count1) = subim_noisy;
    end
end

data = data(:,:,:,1:train_imgs);
label = label(:,:,:,1:train_imgs);
data_val = data_val(:,:,:,1:valid_imgs);
label_val = label_val(:,:,:,1:valid_imgs);

order = randperm(train_imgs);
data = data(:, :, :, order);
label = label(:, :, :, order);
order = randperm(valid_imgs);
data_val = data_val(:, :, :, order);
label_val = label_val(:, :, :, order);

data(data<0)=0;
data(data>1)=1;
data_val(data_val<0)=0;
data_val(data_val>1)=1;
%% writing to HDF5 (Train)
chunksz = 16;
created_flag = false;
totalct = 0;

for batchno = 1:floor(train_imgs/chunksz)
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

%% writing to HDF5 (Val.)
chunksz = valid_imgs;
created_flag = false;
totalct = 0;

batchno=1;
batchdata = data_val;
batchlabs = label_val;
startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
curr_dat_sz = store2hdf5(savepath_val, batchdata, batchlabs, ~created_flag, startloc, chunksz, 'single'); 
created_flag = true;
totalct = curr_dat_sz(end);

h5disp(savepath_val); 