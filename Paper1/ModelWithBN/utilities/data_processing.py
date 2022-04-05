import h5py, os

def data_processing(data_path, data_name):
    print('Reading and Preparing the Data...\n')
    path = os.path.join(data_path, data_name)
    f = h5py.File(path, 'r')
    input_imgs = f['data'][...].transpose(0,2,3,1)
    label_imgs = f['label'][...].transpose(0,2,3,1)
    f.close()
    return input_imgs, label_imgs