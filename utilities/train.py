from utilities.model import DRDNet
import tensorflow as tf
from utilities.data_processing import data_processing
import numpy as np

def train(train_path, valid_path, model_path,
          train_data_name, valid_data_name,
          No_of_GPUs, BatchReNormalization,
          lr, input_shape, BatchSize,epochs, NL, step):
    ######################################################################################################
    ## Setting the Callbacks
    def adjust_learning_rate(epoch, lr):
        print('The current epoch is {}, and the learning rate is {}'.format(epoch, lr))
        lr = round(lr, 5)
        if epoch%step == 0 and epoch !=0:
            lr = round(lr * np.exp(-0.35), 5)
        print('The new learning rate is {}'.format(lr))
        return lr
    
    save_path = model_path + 'Model_for_NL_' + str(NL)
    save_model_callback = tf.keras.callbacks.ModelCheckpoint(filepath = save_path,
                                                             monitor = 'val_loss',
                                                             verbose = 1,
                                                             save_best_only = True,
                                                             save_weights_only = False,
                                                             save_freq = 'epoch')
    lr_callback = tf.keras.callbacks.LearningRateScheduler(adjust_learning_rate)
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                          patience = 50,
                                                          verbose = 1)
    # Reading images
    input_train, label_train = data_processing(data_path = train_path, data_name = train_data_name)
    input_train, label_train = input_train[0:60000], label_train[0:60000]
    print('There are {} TRAINING images of sizes: {}\n'.format(input_train.shape[0], input_train.shape[1:]))
    input_valid, label_valid = data_processing(data_path = valid_path, data_name = valid_data_name)
    input_valid, label_valid = input_valid[1:int(np.ceil(60000*12.5/100))], label_valid[1:int(np.ceil(60000*12.5/100))]
    print('There are {} VALIDATION images of sizes: {}\n'.format(input_valid.shape[0], input_valid.shape[1:]))
    # Instantiating the Model
    if No_of_GPUs>1 and (not BatchReNormalization):
        print('The number of GPUS are more than 1 and BatchReNormalization is {}; Therefore, deploying distributed training across all {} GPUs'.format(BatchReNormalization, No_of_GPUs))
        strategy = tf.distribute.MirroredStrategy()
        # print("Number of devices: {}".format(strategy.num_replicas_in_sync))
        # Open a strategy scope.
        with strategy.scope():
            mymodel = DRDNet(input_shape = input_shape)
            mymodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr),
                                loss = tf.keras.losses.MeanSquaredError())
        BatchSize = BatchSize * No_of_GPUs
        print('Since there are {} GPUS, the new BatchSize is {}'.format(No_of_GPUs, BatchSize))
        history = mymodel.fit(x=input_train, y = label_train,
                              epochs=epochs, verbose=1, validation_data=(input_valid, label_valid),
                              batch_size = BatchSize, 
                              callbacks = [lr_callback, earlystop_callback, save_model_callback])
    elif No_of_GPUs<=1 or BatchReNormalization:
        print('The number of GPUS are less than 1 or BatchReNormalization is {}; Therefore, deploying distributed training across 1 GPUs'.format(BatchReNormalization))
        mymodel = DRDNet(input_shape = input_shape, BatchReNormalization = BatchReNormalization)
        mymodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr),
                            loss = tf.keras.losses.MeanSquaredError())
        history = mymodel.fit(x=input_train, y = label_train,
                              epochs=epochs, verbose=1, validation_data=(input_valid, label_valid),
                              batch_size = BatchSize, 
                              callbacks = [lr_callback, earlystop_callback, save_model_callback])
    else:
        print('Something not good in your code!!!')

    print('Training is Scuccessfully Completed......')
    return history