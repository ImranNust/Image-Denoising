# Imports
import tensorflow as tf
import os




def step_decay(epoch, lr):
    if epoch<30:   #tcw
        #lr = initial_lr/10 #tcw
         lr= lr
    else:
        lr = lr/10
    return lr

lr_schedule = tf.keras.callbacks.LearningRateScheduler(step_decay)

def train(model, X_train, Y_train, X_val, Y_val, save_model,save_frequency,
          batch_size, epochs, std):
    
    if save_model:
        saved_model_dir = os.path.join(os.getcwd(), 'saved_model'+'_STD_'+str(std))
        if not os.path.exists(saved_model_dir):
            os.mkdir(saved_model_dir)
        save_model_callback = tf.keras.callbacks.ModelCheckpoint(filepath = saved_model_dir,
                                                                monitor = 'val_loss',
                                                                verbose = 1,
                                                                save_best_only = True,
                                                                save_weights_only = False,
                                                                save_freq = 'epoch')
        
        history = model.fit(x = X_train, y = Y_train, batch_size = batch_size, 
                            validation_data = (X_val, Y_val), epochs = epochs,
                            callbacks = [lr_schedule, save_model_callback])
        print('Training is complete....')
    else:
        history = model.fit(x = X_train, y = Y_train, batch_size = batch_size, 
                            validation_data = (X_val, Y_val), epochs = epochs)
        print('Training is complete....')
        
    return model, history