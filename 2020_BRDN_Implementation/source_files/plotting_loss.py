import matplotlib.pyplot as plt
import numpy as np


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