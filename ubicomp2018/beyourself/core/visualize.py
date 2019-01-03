import matplotlib.pyplot as plt
import numpy as np
from ..settings import *



def visualize(data, prediction, groundtruth = None, outPath = None):
    """
    visualize the prediction of one single run
    """
    assert len(data.shape) == 2 and len(prediction.shape) == 1
    assert data.shape[0] == prediction.shape[0]

    # Repeat since prediction is for each chunk
    pred = np.repeat(prediction, WIN, axis =0)

    data = np.reshape(data,(-1,N_SENSOR))

    if groundtruth is None:
        fig, ax = plt.subplots(nrows = 2, sharex = True, figsize=(20,5))
        ax[0].plot(data[:,0], label='proximity')
        ax[0].legend()

        ax[1].plot(pred,'r', label='predictions')
        ax[1].legend()
        ax[1].set_ylim([0,N_CLASS_CHEWING_OTHERS - 1 + 0.2])
    else:

        # Repeat since prediction is for each chunk
        groundtruth = np.repeat(groundtruth, WIN, axis =0)

        fig, ax = plt.subplots(nrows = 3, sharex = True, figsize=(20,5))
        ax[0].plot(data[:,0], label='proximity')
        ax[0].legend()

        ax[1].plot(pred,'r', label='predictions')
        ax[1].legend()
        ax[1].set_ylim([0,N_CLASS_CHEWING_OTHERS - 1 + 0.2])

        ax[2].plot(groundtruth,'r', label='groundTruth')
        ax[2].legend()
        ax[2].set_ylim([0,N_CLASS_CHEWING_OTHERS - 1 + 0.2])

    if outPath:
        # filename, _ = os.path.splitext(outpath)
        # outfile = filename + '_' + str(b) + '.png'
        # print outPath
        plt.savefig(outPath)
        plt.close()
    else:
        return fig
