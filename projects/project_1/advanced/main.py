from utils import load_dataset_mnist
from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np

load_dataset_mnist()

mndata = MNIST('data_mnist')

images, labels = mndata.load_training()
N = len(images)
images = np.array(images)
images = images.reshape((N, 28, 28))
plt.imshow(images[0])
plt.show()




