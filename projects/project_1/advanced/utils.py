import os
import requests
import gzip
import shutil
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_dataset_mnist():
    print("-------> Downloading MNIST dataset")
    download_files = ["http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                      "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                      "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                      "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"]

    if "data_mnist" not in os.listdir():
        os.mkdir("data_mnist")

    for file_url in download_files:
        file_name = file_url.split("/")[-1]
        uncompressed_file_name = "".join(file_name.split(".")[:-1])
        if uncompressed_file_name in os.listdir("./data_mnist"):
            continue
        abs_path = os.path.abspath("./data_mnist")
        with open(abs_path + "/" + file_name, "wb") as f:
            r = requests.get(file_url)
            f.write(r.content)
        with gzip.open(abs_path + "/" + file_name, 'rb') as f_in:
            with open(abs_path + "/" + uncompressed_file_name, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.system("rm -rf " + file_name)

    print("-------> Finish")


def plot_image(image):
    image = np.array(image)
    image = image.reshape((28, 28))
    plt.imshow(image)
    plt.show()


def one_hot_encoding(y):
    one_hot = OneHotEncoder()
    y = np.array(y)
    y = y.reshape((-1, 1))
    return one_hot.fit_transform(y).toarray()


def preprocess_data(X, y, nn=False):
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    if nn:
        X = X.reshape((-1, 28, 28, 1))

    y = one_hot_encoding(y)
    return X, y