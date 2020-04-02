import numpy as np
from urllib import request
import gzip
import pickle
import os.path as osp, os

filename = [
    ["training_images","train-images-idx3-ubyte.gz"],
    ["test_images","t10k-images-idx3-ubyte.gz"],
    ["training_labels","train-labels-idx1-ubyte.gz"],
    ["test_labels","t10k-labels-idx1-ubyte.gz"]
    ]

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        if osp.isfile(name[1]):
            print('Found {0}, not downloading'.format(name[1]))
            continue
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28,28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)

    for key in mnist:
        if key.endswith('images'):
            np.savez(key, img=mnist[key])
        else:
            np.savez(key, label=mnist[key])

def init():
    if not osp.isdir('mnistdata'): os.makedirs('mnistdata')
    _ = os.getcwd()
    try:
        os.chdir('mnistdata')
        download_mnist()
        save_mnist()
    finally:
        os.chdir(_)


if __name__ == '__main__':
    init()
