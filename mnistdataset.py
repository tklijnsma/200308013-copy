import os.path as osp, glob, numpy as np, sys
import torch
from torch_geometric.data import (Data, Dataset)

class MNISTDataset(Dataset):
    """
    Stores processed data only in memory
    """
    def __init__(self, root, train, transform=None):
        super(MNISTDataset, self).__init__(root, transform)
        self.is_train = train
        self.is_loaded = False
        self.image_size = 28
        self.x_oneimage = np.tile(np.arange(self.image_size), self.image_size)
        self.y_oneimage = np.repeat(np.arange(self.image_size-1,-1,-1), self.image_size)
        self.graphs = []
        self.use_only_nonzero = True
        self.num_classes = 10

    @property
    def raw_file_names(self):
        if not hasattr(self,'input_files'):
            self.input_files = glob.glob(osp.join(self.raw_dir, '*.npz'))
        return [ osp.basename(f) for f in self.input_files ]

    @property
    def processed_file_names(self):
        return []

    def process(self):
        pass

    def load(self):
        if self.is_loaded: return
        key = 'train' if self.is_train else 'test'
        images_file = [f for f in self.raw_paths if (key in f and 'image' in f)][0]
        labels_file = [f for f in self.raw_paths if (key in f and 'label' in f)][0]
        print('Processing {0} images file {1}'.format(key, images_file))
        print('Processing {0} labels file {1}'.format(key, labels_file))
        with np.load(images_file) as npdata:
            images = npdata['img']
            n_images = images.shape[0]
            values = np.reshape(images, (n_images, self.image_size**2))
            x = np.repeat(self.x_oneimage.reshape((1,self.x_oneimage.shape[0])), n_images, axis=0)
            y = np.repeat(self.y_oneimage.reshape((1,self.y_oneimage.shape[0])), n_images, axis=0)
            assert values.shape == (n_images, self.image_size**2)
            assert x.shape == (n_images, self.image_size**2)
            assert y.shape == (n_images, self.image_size**2)
            # features = np.stack((x, y, values), axis=-1)
            # assert features.shape == (n_images, values.shape[1], 3)
        with np.load(labels_file) as npdata:
            labels = npdata['label'].reshape((n_images, 1))
            assert labels.shape == (n_images, 1)
        # Put all graphs as Data objects in memory
        for i_img in range(n_images):
            x_thisimg = x[i_img]
            assert x_thisimg.shape == (self.image_size**2,)
            y_thisimg = y[i_img]
            values_thisimg = values[i_img]
            label = labels[i_img]
            assert label.shape == (1,)
            if self.use_only_nonzero:
                nonzero_indices = np.nonzero(values_thisimg)
                x_thisimg = x_thisimg[nonzero_indices]
                y_thisimg = y_thisimg[nonzero_indices]
                values_thisimg = values_thisimg[nonzero_indices]
            # features = np.stack((x_thisimg, y_thisimg, values_thisimg), axis=-1)
            # assert features.shape == (x_thisimg.shape[0], 3)
            pos = np.stack((x_thisimg, y_thisimg), axis=-1)
            assert pos.shape == (x_thisimg.shape[0], 2)
            values_thisimg = np.reshape(values_thisimg, (values_thisimg.shape[0], 1))
            assert values_thisimg.shape == (x_thisimg.shape[0], 1)
            self.graphs.append(Data(
                pos = torch.from_numpy(pos).type(torch.FloatTensor),
                x = torch.from_numpy(values_thisimg).type(torch.FloatTensor),
                y = torch.from_numpy(label).type(torch.LongTensor)
                ))
        self.is_loaded = True

    def __len__(self):
        self.load()
        return len(self.graphs)

    def get(self, i):
        self.load()
        return self.graphs[i]
