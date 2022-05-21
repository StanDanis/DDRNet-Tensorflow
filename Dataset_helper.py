import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from  tensorflow import keras
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras import layers
import random


class CreateDataset(keras.utils.Sequence):
    """ generate dataset with batch size .... as sequence dataset 
    """

    def __init__(self, img_paths, mask_paths, img_size, batch_size, shuffle=True):
        # initialization
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # return Sequence length
        # warning: with one "/" python make floating point division
        return len(self.img_paths) // self.batch_size

    def on_epoch_end(self):
        # shuffle data every epoch
        if (self.shuffle == True):
            rand_helper = list(zip(self.img_paths, self.mask_paths))
            random.shuffle(rand_helper)

            self.img_paths, self.mask_paths = zip(*rand_helper)
        else:
            pass

    def __getitem__(self, idx):
        # Returns preprocessed data (image, mask)
        i = idx * self.batch_size
        
        img_paths_b = self.img_paths[i : i + self.batch_size]
        mask_paths_b = self.mask_paths[i : i + self.batch_size]

        # create output array for batch
        imgs_out = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        masks_out = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        
        # load data to array
        for j, path in enumerate(img_paths_b):
            # load img
            img = load_img(path, target_size=self.img_size, color_mode='rgb')
            # normalize photo
            img = img_to_array(img)/255
            imgs_out[j] = img
        
        # load masks
        for j, path in enumerate(mask_paths_b):
            # load image with one channel (greyscale)
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            # change image to array and normalize img
            img = (img_to_array(img)/255)*12
            # change range mask to 0-1
            masks_out[j] = np.around(img, 0).astype('uint8')

        return imgs_out, masks_out
