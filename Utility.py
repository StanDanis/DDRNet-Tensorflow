import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array
import math

def sort_paths(list_in):

    split = list_in[0].split('\\')
    path = split[0]
    format = f".{split[1].split('.')[1]}"

    sort_images = [int(x.split('\\')[-1].split('.')[0]) for x in list_in]
    sort_images.sort()
    result = [f'{path}\\{num}{format}' for num in sort_images]
    return result

def load_data(path_img, path_mask, format='.png', seed=None, split=False, per_split=[0.7, 0.15, 0.15],
                 one_batch=[False, 32]):
    """ load data as path to imgs and masks,
        can make split to val and train dataset
    """
    images_val, masks_val, image_test, mask_test = [], [], [], []

    if type(format) != list:
        format1 = format
        format2 = format
    else:
        format1 = format[0]
        format2 = format[1]

    # load lists of path to img and mask
    images = sorted([os.path.join(path_img, path) for path in os.listdir(path_img) 
    if path.endswith(format1)], key=os.path.getmtime)
    masks = sorted([os.path.join(path_mask, path) for path in os.listdir(path_mask)
    if path.endswith(format2)], key=os.path.getmtime)

    images = sort_paths(images)
    masks = sort_paths(masks)

    num_sample = len(images)

    if split:

        if (one_batch[0] == True):
            train_samples = num_sample - one_batch[1]
        else:
            train_sample = int(num_sample*per_split[0])
            test_sample =  train_sample+ int(num_sample*per_split[1])
            val_sample = test_sample + int(num_sample*per_split[2])

        # shuffle data
        if seed is not None:
            random.Random(seed).shuffle(images)
            random.Random(seed).shuffle(masks)
    
        images_train = images[0:train_sample]
        masks_train = masks[0:train_sample]

        images_val = images[train_sample:test_sample]
        masks_val = masks[train_sample:test_sample]

        image_test = images[test_sample:]
        mask_test = masks[test_sample:]

        print('--------------------------------------------------')    
        print('Train data: {}\nVal data: {}\nTest data: {}'.format(len(images_train), 
                                                           len(images_val), 
                                                           len(image_test)))
        print('--------------------------------------------------') 

        print('--------------------------------------------------')    
        print('Number of same samples between train, val and test dataset {}'
            .format(len(list(set(images_train).intersection(images_val).intersection(image_test)))))
        print('--------------------------------------------------') 

        print('Return order: images_train, masks_train, images_val, masks_val, image_test, mask_test')

    print('--------------------------------------------------')  
    print('Number of samples from input data: {}'.format(num_sample))
    print('Number of samples after split: {}'.format(len(images_train + images_val + image_test)))
    print('--------------------------------------------------') 

    return images_train, masks_train, images_val, masks_val, image_test, mask_test


def display_images(list_of_paths, title=None, style='default', fonts=15):
    ''' func display images in list (path to image or image as numpy)
      style='column' - make n figures in one column (with one img)
    '''
    # input - only one sample... make list 
    if (type(list_of_paths) != list) : list_of_paths = [list_of_paths]

    # input - image as numpy
    idx = []
    for i in range(len(list_of_paths)): 
        if type(list_of_paths[i]) != str:
            idx.append(i)
  
    num_img = len(list_of_paths)

    numbers = np.linspace(0, 3000, 101, dtype=int)
    helper = np.where(num_img <= numbers)[0][0]
    
    fsize = (13, 33*helper)
    row = 10*helper
    
#   prob. not universal set.
    if style == 'column':
        column = 1
        row = num_img+1
        fsize = (5, 80*helper)
    elif style == '2column':
        column = 2
        fsize = (10, 40*helper)
    else:
        column = 3
        fsize = (13, 33*helper)

    # generate title list (correct dimen.)
    if (title == None): 
        title = ['' for i in range(num_img)] 
    
    elif (type(title) != list):
        title = [title]
        
        if len(title) != num_img:
            title = num_img * title

    else:
        if len(title) != num_img:
            rest = num_img - len(title)
            for i in range(rest):
                title.append(title[i])

    # create figure
    f = plt.figure(figsize=fsize)

    for i, image in enumerate(list_of_paths):

        plt.subplot(row, column, i+1)
        plt.title(title[i], fontsize=fonts)

        if type(image) != str:  
            img = list_of_paths[i]
        else:
            img = mpimg.imread(image) # load image from path (str)

        plt.imshow(img) # show image
        plt.axis('off')
    f.tight_layout()
    plt.show()

def show_prediction(pred_mask, img_val, mask_val, num, img_size, title=['orig_img', 'orig_mask', 'pred_mask']):
    """ func show orig img, orig mask and pred mask in same size as model
        get it
    """
    # input - only one sample... make list 
    if (type(img_val) != list) : img_val = [img_val]
    if (type(mask_val) != list) : mask_val = [mask_val]

    this_display = []

    for i in range(num):
      
        # orig imgs
        orig_img = load_img(img_val[i], target_size=img_size)
        orig_img = img_to_array(orig_img)/255
        this_display.append(orig_img)

        # orig mask
        img = load_img(mask_val[i], target_size=img_size, color_mode="grayscale")
        img = img_to_array(img)/255
        orig_mask = np.around(img, 0)
        orig_mask = img.squeeze(axis=2)
        this_display.append(orig_mask)

        # predi. mask
        mask2 = np.argmax(pred_mask[i], axis=-1)
        this_display.append(mask2)

    # display all
    display_images(this_display, title=title)

def pair_correct(list1, list2, two_in_one=False):
    """ func pait two lists in an alternating style
    """
    result = [None]*(len(list1)+len(list2))
    result[::2] = list1
    result[1::2] = list2

    if two_in_one:
        result = ([[result[i], result[i+1]] for i in range(0, len(result), 2)])

    return result

def load_lable(path):
    data = path.read()
    labels = data.split('\n')[1:-1]
    name_class, color_map = [], []

    for i in labels:
      x = i.split(':')
      name_class.append(x[0])
      color_map.append([int(num) for num in x[1].split(',')])

    return name_class, color_map

def process_mask(rgb_mask, colormap):
    output_mask = []
    rgb_mask = rgb_mask 

    for i, color in enumerate(colormap):
        cmap = np.all(np.equal(rgb_mask, color), axis=-1)
        output_mask.append(cmap)

    output_mask = np.stack(output_mask, axis=-1)
    return output_mask

def get_1d_mask(old_mask_path, new_path_mask, lable_path):
    name_class, color_map = load_lable(lable_path)
    
    for count, filename in enumerate(os.listdir(old_mask_path)):
        
        img = mpimg.imread(f'{old_mask_path}\\{filename}')
        if np.max(np.unique(img)) != 255:
            img = img * 255

        processed_mask = process_mask(img, color_map)
        grayscale_mask = np.argmax(processed_mask, axis=-1)
    #     grayscale_mask = (grayscale_mask / len(color_map)) * 255
        grayscale_mask = np.expand_dims(grayscale_mask, axis=-1)

        name = f'{new_path_mask}\\{filename}'
        plt.imsave(name, grayscale_mask[:,:,0])
        
    return name_class, color_map