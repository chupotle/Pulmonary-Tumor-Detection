from __future__ import print_function, division
from skimage import measure
from skimage.transform import resize
#import SimpleITK as sitk
import numpy as np
import os
from glob import glob
#from tqdm import tqdm
import pandas as pd

outputPath = "/home/nle2/CSC 298/Output1/"

#    Here we're applying the masks and cropping and resizing the image
fileList = glob(outputPath + "lungs_*.npy")
out_images = []  # final set of images
out_nodemasks = []  # final set of nodemasks

asd = 1
sad = 1
for fname in fileList:
    print("on file " + fname + ": "+ str(sad))
    sad = sad + 1

    imgs_to_process = np.load(fname.replace("lungs", "images"))
    masks = np.load(fname)
    node_masks = np.load(fname.replace("lungs", "masks"))
    for i in range(len(imgs_to_process)):
        print(str(asd))
        asd = asd+1 
        
        mask = masks[i]
        node_mask = node_masks[i]
        img = imgs_to_process[i]
        new_size = [512, 512]  # we're scaling back up to the original size of the image
        img = mask * img  # apply lung mask
        #
        # renormalizing the masked image (in the mask region)
        #
        new_mean = np.mean(img[mask > 0])
        new_std = np.std(img[mask > 0])
        #
        #  Pulling the background color up to the lower end
        #  of the pixel range for the lungs
        #
        old_min = np.min(img)  # background color
        img[img == old_min] = new_mean - 1.2 * new_std  # resetting backgound color
        img = img - new_mean
        img = img / new_std
        # make image bounding box  (min row, min col, max row, max col)
        labels = measure.label(mask)
        regions = measure.regionprops(labels)
        #
        # Finding the global min and max row over all regions
        #
        min_row = 512
        max_row = 0
        min_col = 512
        max_col = 0
        for prop in regions:
            B = prop.bbox
            if min_row > B[0]:
                min_row = B[0]
            if min_col > B[1]:
                min_col = B[1]
            if max_row < B[2]:
                max_row = B[2]
            if max_col < B[3]:
                max_col = B[3]
        width = max_col - min_col
        height = max_row - min_row
        if width > height:
            max_row = min_row + width
        else:
            max_col = min_col + height
        #
        # cropping the image down to the bounding box for all regions
        # (there's probably an skimage command that can do this in one line)
        #
        img = img[min_row:max_row, min_col:max_col]
        mask = mask[min_row:max_row, min_col:max_col]
        if max_row - min_row < 5 or max_col - min_col < 5:  # skipping all images with no god regions
            pass
        else:
            # moving range to -1 to 1 to accomodate the resize function
            mean = np.mean(img)
            img = img - mean
            min = np.min(img)
            max = np.max(img)
            img = img / (max - min)
            new_img = resize(img, [512, 512])
            new_node_mask = resize(node_mask[min_row:max_row, min_col:max_col], [512, 512])
            out_images.append(new_img)
            out_nodemasks.append(new_node_mask)

num_images = len(out_images)
#
#  Writing out images and masks as 1 channel arrays for input into network
#
final_images = np.ndarray([num_images, 1, 512, 512], dtype=np.float32)
final_masks = np.ndarray([num_images, 1, 512, 512], dtype=np.float32)
for i in range(num_images):
    final_images[i, 0] = out_images[i]
    final_masks[i, 0] = out_nodemasks[i]

rand_i = np.random.choice(range(num_images), size=num_images, replace=False)
test_i = int(0.2 * num_images)
np.save(outputPath + "trainingImages.npy", final_images[rand_i[test_i:]])
np.save(outputPath + "trainingMasks.npy", final_masks[rand_i[test_i:]])
np.save(outputPath + "testImages.npy", final_images[rand_i[:test_i]])
np.save(outputPath + "testMasks.npy", final_masks[rand_i[:test_i]])
