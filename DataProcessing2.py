from __future__ import print_function, division
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
#import SimpleITK as sitk
import numpy as np
import os
from glob import glob
#from tqdm import tqdm
import pandas as pd

# Get rows in data frame associated with each file
def getFilename(fileList, case):
    for f in fileList:
        if case in f:
            return(f)

# Get list of image files
inputPath = "/home/nle2/CSC 298/Data/"
outputPath = "/home/nle2/CSC 298/Output1/"

fileList = glob(outputPath + "images_*.npy")

for imageFile in fileList:

    imgs_to_process = np.load(imageFile).astype(np.float64)

    print("on image" + imageFile)

    for i in range(len(imgs_to_process)):
        img = imgs_to_process[i]

        # Standardize the pixel values
        mean = np.mean(img)
        std = np.std(img)
        img = img - mean
        img = img / std

        # Find the average pixel value near the lungs
        # to renormalize washed out images
        middle = img[100:400, 100:400]
        mean = np.mean(middle)
        max = np.max(img)
        min = np.min(img)

        # To improve threshold finding, I'm moving the
        # underflow and overflow on the pixel spectrum
        img[img == max] = mean
        img[img == min] = mean

        # Using Kmeans to separate foreground (radio-opaque tissue)
        # and background (radio transparent tissue ie lungs)
        # Doing this only on the center of the image to avoid
        # the non-tissue parts of the image as much as possible
        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image

        # I found an initial erosion helful for removing graininess from some of the regions
        # and then large dialation is used to make the lung region
        # engulf the vessels and incursions into the lung cavity by
        # radio opaque tissue
        eroded = morphology.erosion(thresh_img, np.ones([4, 4]))
        dilation = morphology.dilation(eroded, np.ones([10, 10]))

        #  Label each region and obtain the region properties
        #  The background region is removed by removing regions
        #  with a bbox that is to large in either dimnsion
        labels = measure.label(dilation)
        label_vals = np.unique(labels)
        regions = measure.regionprops(labels)
        good_labels = []

        for prop in regions:
            B = prop.bbox
            if B[2] - B[0] < 475 and B[3] - B[1] < 475 and B[0] > 40 and B[2] < 472:
                good_labels.append(prop.label)

        mask = np.ndarray([512, 512], dtype=np.int8)
        mask[:] = 0

        for N in good_labels:
            mask = mask + np.where(labels == N, 1, 0)

        mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation
        imgs_to_process[i] = mask

    np.save(imageFile.replace("images", "lungs"), imgs_to_process)
