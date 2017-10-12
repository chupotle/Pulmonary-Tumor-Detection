from __future__ import print_function, division
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
import SimpleITK as sitk
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import scipy.misc

def show_3d_img(img):
    for z in range(img.shape[0]):
        #plt.imshow(img[z], cmap='gray')
        #plt.show()
        scipy.misc.imsave('images/{}.jpg'.format(z), img[z])

# Get rows in data frame associated with each file
def getFilename(fileList, case):
    for f in fileList:
        if case in f:
            return(f)

# Get list of image files
inputPath = "datasets/nodules/scans/"
outputPath = "datasets/nodules/heatmaps/"
annotationPath = "datasets/nodules/annotations/"

# For each subset
for i in range(1):
    print("subset" + str(i))
    subsetPath = inputPath + "subset" + str(1) + "/"
    fileList = glob(subsetPath + "*.mhd")
    print(fileList)

    # Get locations of the nodules
    df = pd.read_csv(annotationPath + "annotations.csv")
    df["file"] = df["seriesuid"].map(lambda fileName: getFilename(fileList, fileName))
    df = df.dropna()

    # For each image file
    for fileCount, imageFile in enumerate(tqdm(fileList)):
        # Get all nodules present
        miniDF = df[df["file"] == imageFile]

        # If file has at least 1 nodule
        if miniDF.shape[0] > 0:
            # Extract coordinates
            itkImage = sitk.ReadImage(imageFile)
            npImage = sitk.GetArrayFromImage(itkImage)
            npZ, height, width = npImage.shape
            npOrigin = np.array(itkImage.GetOrigin())
            npSpacing = np.array(itkImage.GetSpacing())

            coordinates = np.mgrid[:npZ, :height, :width]
            heat_map_final = np.ndarray([npZ, height, width], dtype = np.float32)
            # For each nodule
            for noduleIndex, row in miniDF.iterrows():
                diam = row["diameter_mm"] # TODO: convert these to pixel values

                # Convert world coordinates to voxel coordinates for nodule center
                worldCoord = np.array([row["coordX"], row["coordY"], row["coordZ"]])
                voxelCoord = np.rint((worldCoord - npOrigin) / npSpacing)

                # Create heat map
                distance_map = ((coordinates[0]-voxelCoord[2])**2 +
                                (coordinates[1]-voxelCoord[1])**2 +
                                (coordinates[2]-voxelCoord[0])**2)

                heat_map_final = heat_map_final + np.exp(-distance_map / (2 * diam ** 2)) # TODO: change this to max rather than addition

            #np.save(os.path.join(outputPath,"images_%04d.npy" % (fileCount, noduleIndex)), images)
            np.save(os.path.join(outputPath,"masks_%04d.npy" % (fileCount)), heat_map_final)
