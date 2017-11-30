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

# For each subset
for i in range(1):
    print("subset" + str(i))
    subsetPath = inputPath + "subset" + str(5) + "/"
    fileList = glob(subsetPath + "*.mhd")

    # Get locations of the nodules
    df = pd.read_csv(inputPath + "annotations.csv")
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

            # For each nodule
            for noduleIndex, row in miniDF.iterrows():
                diam = row["diameter_mm"]

                images = np.ndarray([3, height, width], dtype = np.float32)
                masks = np.ndarray([3, height, width], dtype = np.uint8)

                # Convert world coordinates to voxel coordinates for nodule center
                worldCoord = np.array([row["coordX"], row["coordY"], row["coordZ"]])
                voxelCoord = np.rint((worldCoord - npOrigin) / npSpacing)

                for i, i_z in enumerate(np.arange(int(voxelCoord[2]) - 1, int(voxelCoord[2]) + 2).clip(0, npZ - 1)):

                    mask = np.zeros([height, width])

                    # Defining the voxel range for nodule
                    voxelCoord = (worldCoord - npOrigin) / npSpacing # Convert world coordinates to voxel coordinates for nodule center
                    voxelDiam = int(diam / npSpacing[0] + 5)
                    voxelXmin = np.max([0, int(voxelCoord[0] - voxelDiam) - 5])
                    voxelXmax = np.min([width - 1, int(voxelCoord[0] + voxelDiam) + 5])
                    voxelYmin = np.max([0, int(voxelCoord[1] - voxelDiam) - 5])
                    voxelYmax = np.min([height - 1, int(voxelCoord[1] + voxelDiam) + 5])

                    # Fill in 1 within sphere around nodule
                    for voxelX in range(voxelXmin, voxelXmax + 1):
                        for voxelY in range(voxelYmin, voxelYmax + 1):
                            p_x = npSpacing[0] * voxelX + npOrigin[0]
                            p_y = npSpacing[1] * voxelY + npOrigin[1]
                            if np.linalg.norm(worldCoord - np.array([p_x, p_y, i_z * npSpacing[2] + npOrigin[2]])) <= diam:
                                mask[int((p_y - npOrigin[1]) / npSpacing[1]), int((p_x - npOrigin[0]) / npSpacing[0])] = 1.0

                    masks[i] = mask
                    images[i] = npImage[i_z]

                np.save(os.path.join(outputPath,"images_%04d_%04d.npy" % (fileCount, noduleIndex)), images)
                np.save(os.path.join(outputPath,"masks_%04d_%04d.npy" % (fileCount, noduleIndex)), masks)
