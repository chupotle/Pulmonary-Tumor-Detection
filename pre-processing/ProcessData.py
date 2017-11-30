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
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.misc
from Heatmap import create_heatmap
from CropSamples import crop_samples
from Respace import respace
import pickle

def show_3d_img(img):
    for z in range(img.shape[0]):
        plt.imshow(img[z], cmap='gray')
        plt.show()
        #scipy.misc.imsave('images/{}.jpg'.format(z), img[z])

def plot_3d(image, threshold=0.5):

    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)

    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)
    #ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

# Get rows in data frame associated with each file
def get_filename(fileList, case):
    for f in fileList:
        if case in f:
            return(f)

# Takes a pandas data frame read from the annotations csv file append
# reeturns a list of dictionary containing the nodules and converted coordinates
def change_nodule_coordinate_spacing(nodulesDF, origin, sourceSpacing, targetSpacing):
    nodules=[]
    targetSpacing = [targetSpacing[1], targetSpacing[2], targetSpacing[0]]
    for noduleIndex, row in nodulesDF.iterrows():
        worldCoord = np.array([row["coordX"], row["coordY"], row["coordZ"]])
        voxelCoord = np.rint(np.absolute(worldCoord - origin) / targetSpacing)
        voxelCoord = np.array([int(voxelCoord[2]), int(voxelCoord[0]), int(voxelCoord[1])])

        nodule= {
        'seriesuid' : row['seriesuid'],
        'coords' : voxelCoord,
        }
        #print(row['seriesuid'])
        if 'diameter_mm' in row.index:
            nodule['diameter_mm'] = row['diameter_mm']
        nodules.append(nodule)
    return nodules
def read_image(imageFile):
    # Read the scan
    itkImage = sitk.ReadImage(imageFile)
    npImage = sitk.GetArrayFromImage(itkImage).transpose(0,2,1)
    npZ, height, width = npImage.shape
    npOrigin = np.array(itkImage.GetOrigin())
    npSpacing = itkImage.GetSpacing()
    npSpacing = np.array([npSpacing[2], npSpacing[0], npSpacing[1]])
    return npImage, npSpacing, npOrigin

def normalize_values(npImage):
    maxHU = 400.0
    minHU = -1000.0
    npImage = (npImage-minHU) / (maxHU - minHU)
    npImage[npImage>1] = 1
    npImage[npImage<0] = 0
    npImage = npImage*2 -1
    return npImage

# Get list of image files
inputPath = "datasets/nodules/scans_unprocessed/"
outputPath = "datasets/nodules/scans_processed/"
annotationPath = "datasets/nodules/annotations/"
targetSpacing = np.array([1.25, 1, 1])

# For each subset
for i in range(1,1):
    print("subset" + str(i))
    subsetPath = inputPath + "subset" + str(i) + "/"
    fileList = glob(subsetPath + "*.mhd")

    # Get locations of the nodules
    nodulesDF = pd.read_csv(annotationPath + "annotations.csv")
    nodulesDF["file"] = nodulesDF["seriesuid"].map(lambda fileName: get_filename(fileList, fileName))
    nodulesDF = nodulesDF.dropna()

    # Get locations of candidates
    candidatesDF = pd.read_csv(annotationPath + "candidates.csv")
    candidatesDF["file"] = candidatesDF["seriesuid"].map(lambda fileName: get_filename(fileList, fileName))
    candidatesDF = candidatesDF.dropna()

    samples = []
    # For each image file
    for fileCount, imageFile in enumerate(tqdm(fileList)):
        #if fileCount < 52 : continue
        suid = imageFile.split("/")[-1][:-4]

        print("reading image")
        npImage, npSpacing, npOrigin = read_image(imageFile)

        # Get all nodules and candidates present
        # z x y
        print("resizing nodule coords")
        nodules = change_nodule_coordinate_spacing(nodulesDF[nodulesDF["file"] == imageFile], npOrigin, npSpacing, targetSpacing)
        #print("{} nodules in {}".format(len(nodules), suid))
        #candidates = change_nodule_coordinate_spacing(candidatesDF[candidatesDF["file"] == imageFile], npOrigin, npSpacing, targetSpacing)

        # Rescale the scan
        print("respacing image")
        scan = respace(npImage, npSpacing, targetSpacing)
        print("renormalizing")
        scan = normalize_values(scan)

        # Generate heatmap
        print("generating heatmap")
        heatmap_real = create_heatmap(nodules, scan.shape, candidates=False)
        #heatmap_candidates = create_heatmap(candidates, scan.shape, candidates=True)
        #plot_3d(heatmap_candidates)

        # Generate sample boxes to be cropped at test time
        print("generating crops")
        samples.append(crop_samples(nodules, scan.shape, np.array([64, 96, 96]), 20, 0.8))
        #coords = nodules[0]["coords"]
        #show_3d_img(scan[coords[0]-10:coords[0]+10, coords[1]-50:coords[1]+50,coords[2]-50:coords[2]+50])
        #for sample in samples[suid]:
        #    coords = sample['bounds']
        #    show_3d_img(heatmap_real[coords[0]+50:coords[0]+51, coords[1]:coords[4], coords[2]:coords[5]])
        #    show_3d_img(scan[coords[0]+50:coords[0]+51, coords[1]:coords[4], coords[2]:coords[5]])
        #samples = samples + crop_samples(candidates, scan.shape, np.array([100, 100, 125]), 10, 0.8, candidates="candidates")

        print("saving")
        #save the mask and scan as compressed npz files
        if not os.path.exists(os.path.join(outputPath, "heatmaps", "real")):
            os.makedirs(os.path.join(outputPath, "heatmaps", "real"))
        if not os.path.exists(os.path.join(outputPath, "heatmaps", "candidates")):
            os.makedirs(os.path.join(outputPath, "heatmaps", "candidates"))
        if not os.path.exists(os.path.join(outputPath, "scans_processed")):
            os.makedirs(os.path.join(outputPath, "scans_processed"))
        np.savez_compressed(os.path.join(outputPath, "heatmaps", "real", "%s" % (suid)), a = heatmap_real)
        #np.savez_compressed(os.path.join(outputPath, "heatmaps", "candidates", "%s" % (nodules[0]['seriesuid'])), a = heatmap_candidates)
        np.savez_compressed(os.path.join(outputPath, "scans_processed", "%s" % (suid)), a = scan)

        #save crop boxes
        f = open(os.path.join(outputPath, "crops.pkl"), 'wb')
        print(samples)
        pickle.dump(samples, f)
        f.close()
