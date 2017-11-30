import numpy as np
import matplotlib.pyplot as plt

#creates a list of number boxes for crops of shape cropShape where a percent of
#them have at leas one nodule in them and the rest have no nodules
def crop_samples(nodules, imageShape, cropShape, number, percentContainingNodules, candidates="real"):
    if np.isin(True, cropShape>imageShape):
        print("Crop shape is larger than the image shape")
        return []
    #if len(nodules) == 0: return []

    #creates a mask where
    # 0 represents an invalid box center
    # 1 represents a box center that contains at least one nodule
    # 2 represents a box center that contains no nodules
    validMask = np.zeros(imageShape)
    validMask[
        cropShape[0]/2:imageShape[0]-cropShape[0]/2,
        cropShape[1]/2:imageShape[1]-cropShape[1]/2,
        cropShape[2]/2:imageShape[2]-cropShape[2]/2,
    ] = 1
    nodule_coords = []

    for nodule in nodules:
        coords = nodule['coords']
        validMask[
        int(max(cropShape[0]/2, coords[0]-cropShape[0]/2)) : int(min(imageShape[0]-cropShape[0]/2, coords[0]+cropShape[0]/2)),
        int(max(cropShape[1]/2, coords[1]-cropShape[1]/2)) : int(min(imageShape[1]-cropShape[1]/2, coords[1]+cropShape[1]/2)),
        int(max(cropShape[2]/2, coords[2]-cropShape[2]/2)) : int(min(imageShape[2]-cropShape[2]/2, coords[2]+cropShape[2]/2))
        ] = 2
        nodule_coords = nodule_coords + [{"coords" : nodule["coords"], "diameter" : nodule["diameter_mm"]}]

    #get a list of arguments for earch category
    nonblank = np.argwhere(validMask == 2)
    blank = np.argwhere(validMask == 1)


    # randomly choose which box centers to take
    numNodules = int(percentContainingNodules * number)
    numblank = number-numNodules
    if numNodules < nonblank.shape[0]:
        nonblank = np.take(nonblank, np.random.randint(0, high=nonblank.shape[0], size=numNodules), axis=0)
    if numblank < blank.shape[0]:
        blank = np.take(blank, np.random.randint(0, high=blank.shape[0], size=numblank), axis=0)

    #create a box around the box centers
    nonblank = np.concatenate((nonblank - cropShape/2, nonblank + cropShape/2), axis=1).tolist()
    blank = np.concatenate((blank - cropShape/2, blank + cropShape/2), axis=1).tolist()

    samples = []
    for s in nonblank:
        samples.append({'suid': nodules[0]['seriesuid'], 'has_nodules' : True, 'real' : True, "bounds": s, "nodules" : nodule_coords})
    for s in blank:
        samples.append({'suid': nodules[0]['seriesuid'], 'has_nodules' : False, 'real' : True, "bounds": s, "nodules" : nodule_coords})
    return samples
