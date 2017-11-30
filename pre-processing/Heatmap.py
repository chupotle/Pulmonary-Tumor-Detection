
import numpy as np


def create_heatmap(nodules, shape, candidates=False):
    maxS = 80
    coordinates = np.mgrid[:maxS, :maxS, :maxS]
    distance_map = ((coordinates[0]-maxS/2)**2 +
                    (coordinates[1]-maxS/2)**2 +
                    (coordinates[2]-maxS/2)**2)

    heat_map_final = np.ndarray([shape[0], shape[1], shape[2]], dtype = np.float32)
    # For each nodule
    for nodule in nodules:
        if candidates:
            diam = 3
        else:
            diam = nodule["diameter_mm"]

        coords = nodule['coords']
        shape = np.array(shape)
        top = np.rint(np.minimum( shape-coords, maxS/2))
        bottom = np.rint(np.minimum( coords, maxS/2 ))

        area = heat_map_final[
            int(coords[0]-bottom[0]) : int(coords[0] + top[0]),
            int(coords[1]-bottom[1]) : int(coords[1] + top[1]),
            int(coords[2]-bottom[2]) : int(coords[2] + top[2])
        ]
        gausian = np.exp(-distance_map / (2*(diam/2) ** 2))
        gausian = gausian[
            int(bottom[0] - maxS/2) : int(top[0] + maxS/2),
            int(bottom[1] - maxS/2) : int(top[1] + maxS/2),
            int(bottom[2] - maxS/2) : int(top[2] + maxS/2)
        ]
        heat_map_final[
            int(coords[0]-bottom[0]) : int(coords[0] + top[0]),
            int(coords[1]-bottom[1]) : int(coords[1] + top[1]),
            int(coords[2]-bottom[2]) : int(coords[2] + top[2])
        ] = np.maximum(area, gausian)

    return heat_map_final
