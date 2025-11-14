import numpy as np
import cv2


def imgDenoising_Median(image, shape):
    imgNew = image.copy()
    #Apply patch to each channel.
    for i in range(0, image.shape[0] - shape, shape):
        for j in range(0, image.shape[1] - shape, shape):
            for k in range(3):
                imgNew[i:i+shape,j:j+shape,k] = medianPatch(imgNew[i:i+shape, j:j+shape, k], shape)
    return imgNew

def medianPatch(img, shape):
    #Create new boundary pixels to handle literal edge-cases.
    tempArray = np.zeros((shape + 2, shape+2))
    tempArray[1:shape + 1,1:shape + 1] = img
    #Change the columns/rows to match
    tempArray[0,:] = tempArray[1,:]
    tempArray[shape+1,:] = tempArray[shape,:]
    tempArray[:,0] = tempArray[:,1]
    tempArray[:,shape+1] = tempArray[:,shape]
    #Calculate new patch based on the median
    newPatch = np.zeros((shape, shape))

    #Fill newpatch by taking shape-shape patches of each image.
    for i in range(shape):
        for j in range(shape):
            newPatch[i,j] = np.median(tempArray[i:i+3,j:j+3])

    return newPatch

def histEq():
    pass


test = np.array(([1,2,3], [4,8,6], [7,5,9]))
print(medianPatch(test, 3))