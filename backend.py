import numpy as np
import cv2


def imgDenoising_Median(image, halfWidth):
    pass

def medianPatch(img, shape):
    #Create new boundary pixels to handle literal edge-cases.
    tempArray = np.zeros((shape + 2, shape+2))
    tempArray[1:shape + 1,1:shape + 1] = img
    #Change the columns/rows to match

    return tempArray

def histEq():
    pass


test = np.array(([1,2,3], [4,8,6], [7,5,9]))
print(medianPatch(test, 3))