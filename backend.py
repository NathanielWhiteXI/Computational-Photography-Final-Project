import numpy as np
import cv2
from scipy.signal import convolve2d 
#Gaussian functions/constants
from math import exp, sqrt, pi


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

    #Fill newpatch by taking shape-by-shape patches of each image.
    for i in range(shape):
        for j in range(shape):
            newPatch[i,j] = np.median(tempArray[i:i+3,j:j+3])

    return newPatch

def imgDenoising_Gaussian(image, shape):
    imgNew = image.copy()
    #Apply patch to each channel.
    for i in range(0, image.shape[0] - shape, shape):
        for j in range(0, image.shape[1] - shape, shape):
            for k in range(3):
                imgNew[i:i+shape,j:j+shape,k] = gaussian_patch(imgNew[i:i+shape, j:j+shape, k], shape)
    return imgNew
#Gaussian function used for gaussian smoothing
def gaussian(sigma, x, y):
    return (1/(sigma*sqrt(2*pi)))*exp(-(((x)**2) + (y)**2)/(2*sigma**2))

def gaussian_patch(patch, shape):
    tempPatch = patch.copy()
    tempPatch = np.array(patch, dtype=np.float32)
    #Converts kernel to float
    tempPatch /=255
    blur = 100000
    sigma = blur / 3
    
    #Create Kernel
    kernel = np.ones((shape, shape))
    for i in range(shape):
        for j in range(shape):
            kernel[i,j] = gaussian(sigma, i - blur, j - blur)
    
    #Normalize Kernel
    kernel /= np.sum(kernel)
    
    newPatch = convolve2d(tempPatch,kernel, mode='valid')
    
    newPatch *= 255
    newPatch = np.array(newPatch, dtype=np.uint8)
    return newPatch

def imgDenoising_Bilateral():
    pass

def histEq(img):
    #Convert to greyscale first.
    img_Grey = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

    #Create probability distribution function
    bins = np.arange(256 + 1)
    imageHist, v = np.histogram(img_Grey, bins) #v is defined so we only get the party we care about for calculations.

    #Create CDF
    pdf = imageHist / np.sum(imageHist)
    cdf = np.cumsum(pdf)

    #Apply cdf to original image, effectively fixing the colours.
    img_RGBA_Equal = (cdf*255)[img].astype(np.uint8)
    
    return img_RGBA_Equal


def claheEnhancement():
    pass

def unsharpMask(img, shape, strength=1):
    #Pseudocode:
    #1: Blur image with Gaussian
    #2: Get the difference by subtacting og image with blur
    #3: Add mask to the image.
    
    #Works in float32 space to avoid errors.
    img_f = img.copy().astype(np.float32)
    
    #Apply Gaussian Blur (which is what the gaussian filter is)
    newMask = imgDenoising_Gaussian(img_f, shape)
    maskDiff = img - newMask
    sharpened = np.clip(img + (strength * maskDiff),0,255).astype(np.uint8)
    #Fixes alpha channel
    sharpened[:,:,3] = 255
    return sharpened


test = np.array(([1,2,3], [4,8,6], [7,5,9]))
print(medianPatch(test, 3))