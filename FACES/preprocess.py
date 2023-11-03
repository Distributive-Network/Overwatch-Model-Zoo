import numpy as np
import cv2 as cv

def preprocess(bytes, inputNames):
    feeds      = dict()
    inputNames = str(inputNames[0])
    
    bytesIn    = np.frombuffer( bytes, dtype= np.uint8)
    npBytes    = cv.imdecode(bytesIn, cv.IMREAD_COLOR)
    imgMean    = np.array([127, 127, 127])
    image      = cv.cvtColor(npBytes, cv.COLOR_BGR2RGB)
    image      = cv.resize(image, (320, 240))
    image      = (image - imgMean) / 128
    image      = np.transpose(image, [2, 0, 1])
    image      = np.expand_dims(image, axis=0)
    image      = image.astype(np.float32)
    image      = np.ascontiguousarray(image)
    
    feeds[inputNames] = image
    return feeds
