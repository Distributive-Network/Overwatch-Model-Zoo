#IMPORT PYODIDE PACKAGES 
import numpy as np
import cv2 as cv

#DEFINE FUNCTION 
#Must be called as preprocess(bytes, inputNames) 
def preprocess(bytes, inputNames):

    #DECLARE FEEDS DICTIONOARY 
    feeds = dict()
    inputNames = str(inputNames[0])

    #Load image and transform to 28x28 greyscale
    bytesInput = np.frombuffer( bytes, dtype=np.uint8)
    image = cv.imdecode(bytesInput, cv.IMREAD_COLOR)
    image   = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = image / 255
    image = cv.resize(image, (28,28))
    image = np.reshape(np.asarray(image.astype(np.float32)), (1, 1, 28, 28))
    
    #ADD INPUT TO FEEDS DICTIONARY
    feeds[inputNames] = image
    
    #RETURN FEEDS 
    return feeds
