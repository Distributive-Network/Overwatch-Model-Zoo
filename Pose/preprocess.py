import numpy as np
import cv2 as cv

ratio = None
dw, dh = None, None

def letterbox(im, new_shape=(256, 256), 
              color=(114, 114, 114), auto=True, 
              scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv.resize(im, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def preprocess(bytes, inputNames):
    global ratio, dw, dh
    feeds      = dict()
    inputName = str(inputNames[0])
		
		# load the image
    bytesIn    = np.frombuffer( bytes, dtype= np.uint8)
    img0       = cv.imdecode(bytesIn, cv.IMREAD_COLOR)
    img, ratio, (dw, dh)        = letterbox(img0, (256, 256), stride = 32, auto=False, scaleFill=False)
    img = img.astype(np.float32)
    
    # normalize the image according to these colour means and standard deviations
    MEAN = 255 * np.array([0.485, 0.456, 0.406])
    STD = 255 * np.array([0.229, 0.224, 0.225])
    img = (img - MEAN[None, None, :]) / STD[None, None, :] 
    
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img.astype(np.float32))[None]
    feeds[inputName] = img
    
    return feeds
