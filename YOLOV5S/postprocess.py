import cv2 as cv
import numpy as np

def postprocess(out, labels, outputNames):
    out = out[outputNames[0]]
    out = out[0]
    
    #object list
    objects = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',  'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave','oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    refined_output = []
    #get rid of all out puts that don't meet the 0.2 threshold
    for i in range(len(out)):
        if out[i,4] > 0.2:
            refined_output.append(np.array(np.append(out[i,:5], np.argmax(out[i, 5:]))))
            
    #take the top 5 most confident estimations and return them
    # I tried using non maximum suppression here but it wasn't working so well
    # it was letting too many guesses through       
    k = 5
    refined_output = np.array(refined_output)
    res = {}
    
    if(len(refined_output) > 0):
        ind = np.argsort( -refined_output[:, 4] )
        refined_output = refined_output[ind]
        for i in range(k):
            res["prediction " + str(i)] = {"class" : objects[np.argmax(np.float32(refined_output[i, 5:]))], "conf" : float(refined_output[i, 4]), "bbox coordinates": np.int16(refined_output[i, :4])}
    return res
