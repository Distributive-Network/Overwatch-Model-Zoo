def postprocess(out, labels, outputNames):
		# the post processing is based off of this colab/medium story
		# https://satyajitghana.medium.com/human-pose-estimation-and-quantization-of-pytorch-to-onnx-models-a-detailed-guide-b9c91ddc0d9f
    import re
    import cv2 as cv
    from operator import itemgetter

    out = out[outputNames[0]]
    out = out[0]
    
    _, OUT_HEIGHT, OUT_WIDTH = out.shape
    
    #joint list
    JOINTS = ['0 - r ankle', '1 - r knee', '2 - r hip', '3 - l hip', '4 - l knee', '5 - l ankle', '6 - pelvis', '7 - thorax', '8 - upper neck', '9 - head top', '10 - r wrist', '11 - r elbow', '12 - r shoulder', '13 - l shoulder', '14 - l elbow', '15 - l wrist']
    JOINTS = [re.sub(r'[0-9]+|-', '', joint).strip().replace(' ', '-') for joint in JOINTS]
    POSE_PAIRS = [[9, 8],[8, 7],[7, 6], 
                  [6, 2], [2, 1],[1, 0],
                  [6, 3],[ 3, 4],[4, 5],
                  [7, 12], [12, 11], [11, 10],
                  [7, 13], [13, 14], [14, 15]]
                  
    # this gets the key points and if their confidence values exceed the threshold           
    get_keypoints = lambda pose_layers: map(itemgetter(1, 3), [cv.minMaxLoc(pose_layer) for pose_layer in pose_layers])
    THRESHOLD = 0.8
    OUT_SHAPE = (OUT_HEIGHT, OUT_WIDTH)
    pose_layers = out
    key_points = list(get_keypoints(pose_layers=pose_layers))
    res = {'r-ankle' : None, 'r-knee' : None, 'r-hip' : None, 'l-hip' : None, 'l-knee' : None, 'l-ankle' : None, 'pelvis' : None, 'thorax' : None, 'upper-neck' : None, 'head-top' : None, 'r-wrist' : None, 'r-elbow' : None, 'r-shoulder' : None, 'l-shoulder' : None, 'l-elbow' : None, 'l-wrist' : None}
    
    #returns the final results and rescales them for 256x256 image
    for i in range(len(key_points)):
      res[JOINTS[i]] = ([key_points[i][1][0] * 256 / OUT_SHAPE[0], key_points[i][1][1] * 256 / OUT_SHAPE[1]])
    return res
