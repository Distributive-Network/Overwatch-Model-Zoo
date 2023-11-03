import numpy as np
import cv2 as cv

def postProcess(out, labels, outputNames):
    boxOutput       = out[outputNames[1]][0]
    scoreOutput     = out[outputNames[0]][0]
    pickedBoxProbs  = [];
    pickedLabels    = [];
    threshold       = 0.7;
    iouThresh       = 0.5;
    width           = 612;
    height          = 408;
    for i in range(1, scoreOutput.shape[1]):
        probs   = scoreOutput[:, i]
        mask    = probs > threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subsetBoxes = boxOutput[mask, :]
        boxProbs = np.concatenate([subsetBoxes, probs.reshape(-1, 1)], axis=1)
        boxProbs = nms(boxProbs,
           iouThresh = iouThresh,
           top_k = -1,
           )
        pickedBoxProbs.append(boxProbs)
        pickedLabels.extend([i] * boxProbs.shape[0])
    if not pickedBoxProbs:
        return {"boxes": [], "scores": [], "labels": []}
    pickedBoxProbs = np.concatenate(pickedBoxProbs)
    pickedBoxProbs[:, 0] *= width
    pickedBoxProbs[:, 1] *= height
    pickedBoxProbs[:, 2] *= width
    pickedBoxProbs[:, 3] *= height
    postReturn = {
      "boxes": pickedBoxProbs[:,:4].tolist(),
      "scores": pickedBoxProbs[:,4].tolist(),
      "labels": np.array(pickedLabels).tolist(),
    }
    return postReturn

def areaOf(left_top, right_bottom):
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = areaOf(overlap_left_top, overlap_right_bottom)
    area0 = areaOf(boxes0[..., :2], boxes0[..., 2:])
    area1 = areaOf(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def nms(box_scores, iouThresh, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iouThresh]

    return box_scores[picked, :]
