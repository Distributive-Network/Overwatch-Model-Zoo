from matplotlib import pyplot as plt
import cv2 as cv

def YOLOV5S_plot(inputs, resp, threshold=1):
  for ind, input in enumerate(inputs):
      image_name = str(input).strip('./')
      det = resp[str(image_name)]
      image = cv.imread(input)

      for count, i in enumerate(det):
        if count >= threshold:
          break
        x,y,w,h = det[i]['bbox coordinates']['0'], det[i]['bbox coordinates']['1'], det[i]['bbox coordinates']['2'],  det[i]['bbox coordinates']['3']
        image = cv.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 10)
        cv.putText(image, det[i]['class'], (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

      
      plt.imshow(image[...,::-1])
      plt.show()
