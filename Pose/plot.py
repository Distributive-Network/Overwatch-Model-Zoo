from matplotlib import pyplot as plt
import cv2 as cv

def Pose_plot(inputs, resp):
  for ind, input in enumerate(inputs):
    image_name = str(input).strip('./')

    det = resp[str(image_name)]

    image = cv.imread(input)
    w, h, d = image.shape
    for i in det:
      x, y = det[i]
      x = int(x/256*h)
      y = int(y/256*w)
      image = cv.circle(image, (x, y), radius = 3, color=(0, 0, 255), thickness=3)

    plt.imshow(image[...,::-1])
    plt.show()
