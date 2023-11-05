from matplotlib import pyplot as plt
import cv2 as cv

def FACES_plot(inputs, resp):
  for ind, input in enumerate(inputs):
    image_name = str(input).strip('./')

    det = resp[str(image_name)]

    image = cv.imread(input)
    w, h, d = image.shape
    for i in det['boxes']:
      print(int(i[0]/612*w), int(i[1]/408*h),  int(i[2]/612*w), int(i[3]/408*h))
      image = cv.rectangle(image, [int(i[0]/612*w), int(i[1]/408*h),  int(i[2]/612*w), int(i[3]/408*h)], color=(0, 0, 255), thickness=2)
    plt.imshow(image[...,::-1])
    plt.show()
