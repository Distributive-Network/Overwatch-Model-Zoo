from matplotlib import pyplot as plt
import cv2 as cv

def deepface_plot(inputs, resp):
  for ind, input in enumerate(inputs):
    image_name = str(input).strip('./')

    det = resp[str(image_name)]

    image = cv.imread(input)

    for (name, face) in det.items():
        box = face['facial_area']
        landmarks = face['landmarks']

        faces = image[box[1]:box[3],box[0]:box[2],:].astype(np.float32) / 255.

        if faces.shape[0] == 0 or faces.shape[1] == 0:
            continue
        kernel_width = faces.shape[0]//2
        kernel_height = faces.shape[1]//2
        if kernel_width % 2 == 0:
            kernel_width -=1
        if kernel_height %2 ==0:
            kernel_height-=1

        finFace = cv.GaussianBlur(faces, (kernel_height,kernel_width),sigmaX=100,sigmaY=100, borderType = cv.BORDER_DEFAULT)
        image[box[1]:box[3],box[0]:box[2],:] = (finFace * 255.).astype(np.uint8)

        for (landmark_type, landmark) in landmarks.items():
            landmark = [ int(l) for l in landmark ]
            image = cv.circle(image, (landmark[0], landmark[1]), radius = 2, color=(180, 140, 210), thickness=-1)

    plt.imshow(image[...,::-1])
    plt.show()
