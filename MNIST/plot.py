from matplotlib import pyplot as plt
import cv2 as cv

def MNIST_plot(inputs, resp):
  for ind, input in enumerate(inputs):
      image_name = str(input).strip('./')

      det = resp[str(image_name)]

      image = cv.imread(input)
      plt.subplot(121)
      plt.imshow(image[...,::-1])
      plt.title("Original Image")
      # disabling xticks by Setting xticks to an empty list
      plt.xticks([])  
      
      # disabling yticks by setting yticks to an empty list
      plt.yticks([])  
      
      plt.subplot(122)
      
      y=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
      
      # getting values against each value of y
      x= det['output'][0]
      plt.barh(y, x)
      
      # setting label of y-axis
      plt.ylabel("Number")
      
      # setting label of x-axis
      plt.xlabel("Probability") 
      plt.title("Prediction")
      
      plt.suptitle("Original Images and Predictions")

      #plt.imshow(image[...,::-1])
      plt.show()
