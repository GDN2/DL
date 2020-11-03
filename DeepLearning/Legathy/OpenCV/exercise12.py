import cv2
from matplotlib import pyplot as plt

def showImage():
    imgfile = 'imgs/notebook.jpg'
    img = cv2.imread('imgs/notebook.jpg', cv2.IMREAD_COLOR)

    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

showImage()