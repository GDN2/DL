import cv2
import matplotlib.pyplot as plt

def showImage():
    imgfile = 'imgs/notebook.jpg'
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)

    b, g, r = cv2.split(img)
    img2 = cv2.merge([r,g,b])

    plt.imshow(img2)
    plt.xticks([])
    plt.yticks([])
    plt.show()
showImage()