import cv2

def showImage():
    imgfile = 'imgs/notebook.jpg'
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)

    imgfile2 = 'imgs/1.png'
    img2 = cv2.imread(imgfile2, cv2.IMREAD_COLOR)

    cv2.imshow('title', img)
    cv2.imshow('title2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

showImage()