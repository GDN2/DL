import cv2

def showImage():
    imgfile = 'C:/img_saves/notebook.jpg'
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)

    cv2.namedWindow('title', cv2.WINDOW_NORMAL)
    cv2.imshow('title', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

showImage()