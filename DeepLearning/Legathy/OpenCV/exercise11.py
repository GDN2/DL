import cv2

def showImage():
    imgfile = 'C:/img_saves/notebook.jpg'
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
    cv2.imshow('title', img)

    k = cv2.waitKey(0) & 0xFF

    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('c'):
        cv2.imwrite('C:/img_saves/notebook_copy.jpg', img)
        cv2.destroyAllWindows()

showImage()
