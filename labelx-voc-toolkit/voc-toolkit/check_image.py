import os
import sys
import cv2
import urllib
import cv2
import numpy as np
def readImage_fun(isUrlFlag=None, imagePath=None):
    """
        isUrlFlag == True , then read image from url
        isUrlFlag == False , then read image from local path
    """
    im = None
    if isUrlFlag == True:
        try:
            data = urllib.urlopen(imagePath.strip()).read()
            nparr = np.fromstring(data, np.uint8)
            if nparr.shape[0] < 1:
                im = None
        except:
            im = None
        else:
            try:
                im = cv2.imdecode(nparr, 1)
            except:
                im = None
        finally:
            return im
    else:
        im = cv2.imread(imagePath, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
    if np.shape(im) == ():
        return None
    return im
def main():
    imageDir=""
    for i in sorted(os.listdir(imageDir)):
        image_path = os.path.join(imageDir,i)
        img = readImage_fun(isUrlFlag=False,imagePath=image_path)
        if np.shape(img) == ():
            print(image_path)

if __name__ == '__main__':
    main()
