#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import cv2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def dist_thresholding(des1, des2, threshold_value) -> list:
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=100)
    good = []
    for m in matches:
        od = []
        for i in m:
            if i.distance < threshold_value:
                od.append(i)
        good.append(od)
    return good


def nn(des1, des2, threshold_value) -> list:
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=1)
    good = []
    for m in matches:
        if threshold_value == -1 or m[0].distance < threshold_value:
            good.append([m[0]])
        else:
            good.append([])
    return good


def nndr(des1, des2, threshold_value) -> list:
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m in matches:
        if m[0].distance < threshold_value * m[1].distance:
            good.append([m[0]])
        else:
            good.append([])
    return good

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# vim:set et sw=4 ts=4:
