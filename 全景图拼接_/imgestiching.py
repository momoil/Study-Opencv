# -*- coding = utf-8 -*-
# @ time : 2021/12/30 9:24
# @Author : momo
# @File : imgestiching.py
# @software: PyCharm

import cv2
from Stitcher import Stitcher

imageA = cv2.imread('imageA.png')
imageB = cv2.imread('imageB.png')

#把图片拼接成全景图
stitcher = Stitcher() #新建一个对象
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

cv2.imshow("imageA", imageA)
cv2.imshow("imageB", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()