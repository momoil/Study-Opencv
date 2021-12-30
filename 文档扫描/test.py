# -*- coding = utf-8 -*-
# @ time : 2021/12/23 13:55
# @Author : momo
# @File : test.py
# @software: PyCharm

#OCR工具包
from PIL import Image
import pytesseract
import cv2
import os

preprocess = "bLur" #thresh

image = cv2.imread('scan.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if preprocess == "thresh":
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
if preprocess == "blur":
    gray = cv2.mediqnBlur(gray, 3)

filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

text = pytesseract.image_to_string(Image.open(filename))
print(text)
os.remove(filename)

cv2.imshow("Image", image)
cv2.imshow("output", gray)
cv2.waitKey(0)
