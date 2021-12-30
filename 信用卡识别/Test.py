# -*- coding = utf-8 -*-
# @ time : 2021/12/21 16:25
# @Author : momo
# @File : Test.py
# @software: PyCharm


# 导入工具包
from imutils import contours
import numpy as np
import argparse
import cv2
import myutils
import configparser

# 读取配置文件
ap = configparser.ConfigParser()
ap.read("test.ini")

# 初始化参数
predict_card = str(ap.get("image", "image"))  # 读取[image]中的image的值
template = str(ap.get("image", "template"))  # 读取[image]中的template的值

# 绘图展示
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 读取一个模板图像
img = cv2.imread(template)
cv_show('img', img)
