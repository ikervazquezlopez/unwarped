import cv2
import numpy as np
import math
import csv
from os.path import isfile, join, isdir
import sys

TO_RADIANS = math.pi / 180
epsilon = 0.9

############## PROJECTION TO LATITUDE AND LONGITUDE ###########################
def equirectangular2latlng(x, y,  img_w, img_h, lat0=0, lng0=0):
    w = img_w
    h = img_h
    xx = 2*math.pi*x/w - math.pi
    yy = math.pi*y/h - math.pi/2
    lng = xx/math.cos(lat0) + lng0
    lat = yy + lat0
    return (lat, lng)

def sinusoidal2latlng(x, y, lng0, img_w, img_h):
    w = img_w
    h = img_h
    xx = 2*math.pi*x/w - math.pi
    lat = math.pi*y/h - math.pi/2
    lng = xx / math.cos(lat) + lng0
    return (lat, lng)

def cylindrical2latlng(x, y, img_w, img_h, lng0=0):
    w = img_w
    h = img_h
    xx = 2*math.pi*x/w - math.pi
    yy = math.pi*y/h - math.pi/2
    lat = math.atan(yy)
    lng = xx + lng0
    return (lat, lng)

def image2latlong(x, y, img_w, img_h):
    lng = x*2*math.pi/img_w - math.pi
    lat = y*math.pi/img_h - math.pi/2
    return (lat, lng)

"""
def cartesian2latlng(x, y, z):
    D = math.sqrt(x*x + y*y)
    lat = math.atan2(z, D)
    lng = math.atan2(y, x)
    return (lat, lng)
"""



############## LATITUDE AND LONGITUDE TO PROJECTION ###########################
def latlng2equirectangular(lat, lng, img_w, img_h, lat0=0, lng0=0):
    w = img_w
    h = img_h
    xx = (lng-lng0)*math.cos(lat0)
    yy = lat-lat0
    x = w * (xx+math.pi) / (2*math.pi)
    y = h * (yy+math.pi/2) / math.pi
    return (x, y)

def latlng2sinusoidal(lat,lng, img_w, img_h, lng0=0):
    w = img_w
    h = img_h
    xx = (lng-lng0)*math.cos(lat)
    yy = lat
    x = w * (xx+math.pi) / (2*math.pi)
    y = h * (yy+math.pi/2) / math.pi
    return (x, y)

def latlng2cylindrical(lat, lng, img_w, img_h, lng0=0):
    w = img_w
    h = img_h
    xx = lng - lng0
    yy = math.atan2(lat,xx)
    x = w * (xx+math.pi) / (2*math.pi)
    y = h * (yy+math.pi/2) / math.pi
    print((lat,lng), (x,y))
    return (x, y)

"""
def latlng2cartesian(lat, lng):
    x = math.cos(lat) * math.cos(lng + math.pi*0.5)
    y = math.cos(lat) * math.sin(lng + math.pi*0.5)
    z = math.sin(lat)
    return (x, y, z)
"""




############## PROJECTION TO PROJECTION ########################################
def equirectangular2sinusoidal(equi_img):
    img = np.zeros_like(equi_img)
    img_h, img_w, _ = img.shape
    lng0 = 0

    for x in range(0,img_w):
        for y in range(0,img_h):
            d = abs((img_w/2) * math.cos(math.pi*(y/img_h - 0.5)))
            thresh_low = int(img_w/2 - d)
            thresh_high = int(img_w/2+d)
            if x < thresh_low or x > thresh_high: # Limit the sinusoidal shape
                continue
            lat, lng = sinusoidal2latlng(x, y, lng0, img_w, img_h)
            xs, ys = latlng2equirectangular(lat, lng, img_w, img_h)
            if xs-math.floor(xs)==0 and ys-math.floor(ys)==0:
                img[y,x] = equi_img[int(ys), int(xs)]
            else:
                img[y, x] = cv2.getRectSubPix(equi_img, (1,1), (xs,ys))
    return img


def sinusoidal2equirectangular(sin_img):
    img = np.zeros_like(sin_img)
    img_h, img_w, _ = img.shape
    lng0 = 0

    for x in range(0,img_w):
        for y in range(0,img_h):
            lat, lng = equirectangular2latlng(x, y, img_w, img_h)
            xs, ys = latlng2sinusoidal(lat, lng, img_w, img_h)
            if xs-math.floor(xs)==0 and ys-math.floor(ys)==0:
                img[y,x] = sin_img[int(ys), int(xs)]
            else:
                img[y, x] = cv2.getRectSubPix(sin_img, (1,1), (xs,ys))
    return img

def equirectangular2cylindrical(equi_img):
    img = np.zeros_like(equi_img)
    img_h, img_w, _ = img.shape
    for x in range(0,img_w):
        for y in range(0,img_h):
            lat, lng = cylindrical2latlng(x, y, img_w, img_h)
            xs, ys = latlng2equirectangular(lat, lng, img_w, img_h)
            if xs-math.floor(xs)==0 and ys-math.floor(ys)==0:
                img[y,x] = equi_img[int(ys), int(xs)]
            else:
                img[y, x] = cv2.getRectSubPix(equi_img, (1,1), (xs,ys))
    return img

def cylindrical2equirectangular(cyl_img):
    img = np.zeros_like(cyl_img)
    img_h, img_w, _ = img.shape
    for x in range(0,img_w):
        for y in range(0,img_h):
            lat, lng = equirectangular2latlng(x, y, img_w, img_h)
            xs, ys = latlng2cylindrical(lat, lng, img_w, img_h)
            if xs-math.floor(xs)==0 and ys-math.floor(ys)==0:
                img[y,x] = cyl_img[int(ys), int(xs)]
            else:
                img[y, x] = cv2.getRectSubPix(cyl_img, (1,1), (xs,ys))
    return img




ref = cv2.imread("../data/pano4s.png")
#ref = cv2.imread("equirectangular.png")
#ref = cv2.imread("sinusoidal.png")
#img = np.zeros((img_h,img_w), dtype = np.uint8)
img = np.zeros_like(ref)
img_h, img_w, _ = img.shape
img = equirectangular2cylindrical(ref)
img = cylindrical2equirectangular(img)

cv2.imshow("test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()




"""
w = 16
h = 8

table = np.zeros((h, w, 2), dtype=np.float64)

for y in range(h):
    for x in range(w):
        lat, lng = equirectangular2latlng(x, y, w-1, h-1)
        xs, ys = latlng2sinusoidal(lat,lng, w-1,h-1)
        table[y, x] = np.array([xs,ys])

from tabulate import tabulate
table = np.round(table, decimals=2)
headers = [y in range(h)]
table = tabulate(table, headers)
print(table)
"""
