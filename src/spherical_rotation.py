import cv2
import numpy as np
import math
import csv
from os.path import isfile, join, isdir
import sys

TO_RADIANS = math.pi / 180
filename_csv = "../data/data.csv"


def Rx(p, angle):
    alpha = angle #* (math.pi / 180)
    x, y, z = p
    X = x
    Y = y*math.cos(alpha) - z*math.sin(alpha)
    Z = y*math.sin(alpha) + z*math.cos(alpha)
    return (X,Y,Z)

def Ry(p, angle):
    alpha = angle #* (math.pi / 180)
    x, y, z = p
    X = x*math.cos(alpha) + z*math.sin(alpha)
    Y = y
    Z = -x*math.sin(alpha) + z*math.cos(alpha)
    return (X,Y,Z)

def Rz(p, angle):
    alpha = angle #* (math.pi / 180)
    x, y, z = p
    X = x*math.cos(alpha) - y*math.sin(alpha)
    Y = x*math.sin(alpha) + y*math.cos(alpha)
    Z = z
    return (X,Y,Z)


def get_subpixel(img, y, x):
    patch = cv2.getRectSubPix(img, (1,1), (x,y))
    return patch[0][0]

def spherical_rotation(img, rx, ry, rz):
    img_w = img.shape[1]
    img_h = img.shape[0]

    trans = np.zeros_like(img)

    for x in range(0,img_w):
        for y in range(0,img_h):
            # Transform x and y to -pi<long<pi and -pi/2<lat<pi/2 respectively
            xx = 2 * (x + 0.5) / img_w - 1.0;
            yy = 2 * (y + 0.5) / img_h - 1.0;
            lng = math.pi * xx;
            lat = 0.5 * math.pi * yy;

            # Compute cartesian coordinates of the sphere surface
            X = math.cos(lat) * math.cos(lng + math.pi*0.5);
            Y = math.cos(lat) * math.sin(lng + math.pi*0.5);
            Z = math.sin(lat);

            # Perform rotations
            X, Y, Z = Rx((X,Y,Z), float(rx)*TO_RADIANS)
            #X, Y, Z = Ry((X,Y,Z), float(ry)*TO_RADIANS)
            X, Y, Z = Rz((X,Y,Z), float(rz)*TO_RADIANS)

            # Compute the latitude and longitude of the rotated points
            D = math.sqrt(X*X + Y*Y);
            lat = math.atan2(Z, D);
            lng = math.atan2(Y, X);

            # Transform lat long to plane coordinates
            ix = (0.5 * lng / math.pi + 0.5) * img_w - 0.5;
            iy = (lat /math. pi + 0.5) * img_h - 0.5;

            trans[int(y), int((x + img_w / 4) % img_w)] = get_subpixel(img, iy, ix)
    return trans



if __name__ == '__main__':

    filename = sys.argv[1]

    rx = sys.argv[2]
    ry = sys.argv[3]
    rz = sys.argv[4]

    in_dir = sys.argv[5]
    out_dir = sys.argv[6]

    img = cv2.imread(join(in_dir, filename))
    trans = spherical_rotation(img, rx, ry, rz)

    # Save image
    name = filename.split(".")[0]
    filename = "{}_{}_{}_{}.png".format(name,rx, ry, rz)
    cv2.imwrite(join(in_dir, filename), trans)
