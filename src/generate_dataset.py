import cv2
import numpy as np
import random
import math
import sys
from os.path import isfile, isdir, join
from os import listdir

EXTENSIONS = (".png", ".jpg")

def _rotate_x(p, angle):
    alpha = angle #* (math.pi / 180)
    x, y, z = p
    X = x
    Y = y*math.cos(alpha) - z*math.sin(alpha)
    Z = y*math.sin(alpha) + z*math.cos(alpha)
    return (X,Y,Z)

def _rotate_y(p, angle):
    alpha = angle #* (math.pi / 180)
    x, y, z = p
    X = x*math.cos(alpha) + z*math.sin(alpha)
    Y = y
    Z = -x*math.sin(alpha) + z*math.cos(alpha)
    return (X,Y,Z)

def _rotate_z(p, angle):
    alpha = angle #* (math.pi / 180)
    x, y, z = p
    X = x*math.cos(alpha) - y*math.sin(alpha)
    Y = x*math.sin(alpha) + y*math.cos(alpha)
    Z = z
    return (X,Y,Z)




def rotate_projection(img, a_x, a_y, a_z):
    img_w = img.shape[1]
    img_h = img.shape[0]

    rotated = np.zeros_like(img)

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

            # Perform sphere rotations
            X, Y, Z = _rotate_x((X,Y,Z), a_x)
            X, Y, Z = _rotate_y((X,Y,Z), a_y)
            X, Y, Z = _rotate_z((X,Y,Z), a_z)

            # Compute the latitude and longitude of the rotated points
            D = math.sqrt(X*X + Y*Y);
            lat = math.atan2(Z, D);
            lng = math.atan2(Y, X);

            # Transform lat long to plane coordinates
            ix = (0.5 * lng / math.pi + 0.5) * img_w - 0.5;
            iy = (lat /math. pi + 0.5) * img_h - 0.5;

            rotated[int(y), int((x + img_w / 4) % img_w)] = img[int(iy), int(ix)];

    return rotated

def generate_random():
    return (2 * random.random()) - 1.0



if len(sys.argv) != 3:
  print("Incorrect number of arguments!")
  print("Usage: python generate_dataset.py in_dir_path out_dir_path")
  exit("Terminated")

if not isdir(sys.argv[1]):
    exit("in_dir_path is not valid!")
if not isdir(sys.argv[2]):
    exit("out_dir_path is not valid!")

in_dir_path = sys.argv[1]
out_dir_path = sys.argv[2]

files = [f for f in listdir(sys.argv[1]) if isfile(join(sys.argv[1], f))]

f = open(out_dir_path + "info.csv", 'w')
f.write("FILENAME,ROTATION_X,ROTATION_Y,ROTATION_Z")

for name in files:
    if name.endswith(EXTENSIONS):
        filename = "{}/{}".format(in_dir_path, name)

        img = cv2.imread(filename)
        img = cv2.resize(img, (1000,500))

        a_x = math.pi * generate_random()
        a_y = math.pi * generate_random()
        a_z = math.pi * generate_random()

        img = rotate_projection(img, a_x, a_y, a_z)

        cv2.imwrite("{}{}".format(out_dir_path, name), img)
        f.write("\n{},{},{},{}".format(name, str(a_x), str(a_y), str(a_z)))

f.close()
