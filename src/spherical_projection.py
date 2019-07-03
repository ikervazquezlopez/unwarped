import cv2
import numpy as np
import math

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

def get_interpolated_color(img, y, x):
    dx0 = math.ceil(x) - x
    dy0 = math.ceil(y) - y
    x = int(x)
    y = int(y)
    if x >= img.shape[1]-2 or y >= img.shape[0]-2:
        return img[y,x]


    d0 = math.sqrt(dx0*dx0 + dy0*dy0)
    d1 = 1 - d0

    color = img[y,x] * d0 + img[y+1, x+1] * d1
    return np.uint8(color)

test_filename = "../data/pano4s.png"

img = cv2.imread(test_filename)

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
        X, Y, Z = Rx((X,Y,Z), math.pi*0.5)
        #X, Y, Z = Ry((X,Y,Z), math.pi*0.25)
        #X, Y, Z = Rz((X,Y,Z), math.pi*0.5)

        # Compute the latitude and longitude of the rotated points
        D = math.sqrt(X*X + Y*Y);
        lat = math.atan2(Z, D);
        lng = math.atan2(Y, X);

        # Transform lat long to plane coordinates
        ix = (0.5 * lng / math.pi + 0.5) * img_w - 0.5;
        iy = (lat /math. pi + 0.5) * img_h - 0.5;

        trans[int(y), int((x + img_w / 4) % img_w)] = img[int(iy), int(ix)];
        #trans[int(y), int((x + img_w / 4) % img_w)] = get_interpolated_color(img, iy, ix)

cv2.imshow('img', img)
cv2.imshow('trans', trans)
cv2.waitKey(0)
cv2.destroyAllWindows()
