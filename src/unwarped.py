import cv2
import math
import numpy as np

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





# MAIN METHOD

test_filename = "../data/pano4s.png"

img = cv2.imread(test_filename)
trans = rotate_projection(img, 0, math.pi*0.5, 0)

cv2.imshow('img', img)
cv2.imshow('trans', trans)
cv2.waitKey(0)
cv2.destroyAllWindows()
