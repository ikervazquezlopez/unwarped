import cv2
import numpy as np
import math

img_w = 800
img_h = 400

phi_0 = 180
lambda_0 = 360

# Map image coordinates to 360x180 coordinates (Wikipedia assumes the image is that way)
def _image_to_360(x, y):
    x_p = x * (360/img_w)
    y_p = y * (180/img_h)
    return {'x': x_p, 'y': y_p}

# Reverse projection as in Wikipedia for Equirectangular projection
def _reverse_projection(x, y):
    lambda_p = x / (math.cos(phi_0)) + lambda_0
    phi_p = y + phi_0
    return {'long': lambda_p, 'lat': phi_p}

# Perform the translation
def _translation_sphere(long, lat, t_long, t_lat):
    #new_long = (long + t_long) #% 360
    #new_lat = (lat + t_lat) % 180
    new_long = (long*math.cos(t_long) - lat*math.sin(t_long))  % 360
    new_lat = (long*math.sin(t_long) + y*math.cos(t_long)) % 180
    return {'long': new_long, 'lat': new_lat}

# Forward projection as in Wikipedia for Equirectangular projection
def _forward_projection(long, lat):
    x = (long-lambda_0) * math.cos(phi_0)
    y = lat - phi_0
    return {'x': x, 'y': y}

# Map 360x180 coordinates to image
def _360_to_image(long, lat):
    x = long * (img_w/360)
    y = lat * (img_h/180)
    return {'x': int(x), 'y': int(y)}

def to_radians(angle):
    r = angle * (math.pi/180)
    return r


test_filename = "../data/pano4s.png"

img = cv2.imread(test_filename)
img = cv2.resize(img,(360,180))

# Compute central point of the image in latlong
#central_latlong = cartesian2latlong(img_w/2,img_h/2)
#phi_0 = central_latlong['lat']
#lambda_0 = central_latlong['long']

img_w = img.shape[1]
img_h = img.shape[0]

trans = np.zeros_like(img)

for x in range(0,img_w):
    for y in range(0,img_h):
        #print("=====================")
        #print(x,y)
        #map_360 = _image_to_360(x, y)
        #print(map_360)
        #r_projection = _reverse_projection(map_360['x'], map_360['y'])
        r_projection = _reverse_projection(x, y)
        #print(r_projection)
        latlong = _translation_sphere(r_projection['long'], r_projection['lat'], to_radians(10), 0)
        #print(latlong)
        f_projection = _forward_projection(latlong['long'], latlong['lat'])
        #print(f_projection)
        #map_360 = _360_to_image(f_projection['x'], f_projection['y'])
        #print(map_360)
        #trans[map_360['y'], map_360['x']] = img[y,x]
        trans[int(round(f_projection['y'])), int(round(f_projection['x']))] = img[y,x]

cv2.imshow('img', img)
cv2.imshow('trans', trans)
cv2.waitKey(0)
cv2.destroyAllWindows()
