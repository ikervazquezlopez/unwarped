import cv2
import numpy as np
import random
import math
import sys
from os.path import isfile, isdir, join
from os import listdir
import os
import csv

EXTENSIONS = (".png", ".jpg")



if __name__ == '__main__':
    if len(sys.argv) != 3:
      print("Incorrect number of arguments!")
      print("Usage: python generate_dataset.py in_dir_path out_dir_path")
      exit("Terminated")

    if not isdir(sys.argv[1]):
        exit("in_dir_path is not valid!")
    if not isdir(sys.argv[2]):
        exit("out_dir_path is not valid!")

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]

    files = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]

    # Open the CSV file
    filename_csv = join(out_dir, "data.csv")
    f = open(filename_csv, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(["FILENAME", "ORIGINAL", "RX", "RY", "RZ"])

    # Generate rotation for each image
    for filename in files:
        if filename.endswith(EXTENSIONS):
            os.system("copy {} {}".format(join(in_dir, filename), join(out_dir, filename)))
            for _ in range(10):
                rotations = np.random.rand(3)
                rotations = (rotations * 360) - 180
                rx, ry, rz = np.uint8(rotations)
                os.system("python spherical_rotation.py {f} {rx} {ry} {rz} {in_dir} {out_dir}".format(f=filename,
                            in_dir=in_dir, out_dir=out_dir, rx=rx, ry=ry, rz=rz))
                name = filename.split(".")[0]
                name = "{}_{}_{}_{}.png".format(name, rx, ry, rz)
                writer.writerow([name, filename, rx, ry, rz])

    f.close()
