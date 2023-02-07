from PIL import Image
import numpy as np
import os, random

for x in ["train", "test", "valid"]:
    for imgurl in os.listdir("dataset/" + x + "/images"):
        img = Image.open("dataset/" + x + "/images/" + imgurl)
        numpydata = np.asarray(img).copy()

        for i in range(3):
            x1 = random.randint(0, numpydata.shape[0])
            x2 = random.randint(0, numpydata.shape[0])
            if x1 > x2:
                x1, x2 = x2, x1
            while not numpydata.shape[0] / 5 > abs(x1 - x2) > numpydata.shape[0] / 10: 
                x1 = random.randint(0, numpydata.shape[0])
                x2 = random.randint(0, numpydata.shape[0])
                if x1 > x2:
                    x1, x2 = x2, x1

            y1 = random.randint(0, numpydata.shape[1])
            y2 = random.randint(0, numpydata.shape[1])
            if y1 > y2:
                y1, y2 = y2, y1
            while not numpydata.shape[1] / 5 > abs(y1 - y2) > numpydata.shape[1] / 10: 
                y1 = random.randint(0, numpydata.shape[1])
                y2 = random.randint(0, numpydata.shape[1])
                if y1 > y2:
                    y1, y2 = y2, y1

            numpydata[x1 : x2, y1 : y2] = np.full((x2 - x1, y2 - y1, 3), 1) * np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

        im = Image.fromarray(numpydata)
        im.save("dataset_new/" + x + "/images/" + imgurl)
