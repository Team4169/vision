import numpy as np
from math import pi
import pickle
FIELD_TAGS = [[0, 0, 0], [6.808597, -3.859403, (120+90)*pi/180], [7.914259, -3.221609, (120+90)*pi/180], [8.308467, 0.877443, (180+90)*pi/180], [8.308467, 1.442593, (180+90)*pi/180], [6.429883, 4.098925, (270+90)*pi/180 - 2*pi], [-6.429375, 4.098925, (270+90)*pi/180], [-8.308975, 1.442593, (0+90)*pi/180], [-8.308975, 0.877443, (0+90)*pi/180], [-7.914767, -3.221609, (60+90)*pi/180], [-6.809359, -3.859403, (60+90)*pi/180], [3.633851, -0.392049, (300+90)*pi/180], [3.633851, 0.393065, (60+90)*pi/180], [2.949321, -0.000127, (180+90)*pi/180], [-2.950083, -0.000127, (0+90)*pi/180], [-3.629533, 0.393065, (120+90)*pi/180], [-3.629533, -0.392049, (240+90)*pi/180]]

with open('FieldTagsConfig.pkl', 'wb') as f:
    pickle.dump(FIELD_TAGS, f)

step2 = input("Do you want to see the generated files? (y/n) ")

if step2 == 'y':
    new_FIELD_TAGS = []
    with open('FieldTagsConfig.pkl', 'rb') as f:
        new_FIELD_TAGS = pickle.load(f)

    print(new_FIELD_TAGS)