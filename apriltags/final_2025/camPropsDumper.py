import numpy as np
from math import pi
import pickle
cam_props =3
'''

LEFT
np.array([[668.2138474014353, 0.0, 332.83301545896086], [0.0, 666.4860881212383, 214.33779667521517], [0.0, 0.0, 1.0]],dtype=np.float32),
np.array([[0.22224705297101408, -1.7549821808892665, -0.005523738126667523, 0.0051301529546101616, 3.4133532108023994]], dtype = np.float32),
np.array([-0.14605,0.4572,0.3302],dtype=np.float32)

RIGHT
np.array([[660.6703723058181, 0.0, 321.2980455248988], [0.0, 658.6516133373474, 218.49261248405028], [0.0, 0.0, 1.0]],dtype=np.float32),
np.array([[0.18802634354539693, -1.5669527368643557, -0.0006972309753818612, -0.0018548904430247361, 3.04483663171066]], dtype = np.float32),
np.array([-0.13335,0.4572,0.2413],dtype=np.float32)

'''

with open('camConfigRight.pkl', 'wb') as f:
    pickle.dump(
    [
np.array([[660.6703723058181, 0.0, 321.2980455248988], [0.0, 658.6516133373474, 218.49261248405028], [0.0, 0.0, 1.0]],dtype=np.float32),
np.array([[0.18802634354539693, -1.5669527368643557, -0.0006972309753818612, -0.0018548904430247361, 3.04483663171066]], dtype = np.float32),
np.array([-0.13335,0.4572,0.2413],dtype=np.float32)
    ], f)

step2 = input("Do you want to see the generated files? (y/n) ")

if step2 == 'y':
    new_FIELD_TAGS = []
    with open('camConfigRight.pkl', 'rb') as f:
        new_FIELD_TAGS = pickle.load(f)

    print(new_FIELD_TAGS)
