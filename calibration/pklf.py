import pickle

with open('vars/cameraMatrix.pkl', 'rb') as f:
 data = pickle.load(f)

print('CameraMatrixData:\n',data)

print('\n\n\n')

with open('dist.pkl', 'rb') as f:
 data = pickle.load(f)

print('vars/distData:\n',data)
