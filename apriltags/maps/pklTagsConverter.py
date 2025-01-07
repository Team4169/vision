import pickle
mode = 1
if mode == 1:
    my_file = "/home/robotics4169/vision/apriltags/maps/fieldTagsConfig.pkl"

    with open(my_file, 'rb') as f:
        data = pickle.load(f)
        print(data)

############# OR ################
elif mode == 2:
    my_data = [1,2,3]
    my_file = "/home/robotics4169/vision/apriltags/maps/newPklFile.pkl"

    with open(my_file, "wb") as f:
        pickle.dump(my_data, f)

F = 1/39.3700787402 # conversion factor for inches to meters

DATA_FROM_2024 = [[0, 0, 0], [6.808597, -3.859403, 3.6651914291880923], [7.914259, -3.221609, 3.6651914291880923], [8.308467, 0.877443, 4.71238898038469], [8.308467, 1.442593, 4.71238898038469], [6.429883, 4.098925, 0.0], [-6.429375, 4.098925, 6.283185307179586], [-8.308975, 1.442593, 1.5707963267948966], [-8.308975, 0.877443, 1.5707963267948966], [-7.914767, -3.221609, 2.6179938779914944], [-6.809359, -3.859403, 2.6179938779914944], [3.633851, -0.392049, 6.8067840827778845], [3.633851, 0.393065, 2.6179938779914944], [2.949321, -0.000127, 4.71238898038469], [-2.950083, -0.000127, 1.5707963267948966], [-3.629533, 0.393065, 3.6651914291880923], [-3.629533, -0.392049, 5.759586531581287]]

DATA_FROM_2025_INCHES = [[]] # FIND IT ON: https://firstfrc.blob.core.windows.net/frc2025/FieldAssets/2025FieldDrawings-FieldLayoutAndMarking.pdf

DATA_FROM_2025_METERS = []
for tag in DATA_FROM_2025_INCHES:
    DATA_FROM_2025_METERS.append([tag[0] * F, tag[1] * F, tag[2] * F])

print(DATA_FROM_2025_METERS)









