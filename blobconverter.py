import requests

url = "https://blobconverter.luxonis.com/compile"
payload = {
    'compile_type': 'model',
    'model_type': 'caffe',
    'intermediate_compiler_params': '--data_type=FP16 --mean_values [127.5,127.5,127.5] --scale_values [255,255,255]',
    'compiler_params': '-ip U8 -VPU_MYRIAD_PLATFORM VPU_MYRIAD_2480 -VPU_NUMBER_OF_SHAVES 4 -VPU_NUMBER_OF_CMX_SLICES 4'
}
files = {
    'model': open('/home/team4169/vision/content/fine_tuned_model/saved_model/saved_model.pb', 'rb'),
}
params = {
    'version': '2020.1',
}

try:
    response = requests.post(url, data=payload, files=files, params=params)
    response.raise_for_status()
    print(response.text)
except requests.exceptions.RequestException as e:
    print("Request failed:", e)
