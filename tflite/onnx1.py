import onnxruntime as ort
import numpy as np

# Change shapes and types to match model
input1 = np.zeros((1, 100, 100, 3), np.float32)

# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# Following code assumes NVIDIA GPU is available, you can specify other execution providers or don't include providers parameter
# to use default CPU provider.
sess = ort.InferenceSession("dst/path/model.onnx", providers=["CUDAExecutionProvider"])

# Set first argument of sess.run to None to use all model outputs in default order
# Input/output names are printed by the CLI and can be set with --rename-inputs and --rename-outputs
# If using the python API, names are determined from function arg names or TensorSpec names.
results_ort = sess.run(["output1", "output2"], {"input1": input1})

import tensorflow as tf
model = tf.saved_model.load("path/to/savedmodel")
results_tf = model(input1)

for ort_res, tf_res in zip(results_ort, results_tf):
    np.testing.assert_allclose(ort_res, tf_res, rtol=1e-5, atol=1e-5)

print("Results match")
