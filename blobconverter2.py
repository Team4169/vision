import blobconverter

blob_path = blobconverter.from_tf(
    frozen_pb="/home/team4169/vision/content/fine_tuned_model/saved_model/saved_model.pb",
    data_type="FP16",
    shaves=5,
    optimizer_params=[
        "--reverse_input_channels",
        "--input_shape=[1,513,513,3]",
        "--input=1:mul_1",
        "--output=ArgMax",
        "--shaves=5",
    ],
)