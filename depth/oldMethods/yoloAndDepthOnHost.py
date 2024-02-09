import depthai as dai
import cv2
import torch

# Load YOLOv8 model from .pt file
model_path = '/home/jetson/Downloads/yolov81.pt'  # Replace with the path to your .pt file
model_dict = torch.load(model_path)
model = model_dict['model']  # Access the model from the dictionary
model.cuda()

# Setup DepthAI pipeline
pipeline = dai.Pipeline()

# Create Color Camera
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Create Stereo Depth
left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

left.out.link(stereo.left)
right.out.link(stereo.right)

# Create XLinkOuts
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_depth = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
xout_depth.setStreamName("depth")

cam_rgb.video.link(xout_rgb.input)
stereo.depth.link(xout_depth.input)

# Connect to OAK-D and process frames
with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        in_rgb = q_rgb.get()  # Get RGB frame
        in_depth = q_depth.get()  # Get Depth frame

        frame = in_rgb.getCvFrame()
        depth_frame = in_depth.getFrame()

        # Run YOLOv8 detection on host
        results = model(frame).pred[0]

        for *xyxy, conf, cls in results:
            x, y, x2, y2 = [int(i.item()) for i in xyxy]
            # Calculate depth
            depth = depth_frame[y:y2, x:x2]
            depth_value = depth.mean()  # Can also use median, etc.

            print(f"Object at [{x}, {y}], Depth: {depth_value} mm")

        # Display frames (optional)
        cv2.imshow("RGB", frame)
        cv2.imshow("Depth", depth_frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
