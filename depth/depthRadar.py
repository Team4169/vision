import depthai as dai
import cv2
import numpy as np

def create_pipeline():
    pipeline = dai.Pipeline()

    # Define the stereo depth node
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setConfidenceThreshold(255)  # High confidence for better depth estimation
    stereo.setMedianFilter(dai.MedianFilter.KERNEL_7x7)  # Reduce noise in the depth map
    stereo.setLeftRightCheck(True)  # Enable left-right consistency check

    # Define the left and right cameras
    left = pipeline.create(dai.node.MonoCamera)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)

    right = pipeline.create(dai.node.MonoCamera)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # Link the cameras to the stereo depth node
    left.out.link(stereo.left)
    right.out.link(stereo.right)

    # Define the output streams
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    xout_disparity = pipeline.create(dai.node.XLinkOut)
    xout_disparity.setStreamName("disparity")
    stereo.disparity.link(xout_disparity.input)

    return pipeline

# Create the pipeline and start the device
with dai.Device(create_pipeline()) as device:
    depth_queue = device.getOutputQueue("depth", maxSize=4, blocking=False)
    disparity_queue = device.getOutputQueue("disparity", maxSize=4, blocking=False)

    while True:
        depth_frame = depth_queue.get().getFrame()
        disparity_frame = disparity_queue.get().getFrame()

        # Normalize the depth map for better visualization
        depth_frame_norm = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply a colormap to the depth map
        depth_frame_color = cv2.applyColorMap(depth_frame_norm, cv2.COLORMAP_JET)

        # Display the frames
        cv2.imshow("depth", depth_frame_color)
        cv2.imshow("disparity", disparity_frame, fps=True)

        if cv2.waitKey(1) == ord('q'):
            break