from depthai_sdk import OakCamera, ArgsParser
from depthai_sdk.visualize.configs import BboxStyle, TextPosition
import argparse

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-conf", "--config", help="Trained YOLO json config path", default='/home/jetson/vision/depth/yolov8ntest1/YOLOv8nNORO.json', type=str)
args = ArgsParser.parseArgs(parser)

with OakCamera(args=args) as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn(args['config'], color, nn_type='yolo', spatial=True)
    # print(nn.out.nn_data)
    visualizer = oak.visualize(nn.out.passthrough, fps=True)
    visualizer.detections(
        color=(255, 255, 255),
        thickness=2,
        bbox_style=BboxStyle.RECTANGLE,
        label_position=TextPosition.MID,
    ).text(
        font_color=(255, 255, 255),
        auto_scale=True
    ).tracking(
        line_thickness=5
    )
    oak.start(blocking=True)
