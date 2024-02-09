import depthai as dai
import numpy as np
import math, time, cv2, contextlib

class HostSpatialsCalc:
	# We need device object to get calibration data
	def __init__(self, device):
		self.calibData = device.readCalibration()

		# Values
		self.DELTA = 5
		self.THRESH_LOW = 200 # 20cm
		self.THRESH_HIGH = 30000 # 30m

	def setLowerThreshold(self, threshold_low):
		self.THRESH_LOW = threshold_low
	def setUpperThreshold(self, threshold_low):
		self.THRESH_HIGH = threshold_low
	def setDeltaRoi(self, delta):
		self.DELTA = delta

	def _check_input(self, roi, frame): # Check if input is ROI or point. If point, convert to ROI
		if len(roi) == 4: return roi
		if len(roi) != 2: raise ValueError("You have to pass either ROI (4 values) or point (2 values)!")
		# Limit the point so ROI won't be outside the frame
		self.DELTA = 5 # Take 10x10 depth pixels around point for depth averaging
		x = min(max(roi[0], self.DELTA), frame.shape[1] - self.DELTA)
		y = min(max(roi[1], self.DELTA), frame.shape[0] - self.DELTA)
		return (x-self.DELTA,y-self.DELTA,x+self.DELTA,y+self.DELTA)

	def _calc_angle(self, frame, offset, HFOV):
		return math.atan(math.tan(HFOV / 2.0) * offset / (frame.shape[1] / 2.0))

	# roi has to be list of ints
	def calc_spatials(self, depthData, roi, averaging_method=np.mean):

		depthFrame = depthData.getFrame()

		roi = self._check_input(roi, depthFrame) # If point was passed, convert it to ROI
		xmin, ymin, xmax, ymax = roi

		# Calculate the average depth in the ROI.
		depthROI = depthFrame[ymin:ymax, xmin:xmax]
		inRange = (self.THRESH_LOW <= depthROI) & (depthROI <= self.THRESH_HIGH)

		# Required information for calculating spatial coordinates on the host
		HFOV = np.deg2rad(self.calibData.getFov(dai.CameraBoardSocket(depthData.getInstanceNum())))

		averageDepth = averaging_method(depthROI[inRange])

		centroid = { # Get centroid of the ROI
			'x': int((xmax + xmin) / 2),
			'y': int((ymax + ymin) / 2)
		}

		midW = int(depthFrame.shape[1] / 2) # middle of the depth img width
		midH = int(depthFrame.shape[0] / 2) # middle of the depth img height
		bb_x_pos = centroid['x'] - midW
		bb_y_pos = centroid['y'] - midH

		angle_x = self._calc_angle(depthFrame, bb_x_pos, HFOV)
		angle_y = self._calc_angle(depthFrame, bb_y_pos, HFOV)

		spatials = {
			'z': averageDepth,
			'x': averageDepth * math.tan(angle_x),
			'y': -averageDepth * math.tan(angle_y)
		}
		return spatials, centroid

class TextHelper:
	def __init__(self) -> None:
		self.bg_color = (0, 0, 0)
		self.color = (255, 255, 255)
		self.text_type = cv2.FONT_HERSHEY_SIMPLEX
		self.line_type = cv2.LINE_AA
	def putText(self, frame, text, coords):
		cv2.putText(frame, text, coords, self.text_type, 0.5, self.bg_color, 3, self.line_type)
		cv2.putText(frame, text, coords, self.text_type, 0.5, self.color, 1, self.line_type)
	def rectangle(self, frame, p1, p2):
		cv2.rectangle(frame, p1, p2, self.bg_color, 3)
		cv2.rectangle(frame, p1, p2, self.color, 1)

class FPSHandler:
	def __init__(self):
		self.timestamp = time.time() + 1
		self.start = time.time()
		self.frame_cnt = 0
	def next_iter(self):
		self.timestamp = time.time()
		self.frame_cnt += 1
	def fps(self):
		return self.frame_cnt / (self.timestamp - self.start)

def createPipeline():
	# Create pipeline
	pipeline = dai.Pipeline()

	# Define sources and outputs
	monoLeft = pipeline.create(dai.node.MonoCamera)
	monoRight = pipeline.create(dai.node.MonoCamera)
	stereo = pipeline.create(dai.node.StereoDepth)

	# Properties
	monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
	monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
	monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
	monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

	stereo.initialConfig.setConfidenceThreshold(255)
	stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
	stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
	stereo.setLeftRightCheck(True)
	stereo.setExtendedDisparity(True)
	stereo.setSubpixel(False)

	# Linking
	monoLeft.out.link(stereo.left)
	monoRight.out.link(stereo.right)

	xoutDepth = pipeline.create(dai.node.XLinkOut)
	xoutDepth.setStreamName("depth")
	stereo.depth.link(xoutDepth.input)

	xoutDepth = pipeline.create(dai.node.XLinkOut)
	xoutDepth.setStreamName("disp")
	stereo.disparity.link(xoutDepth.input)
	
	return pipeline, stereo


with contextlib.ExitStack() as stack:
	deviceInfos = dai.Device.getAllAvailableDevices()
	usbSpeed = dai.UsbSpeed.SUPER
	openVinoVersion = dai.OpenVINO.Version.VERSION_2021_4

	qRgbMap = []
	devices = []

	for deviceInfo in deviceInfos:
		deviceInfo: dai.DeviceInfo
		device: dai.Device = stack.enter_context(dai.Device(openVinoVersion, deviceInfo, usbSpeed))
		devices.append(device)
		print("===Connected to ", deviceInfo.getMxId())
		mxId = device.getMxId()
		cameras = device.getConnectedCameras()
		usbSpeed = device.getUsbSpeed()
		eepromData = device.readCalibration2().getEepromData()
		print("   >>> MXID:", mxId)
		print("   >>> Num of cameras:", len(cameras))
		print("   >>> USB speed:", usbSpeed)
		if eepromData.boardName != "":
			print("   >>> Board name:", eepromData.boardName)
		if eepromData.productName != "":
			print("   >>> Product name:", eepromData.productName)

		pipeline = createPipeline()
		device.startPipeline(pipeline)


		# Output queue will be used to get the depth frames from the outputs defined above
		depthQueue = device.getOutputQueue(name="depth")
		dispQ = device.getOutputQueue(name="disp")

		text = TextHelper()
		hostSpatials = HostSpatialsCalc(device)
		y = 200
		x = 300
		step = 3
		delta = 5
		hostSpatials.setDeltaRoi(delta)
		lastTime = time.time()

	while True:
		depthData = depthQueue.get()
		# Calculate spatial coordiantes from depth frame
		spatials, centroid = hostSpatials.calc_spatials(depthData, (x,y)) # centroid == x/y in our case

		# Get disparity frame for nicer depth visualization
		disp = dispQ.get().getFrame()
		disp = (disp * (255 / stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
		disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

		text.rectangle(disp, (x-delta, y-delta), (x+delta, y+delta))
		text.putText(disp, "X: " + ("{:.1f}m".format(spatials['x']/1000) if not math.isnan(spatials['x']) else "--"), (x + 10, y + 20))
		text.putText(disp, "Y: " + ("{:.1f}m".format(spatials['y']/1000) if not math.isnan(spatials['y']) else "--"), (x + 10, y + 35))
		text.putText(disp, "Z: " + ("{:.1f}m".format(spatials['z']/1000) if not math.isnan(spatials['z']) else "--"), (x + 10, y + 50))

		# Show the frame
		cv2.imshow("depth", disp)

		print("frame per second: ", round(1 / (time.time() - lastTime)))
		lastTime = time.time()

		key = cv2.waitKey(1)
		if key == ord('q'):
			break
		elif key == ord('w'):
			y -= step
		elif key == ord('a'):
			x -= step
		elif key == ord('s'):
			y += step
		elif key == ord('d'):
			x += step
		elif key == ord('r'): # Increase Delta
			if delta < 50:
				delta += 1
				hostSpatials.setDeltaRoi(delta)
		elif key == ord('f'): # Decrease Delta
			if 3 < delta:
				delta -= 1
				hostSpatials.setDeltaRoi(delta)

		# Output queue will be used to get the rgb frames from the output defined above
		q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
		stream_name = "rgb-" + mxId + "-" + eepromData.productName
		qRgbMap.append((q_rgb, stream_name))
