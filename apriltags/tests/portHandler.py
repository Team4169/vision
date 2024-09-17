

def parse_v4l2_devices(output):
    mappings = {}
    lines = output.split('\n')
    current_device = None
    for line in lines:
        if line.strip().endswith(':'):
            current_device = line.strip()[:-1]
        elif '/dev/video' in line:
            video_index = line.strip().split('/')[-1]
            if current_device:
                mappings[current_device[-4:-1]] = int(video_index[-1])
    return mappings

def get_v4l2_device_mapping():
    try:
        output = subprocess.check_output(['v4l2-ctl', '--list-devices'], text=True)
        return parse_v4l2_devices(output)
    except subprocess.CalledProcessError as e:
        print("Error occurred")
        return parse_v4l2_devices(e.output)