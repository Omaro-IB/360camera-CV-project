from flask import Flask, jsonify, request, render_template, send_file
import os
import yaml
import numpy as np
import glob
import asyncio
import sys
from tqdm import tqdm
import pickle
from calibration import detect_chessboard
# app/photo-scheduler module from https://github.com/QuMuLab/plutus-cameras
sys.path.insert(1, 'photoscheduler')  # must insert this path for relative imports
from photoscheduler.camera import Camera
from photoscheduler import photo_scheduler


# SETUP
# Global Variables
CAMERA_SETTINGS = {}  # settings for image capture and chessboard detection
ALL_CORNERS = []  # numpy ndarray for each camera's corner coordinates
OUTPUT_DIR = 'front/static/capture'  # output directory for frontend to access training_images
DATA_DIR = './data'  # data directory for camera settings


# Initialize global settings from settings.yaml
def initialize():
    global CAMERA_SETTINGS
    global ALL_CORNERS

    with open(os.path.join(DATA_DIR, "settings.yaml"), "r") as settings_file:
        CAMERA_SETTINGS = yaml.safe_load(settings_file)

    CAMERA_SETTINGS["nvr"] = Camera('192.168.1.2', 'admin', 'mulab1',
                                    output_dir=OUTPUT_DIR)  # Camera object for image capture
    CAMERA_SETTINGS["detected"] = [False]*16  # which cameras have detected the corners
    ALL_CORNERS = np.array([None]*16)  # None if camera has not detected corners, else the corners as a np ndarray


# Reset image detection state
def reset_images(idx="all"):
    global CAMERA_SETTINGS
    global ALL_CORNERS

    if idx == "all":
        CAMERA_SETTINGS["detected"] = [False]*16
        ALL_CORNERS = np.array([None]*16)
    elif type(idx) is int:
        CAMERA_SETTINGS["detected"][idx] = False
        ALL_CORNERS[idx] = None
    else:
        for i in idx:
            CAMERA_SETTINGS["detected"][i] = False
            ALL_CORNERS[i] = None


# Get global settings as JSON (for HTTP responses)
def json_global_settings():
    json_settings = {}
    for key, value in CAMERA_SETTINGS.items():
        if key != "nvr":
            json_settings[key] = value

    json_settings["camera_ip"] = CAMERA_SETTINGS["nvr"].ip
    json_settings["camera_username"] = CAMERA_SETTINGS["nvr"].username
    json_settings["camera_password"] = CAMERA_SETTINGS["nvr"].password
    return json_settings


app = Flask(__name__, template_folder='front/views', static_folder='front/static')


# serve frontend
@app.route('/', methods=['GET'])
def frontend():
    return render_template("index.html")


# index of all endpoints
@app.route('/cgi', methods=['GET'])
def index():
    return jsonify({
        '/cgi': 'index (see this page)',
        '/cgi/settings/reset': 'reset all settings to default',
        '/cgi/settings/get': 'get all settings',
        '/cgi/settings/set (settings)': 'set settings given JSON - all keys must be one of /settings/get',
        '/cgi/login': 'login to camera using ip, username, and password as specified in settings',
        '/cgi/photos_in_order': 'take photos in order as specified in settings - each image is saved as <list index of channel>.jpg',
        '/cgi/retake_photo (image_number)': 'retake photo given the list index of the channel (i.e. the image number) as specified in settings',
        '/cgi/detect_chessboards': 'try to detect chessboards in last taken training_images',
        '/cgi/redetect_chessboard (image_number)': 'redetect chessboard given the list index of the channel (i.e. the image number) as specified in settings',
        '/cgi/bintensors': 'get names of available bin tensors',
        '/cgi/timestamp/get': 'get timestamp of last taken photos',
        '/cgi/calibrate (image_numbers, side)': 'Calibrate a set of cameras given their list indices of their channel (i.e. the image numbers) as specified in settings. Side should be one of {+z, -z, +x, -x}'
    }), 200


# Completely reset all settings, training_images, etc.
@app.route('/cgi/settings/reset', methods=['POST'])
def reset():
    initialize()
    return jsonify(json_global_settings()), 200


# get all settings
@app.route('/cgi/settings/get', methods=['GET'])
def get_settings():
    return jsonify(json_global_settings()), 200


# get latest timestamp
@app.route('/cgi/timestamp/get', methods=['GET'])
def get_timestamp():
    timestamp = CAMERA_SETTINGS["nvr"].labels.get("sync")
    if timestamp:
        return jsonify({'timestamp': timestamp}), 200
    else:
        print(CAMERA_SETTINGS["nvr"].labels)
        return jsonify({'status': 'failure', 'message': 'must capture photos first'}), 400


# set timestamp to old timestamp
@app.route('/cgi/timestamp/set', methods=['POST'])
def timestamp_set():
    new_timestamp = request.form.get('timestamp')
    if not new_timestamp:
        return jsonify({'status': 'failure', 'message': '"timestamp" field is required'}), 400

    CAMERA_SETTINGS["nvr"].labels["sync"] = new_timestamp
    return jsonify({'status': 'success'}), 200


# set a setting
@app.route('/cgi/settings/set', methods=['POST'])
def set_setting():
    # Get variables
    global CAMERA_SETTINGS
    settings = request.get_json()
    if not settings:
        return jsonify({'status': 'error', 'message': 'JSON is required with request to set settings'}), 400

    # Define setting types
    int_lists = ("order", "pattern_size", "image_size")
    float_lists = ("+z", "-z", "+x", "-x")
    float_settings = ("crop_x", "crop_y", "focal_length")
    string_settings = ("camera_ip", "camera_username", "camera_password")

    for setting in settings:
        if setting in int_lists:  # int list
            if not (type(settings[setting]) is list and len(settings[setting]) == len(CAMERA_SETTINGS[setting]) and all(type(i) is int for i in settings[setting])):
                return jsonify({'status': 'failure', 'message': f'invalid {setting} - should be list of {len(CAMERA_SETTINGS[setting])} integers'}), 400
            CAMERA_SETTINGS[setting] = settings[setting]

        elif setting in float_lists:  # float list
            if not (type(settings[setting]) is list and len(settings[setting]) == len(CAMERA_SETTINGS[setting]) and all(type(i) is float for i in settings[setting])):
                return jsonify({'status': 'failure', 'message': f'invalid {setting} - should be list of {len(CAMERA_SETTINGS[setting])} floats'}), 400
            CAMERA_SETTINGS[setting] = settings[setting]

        elif setting in float_settings:  # float
            if not (type(settings[setting]) is float):
                return jsonify({'status': 'failure', 'message': f'invalid {setting} - should be float'}), 400
            CAMERA_SETTINGS[setting] = settings[setting]

        elif setting in string_settings:  # string
            if setting == "camera_ip":
                CAMERA_SETTINGS["nvr"] = Camera(settings[setting], CAMERA_SETTINGS["nvr"].username, CAMERA_SETTINGS["nvr"].password, output_dir=OUTPUT_DIR)
            elif setting == "camera_username":
                CAMERA_SETTINGS["nvr"] = Camera(CAMERA_SETTINGS["nvr"].ip, settings[setting], CAMERA_SETTINGS["nvr"].password, output_dir=OUTPUT_DIR)
            elif setting == "camera_password":
                CAMERA_SETTINGS["nvr"] = Camera(CAMERA_SETTINGS["nvr"].ip, CAMERA_SETTINGS["nvr"].username, settings[setting], output_dir=OUTPUT_DIR)
            else:
                CAMERA_SETTINGS[setting] = settings[setting]

        else:  # invalid setting
            return jsonify({'status': 'error', 'message': f'unknown setting - set one of {int_lists + float_lists + float_settings + string_settings}'}), 400

    return jsonify(json_global_settings()), 200


# Login to reolink using NVR's ip, username, and password
@app.route('/cgi/login', methods=['POST'])
def login():
    try:
        CAMERA_SETTINGS["nvr"].login()
    except Exception as e:
        return jsonify({'status': 'failure', 'message': str(e)}), 400
    return jsonify({'status': 'success', 'message': 'successfully logged in'}), 200


# Take photos in order - each image is saved as <list index of channel>.jpg
@app.route('/cgi/photos_in_order', methods=['POST'])
def photos_in_order():
    if not CAMERA_SETTINGS["nvr"].token:
        return jsonify({'status': 'failure', 'message': 'No camera connected, must login first'}), 400
    try:
        asyncio.run(photo_scheduler.take_photos_concurrently_in_order(CAMERA_SETTINGS["nvr"], CAMERA_SETTINGS["order"]))
    except Exception as e:
        return jsonify({'status': 'failure', 'message': str(e)}), 400

    reset_images()
    return jsonify({'status': 'success', 'message': 'successfully captured all photos', 'timestamp': CAMERA_SETTINGS["nvr"].labels["sync"]}), 200


# Retake photo given list index of channel - useful if detect_chessboards fails for 1 or more cameras
@app.route('/cgi/retake_photo', methods=['POST'])
def retake_photo():
    # Get variable
    image_number = request.form.get('image_number')
    if not image_number:
        return jsonify({'status': 'failure', 'message': '"image_number" field is required'}), 400

    labels = CAMERA_SETTINGS["nvr"].labels
    if not labels:
        return jsonify({'status': 'failure', 'message': 'Must capture all photos first'}), 400
    try:
        channel = list(labels.keys())[list(labels.values()).index(image_number)]  # get channel number from image number
    except ValueError:
        return jsonify({'status': 'failure', 'message': 'Must capture all photos first'}), 400
    try:
        status = asyncio.run(photo_scheduler.async_take_photo(CAMERA_SETTINGS["nvr"], channel, group_by_timestamp=True))
    except Exception as e:
        return jsonify({'status': 'failure', 'message': str(e)}), 400
    if not status:
        return jsonify({'status': 'failure', 'message': 'failed to take photo'}), 400

    reset_images(image_number)
    return jsonify({'status': 'success',
                    'message': f'successfully captured image {image_number} (channel {channel})',
                    'timestamp': CAMERA_SETTINGS["nvr"].labels["sync"]}), 200


# Detect chessboard of last taken photos
@app.route('/cgi/detect_chessboards', methods=['POST'])
def detect_chessboards():
    print("Detecting chessboards...\n")
    if not CAMERA_SETTINGS["nvr"].labels or not CAMERA_SETTINGS["nvr"].labels.get("sync"):
        return jsonify({'status': 'failure', 'message': 'Must capture all photos first'}), 400

    timestamp = CAMERA_SETTINGS["nvr"].labels["sync"]

    for im_path in tqdm(glob.glob(os.path.join(CAMERA_SETTINGS["nvr"].output_dir, timestamp, "*.jpg")), unit="image"):
        # Read image and detect chessboards
        status = detect_chessboard.detect_chessboard_lut_search(im_path, detect_chessboard.default_parameters(),
                                                                CAMERA_SETTINGS["pattern_size"],
                                                                CAMERA_SETTINGS["crop_x"], CAMERA_SETTINGS["crop_y"],
                                                                save_image=True)

        # If successful, save corners to memory
        idx = int(os.path.basename(im_path).split(".")[0])
        if status is False:
            print(f" image #{idx} not detected")
        else:
            ALL_CORNERS[idx] = status[1]
            CAMERA_SETTINGS["detected"][idx] = True
            print(f" image #{idx} detected")

    # Return response
    if not any(CAMERA_SETTINGS["detected"]):
        return jsonify({'status': 'failure', 'message': 'no chessboards detected'}), 400
    else:
        return jsonify({'status': 'success',
                        'detected_images': f'{[i for i in range(len(CAMERA_SETTINGS["detected"])) if CAMERA_SETTINGS["detected"][i]]}',
                        'timestamp': timestamp}), 200


# Retake photo given list index of channel - useful if detect_chessboards fails for 1 or more cameras
@app.route('/cgi/redetect_chessboard', methods=['POST'])
def redetect_chessboard():
    # Get variable
    image_number = request.form.get('image_number')
    if not image_number:
        return jsonify({'status': 'failure', 'message': '"image_number" field is required'}), 400
    timestamp = CAMERA_SETTINGS["nvr"].labels.get("sync")
    if not timestamp:
        return jsonify({'status': 'failure', 'message': 'must capture photos first'}), 400

    im_path = os.path.join(CAMERA_SETTINGS["nvr"].output_dir, timestamp, f"{image_number}.jpg")

    # Read image and detect chessboards
    status = detect_chessboard.detect_chessboard_lut_search(im_path, detect_chessboard.default_parameters(),
                                                            CAMERA_SETTINGS["pattern_size"],
                                                            CAMERA_SETTINGS["crop_x"], CAMERA_SETTINGS["crop_y"],
                                                            save_image=True)
    # If successful, save corners to memory
    if status is False:
        return jsonify({'status': 'failure', 'message': 'chessboard not detected'}), 400
    else:
        ALL_CORNERS[int(image_number)] = status[1]
        CAMERA_SETTINGS["detected"][int(image_number)] = True
        return jsonify({'status': 'success',
                        'detected_images': f'{[i for i in range(len(CAMERA_SETTINGS["detected"])) if CAMERA_SETTINGS["detected"][i]]}',
                        'timestamp': timestamp}), 200


# For a given side and cameras which see that side, get their position/orientation relative to side top-center
@app.route('/cgi/calibrate', methods=['POST'])
def calibrate():
    # Get variables
    data = request.get_json()
    image_numbers = data.get('image_numbers')
    side = data.get('side')
    column = CAMERA_SETTINGS[side]

    # Verify variables
    if not (image_numbers and side):
        return jsonify({'status': 'failure', 'message': 'both "image_numbers" and "side" fields are required'}), 400
    # all chessboards detected
    if not all(CAMERA_SETTINGS["detected"]):
        return jsonify({'status': 'failure', 'message': 'all chessboards must be detected before calibration'}), 400
    # column coordinates consistent with pattern size
    if not len(column) == CAMERA_SETTINGS["pattern_size"][1]:
        return jsonify({'status': 'failure', 'message': f'inconsistent pattern_size ({CAMERA_SETTINGS["pattern_size"]}) with number of specified column coordinates ({len(column)})'}), 400

    # List of image corners with respect to provided image numbers
    try:
        image_corners = []
        image_nums = []
        for image_idx in image_numbers.split(","):
            image_corners.append(ALL_CORNERS[int(image_idx)])
            image_nums.append(image_idx)
    except ValueError:
        return jsonify({'status': 'failure', 'message': f'invalid image numbers "{image_numbers}"'}), 400
    except IndexError:
        return jsonify({'status': 'failure', 'message': f'invalid image number "{image_idx}"'}), 400

    # Calibrate
    ret, mtx, dist, rvecs, tvecs = detect_chessboard.calibrate(image_corners, CAMERA_SETTINGS["pattern_size"],
                                                               column, CAMERA_SETTINGS["image_size"], CAMERA_SETTINGS["focal_length"])
    with open(f'{CAMERA_SETTINGS["nvr"].labels["sync"]}({image_numbers}).pickle', 'wb') as f:
        pickle.dump((ret, mtx, dist, rvecs, tvecs, image_corners, image_nums), f)
    return jsonify({'status': 'success', 'file': f.name}), 200


# Get status
@app.route('/cgi/status', methods=['GET'])
def get_status():
    return jsonify({'detected_images': CAMERA_SETTINGS['detected'], 'timestamp': CAMERA_SETTINGS["nvr"].labels.get("sync")}), 200


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python3 server.py <host> <port> <debug>")
        sys.exit(1)

    initialize()

    host = sys.argv[1]
    port = int(sys.argv[2])
    debug = sys.argv[3].lower() in ['true', '1', 't', 'y', 'yes']
    app.run(host=host, port=port, debug=debug)

