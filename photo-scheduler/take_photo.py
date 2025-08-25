from camera import Camera
import warnings
import re
import sys
import time

# TODO: deal with warnings. Right now we suppress warnings so Node.js can read the output
warnings.filterwarnings("ignore")

# constants just for the prototype...
ip = "192.168.1.148"
channel = 0
username = "admin"
password = "627rocks"

args = sys.argv[1:]  # arg 0 is the script name

# User specified a time interval and number of photos to take
if len(args) == 2:
    interval = int(args[0])
    num_photos = int(args[1])

    try:
        camera = Camera(ip, channel, username, password)
        camera.login()
        for i in range(num_photos):
            photo_path = camera.take_photo()
            print(photo_path)
            time.sleep(
                interval
            )  # TODO: make sure new login session is created every hour

        camera.logout()
    except Exception as e:
        print(e)

# Just take one photo
else:
    # This is really slow if the camera is not connected?
    try:
        camera = Camera(ip, username, password)
        camera.login()
        photo_path = camera.take_photo()

        # remove the part of the path up to and including `public` for portability
        photo_path = re.sub(r".*/public/", "/", photo_path)
        print(photo_path)

        camera.logout()
    except Exception as e:
        print(e)
