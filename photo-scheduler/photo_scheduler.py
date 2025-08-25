"""
This script automates photo capture from a Camera object at regular intervals and sends a heartbeat pulse to a specified URL.
Please set the desired configuration in the `config.yaml` file.

Args
    --config: Path to the YAML configuration file (default: config.yaml)
"""

from camera import Camera
import yaml
import os
from datetime import datetime, timedelta
import argparse
import asyncio
from pulse import Heartbeat, device_pulse


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Photo Scheduler")
    # Add path to configuration file
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()


def get_config(path):
    """Load configuration from a YAML file."""
    # Make sure absolute path is used
    config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), path)

    # Load configuration file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_timestamp(time_=None):
    """Return a timestamp formatted like '2024-01-01 12:34:56'."""
    if time_ is None:
        time_ = datetime.now()
    return time_.strftime("%Y-%m-%d %H:%M:%S")


def login(nvr, force=False):
    """
    Wrapper function for Camera.login(), which logs in to the NVR/Camera.
    Handles exceptions and prints status messages.
    """
    # Connect to NVR
    try:
        nvr.login(force)
        print(f"[{get_timestamp()}] Logged into NVR.")
        return True
    except Exception as e:
        print(f"[{get_timestamp()}] Failed to login to NVR. {e}")
        return False


def logout(nvr):
    """
    Wrapper function for Camera.logout(), which logs out of the NVR/Camera.
    Handles exceptions and prints status messages.
    """
    try:
        nvr.logout()
        print(f"[{get_timestamp()}] Logged out of NVR.")
        return True
    except Exception as e:
        print(f"[{get_timestamp()}] Failed to log out of NVR. {e}")
        return False


def take_photo(nvr, channel=0, group_by_timestamp=False):
    """
    Wrapper function for Camera.take_photo(), which takes a photo on the specified channel.
    Handles exceptions and prints status messages.

    Params:
        nvr (Camera): Camera object
        channel (int): Channel number. Default 0, which is the channel for a standalone camera
        group_by_timestamp (bool): If True, groups the photo by timestamp. Default False.
    """
    try:
        save_path = nvr.take_photo(channel=channel, group_by_timestamp=group_by_timestamp)
        print(f"[{get_timestamp()}] Saved image as {save_path}")
        return True
    except Exception as e:
        print(f"[{get_timestamp()}] Failed to take photo. {e}")
        return False


async def async_take_photo(nvr, channel, group_by_timestamp=False):
    """
    Non-blocking take photo. Runs take_photo() in a separate thread so that multiple photos can be taken concurrently.

    Params:
        nvr (Camera): Camera object
        channel (int): Channel number
        group_by_timestamp (bool): If True, groups the photo by timestamp. Default False.
    Returns:
        bool: True if photo was taken successfully, False otherwise
    """
    return await asyncio.to_thread(take_photo, nvr, channel, group_by_timestamp=group_by_timestamp)


async def take_photos_concurrently(nvr, channels: list[int], group_by_timestamp=False, heartbeat=None):
    """
    Take photos on each channel concurrently. If all photos are taken successfully, send a heartbeat pulse.

    Params:
        nvr (Camera): Camera object
        channels (list[int]): List of channel numbers
        group_by_timestamp (bool): If True, groups the photo by timestamp. Default False.
        heartbeat (dict): Dictionary with keys "url" and "data". See pulse.py for details. Leave as None to skip sending heartbeat.
    """
    # Take photos concurrently
    tasks = [async_take_photo(nvr, channel, group_by_timestamp=group_by_timestamp) for channel in channels]
    results = await asyncio.gather(*tasks)

    if heartbeat and all(results):
        # All photos were taken successfully, send heartbeat pulse
        device_pulse(heartbeat["url"], heartbeat["data"])
        print(f"[{get_timestamp()}] Sent heartbeat pulse to {heartbeat['url']}.")


async def take_photos_concurrently_in_order(nvr, channels: list[int], heartbeat=None):
    """
    Take photos on each channel concurrently in order. If all photos are taken successfully, send a heartbeat pulse.
    Photos are grouped by timestamp, each image is saved as <list index of channel>.jpg

    Params:
        nvr (Camera): Camera object
        channels (list[int]): Ordered list of channel numbers
        heartbeat (dict): Dictionary with keys "url" and "data". See pulse.py for details. Leave as None to skip sending heartbeat.
    """
    # Scramble labels and set global timestamp
    for c in range(len(channels)):
        nvr.labels[channels[c]] = str(c)
    nvr.labels["sync"] = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # Take photos concurrently
    await take_photos_concurrently(nvr, channels, True, heartbeat)


def parse_timedelta(time_str):
    """
    Parse a string in the format "1d 2h 3m 4s" (1 day, 2 hours, 3 minutes, 4 seconds) to a timedelta object.
    This is how the capture interval and max up time are specified in the configuration file.

    Params:
        time_str (str): Time string in the format "1d 2h 3m 4s"
    Returns:
        timedelta: Time delta object
    """
    units = {"d": "days", "h": "hours", "m": "minutes", "s": "seconds"}
    kwargs = {}
    for part in time_str.split():
        unit = units[part[-1]]
        value = int(part[:-1])
        kwargs[unit] = value
    return timedelta(**kwargs)


def main():
    args = parse_args()
    config = get_config(args.config)

    IP = config["ip"]
    USERNAME = config["username"]
    PASSWORD = config["password"]
    OUTPUT_DIR = config["savePath"]

    # Configure cameras
    cameras = config["cameras"]
    capture_times = config["captureTimes"]  # TODO: implement capture times
    capture_interval = parse_timedelta(config["captureInterval"])
    shutDownTime = datetime.now() + parse_timedelta(config["maxUpTime"])

    # Configure heartbeat pulse
    # True if we want to send pulses to the heartbeat monitor
    sendPulse = config["sendPulse"]
    # Dictionary with URL and data for heartbeat pulse
    heartbeat = (
        {
            "url": config["heartbeatURL"],
            "data": Heartbeat(
                device_id="NVR",
                interval=int(capture_interval.total_seconds()),
                password=config["heartbeatPassword"],
            ).return_heartbeat(),
        }
        if sendPulse
        else None
    )

    print(
        f"[{get_timestamp()}] Photo scheduler will capture photos every {capture_interval} until planned shutdown at {get_timestamp(shutDownTime)}."
    )

    # Create NVR object
    nvr = Camera(
        IP,
        USERNAME,
        PASSWORD,
        output_dir=OUTPUT_DIR,
        labels={
            camera["channel"]: camera["label"] for camera in cameras
        },  # Use labels from config
    )

    now = datetime.now()
    next_capture = now
    while now < shutDownTime:
        # If it's time to capture a photo, do it
        if now >= next_capture:
            # Check if we need to log in (token expired/expiring in less than 1 minute)
            if nvr.token_expiry - now < timedelta(minutes=1):
                login(nvr, force=True)
            # Take photos concurrently
            asyncio.run(
                take_photos_concurrently(
                    nvr, [camera["channel"] for camera in cameras], heartbeat=heartbeat
                )
            )
            next_capture += capture_interval  # Schedule next capture

        # Update current time
        now = datetime.now()

    # Shut down time has been reached
    logout(nvr)
    print(f"[{get_timestamp()}] Shutting down photo scheduler as planned.")


if __name__ == "__main__":
    main()

    # take_photos_concurrently_in_order demo
    take_photos_concurrently_in_order_demo = False
    if take_photos_concurrently_in_order_demo:
        ip = "192.168.1.2"
        username = "admin"
        password = "mulab1"
        order = [1,7,8,2,15,3,0,6,11,5,4,14,10,13,12,9]
        nvr = Camera(ip, username, password)
        nvr.login()
        asyncio.run(take_photos_concurrently_in_order(nvr, order))
