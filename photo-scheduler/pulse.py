"""
Send pulses to the MuLab heartbeat monitor to indicate that a device is still running.
Author: Liam Seagram
"""

import requests
import time
import os

"""
class: Heartbeat
A class for heartbeat pulse objects.
"""


class Heartbeat:
    def __init__(self, device_id, interval, password, type=True, endpoint=None):
        self.device_id = device_id
        self.interval = interval
        self.type = type
        self.endpoint = endpoint
        self.password = password

    def return_heartbeat(self):
        if self.type:
            heartbeat_type = "Device Pulse"
            return {
                "Device_ID": self.device_id,
                "Interval": self.interval,
                "Password": self.password,
                "Type": heartbeat_type,
            }
        else:
            heartbeat_type = "Endpoint Pulse"
            return {
                "Device_ID": self.device_id,
                "Interval": self.interval,
                "Password": self.password,
                "Type": heartbeat_type,
                "Endpoint": self.endpoint,
            }


"""
function: device_pulse
Function for the device pulse, a pulse for the device in which this script is run.
"""


def device_pulse(heartbeat_url, heartbeat_data):
    try:
        response = requests.get(heartbeat_url, json=heartbeat_data)
    except requests.exceptions.ConnectionError:
        print("Failed to connect to the heartbeat monitor.")
        return
    if response.status_code == 200:
        print(
            f"Heartbeat received successfully for device ID: {heartbeat_data['Device_ID']}"
        )
    else:
        print(f"Failed to receive heartbeat. Status code: {response.status_code}")


"""
function: endpoint_pulse
Function for the endpoint pulse, hits an endpoint and sends a heartbeat pulse if the endpoint exists.
"""


def endpoint_pulse(heartbeat_url, heartbeat_data):
    try:
        heartbeat = requests.head(heartbeat_data['Endpoint'])
    except requests.exceptions.ConnectionError:
        print(f"Endpoint '{heartbeat_data['Endpoint']}' has no response, pulse not sent.")
        return
    if heartbeat.status_code == 200:
        try:
            response = requests.get(heartbeat_url, json=heartbeat_data)
        except requests.exceptions.ConnectionError:
            print("Failed to connect to the heartbeat monitor.")
            return

        if response.status_code == 200:
            print(
                f"Heartbeat received successfully for device ID: {heartbeat_data['Device_ID']}"
            )
        else:
            print(f"Failed to receive heartbeat. Status code: {response.status_code}")
    else:
        print(
            f"Endpoint '{heartbeat_data['Endpoint']}' has no response, pulse not sent."
        )


"""
function: pulse
Actually drives the pulses.
"""


def pulse(heartbeat_url, interval, heartbeats):
    while True:
        for heart in heartbeats:
            if heart["Type"] == "Device Pulse":
                device_pulse(heartbeat_url, heart)
            elif heart["Type"] == "Endpoint Pulse":
                endpoint_pulse(heartbeat_url, heart)
        time.sleep(interval)


if __name__ == "__main__":
    """
    The URL of the heartbeat monitor, and the interval all pulses in this script will use.
    """
    heartbeat_url = "https://pulse.caslab.queensu.ca/pulse"
    interval = 15
    password = "627rocks!"

    """
    To use this script, create new instances of the class "Heartbeat". You will only need one of the device type,
    and you may want any number of endpoint heartbeats.

    Device Heartbeat:
    The class's default values are for a device heartbeat, so you only need to pass in the unique device_id and the
    heartbeat class's interval (which is just passed on to the monitor).

    Endpoint Heartbeat:
    The endpoint heartbeat is initiated by passing in "False" as the type, then the endpoint you want to hit when initializing
    an instance of the heartbeat class.

    These heartbeats must then all be put together into a list and passed into the pulse() function.
    """

    # The first heartbeat, used to create a pulse from the device the script is run from.
    heartbeat_1 = Heartbeat("panoptes", interval, password)

    # The second heartbeat, used to hit an endpoint (here being used to check if the heartbeat monitor itself is working, as an example)
    # This is totally useless to actually do (unless we make a second heartbeat monitor to keep tabs on the first one!)
    heartbeat_2 = Heartbeat(
         "NVR",
         interval,
         password,
         False,
         "http://192.168.1.148",
    )

    # List of heartbeats
    heartbeats = [heartbeat_1.return_heartbeat(), heartbeat_2.return_heartbeat()]

    pulse(heartbeat_url, interval, heartbeats)
