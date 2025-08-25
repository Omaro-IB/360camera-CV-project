import requests
import os
import json
from datetime import datetime, timedelta
import warnings
from urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings("ignore", category=InsecureRequestWarning)


class Camera:
    """
    Creates a Camera object that represents a Reolink NVR (with cameras attached) or a standalone Reolink camera.
    Use a Camera object to log in to a camera, take a photo, and log out.
    Cameras must be logged into every 60 minutes to maintain a session. A token is generated upon login and is used to authenticate requests for any camera actions.

    Params
        ip (str): IP address of the camera.
                  It is recommended to configure the Reolink NVR/camera with a static IP address. This can be done in the Reolink app settings.
        username (str): Username for the camera.
                        The default username is `admin`.
        password (str): Password for the camera. The password on all existing cameras in the MuLab is `627rocks`.
        token (str): If an active token exists, supply it here and you will not have to login() again.
        token_expiry (datetime): If an active token exists, supply the expiry time here.
        output_dir (str): Directory in which to save photos. Default is `public/photos`.
        labels (dict): A dictionary of the format {channel_no (int) : name (str)}.
                       Provide a name for the camera at certain channel numbers. If no label is provided, the channel number itself is used as the label. The label is used to organize photos into folders.
    """

    # Set root directory to `app`
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    def __init__(
        self,
        ip,
        username,
        password,
        token=None,
        token_expiry=None,
        output_dir="public/photos",
        labels=None,
    ):
        self.ip = ip
        self.username = username
        self.password = password
        self.token_expiry = token_expiry
        if token is None:
            self.reset_token()
        self.output_dir = os.path.join(Camera.root_dir, output_dir)
        self.labels = labels or {}

    def login(self, force=False, verbose=False):
        """
        Log into the camera using self.username and self.password to acquire a token that will validate a 60-minute session.

        Params
            force (bool): Create a new token even if one already exists.
            verbose (bool): Print status messages.
        Returns
            None: If successful, the token is stored in self.token and is also saved to a file.
        """
        # If valid token exists don't log in again
        if (
            self.token is not None  # token already exists
            and datetime.now() < self.token_expiry  # token is not expired
            and not force  # force=True creates a new token even if one already exists
        ):
            if verbose:
                print("Already logged in; active token exists")
            return

        url = f"https://{self.ip}/cgi-bin/api.cgi?cmd=Login"
        data = [
            {
                "cmd": "Login",
                "param": {
                    "User": {
                        "Version": "0",
                        "userName": self.username,
                        "password": self.password,
                    }
                },
            }
        ]

        response = requests.post(url, json=data, verify=False)
        # TODO: don't bypass SSL verification

        if response.status_code == 200:
            data = json.loads(response.content)
            self.token = data[0]["value"]["Token"]["name"]
            self.token_expiry = datetime.now() + timedelta(seconds=3600)

            # Create temp/tokens directory if it doesn't exist
            save_dir = os.path.join(Camera.root_dir, "temp/tokens")
            os.makedirs(save_dir, exist_ok=True)

            # Save token to output directory
            save_path = os.path.join(save_dir, f"{self.ip}_token.txt")
            with open(save_path, "wb") as file:
                file.write(response.content)

            if verbose:
                print(f"Token saved to {save_path}")
        else:
            raise TokenError(f"Failed to log in, status code {response.status_code}")

    def logout(self, verbose=False):
        """
        Log out of the camera and relinquish the active token, ending the current session.

        Params
            verbose (bool): Print status messages.
        Returns
            None: If successful, the token is removed from self.token and the file it was previously written to.
        """
        if self.token is None:
            raise TokenError("Cannot log out (no active token)")

        url = f"https://{self.ip}/cgi-bin/api.cgi?cmd=Logout&token={self.token}"
        data = [{"cmd": "Logout", "param": {}}]

        response = requests.post(url, json=data, verify=False)

        if response.status_code == 200:
            self.reset_token()

            # Create temp/tokens directory if it doesn't exist
            save_dir = os.path.join(Camera.root_dir, "temp/tokens")
            os.makedirs(save_dir, exist_ok=True)

            # Save logout confirmation to output directory
            save_path = os.path.join(save_dir, f"{self.ip}_token.txt")
            with open(save_path, "wb") as file:
                file.write(response.content)

            if verbose:
                print(f"Removed token at {save_path}")
        else:
            raise TokenError(f"Failed to log out, status code {response.status_code}")

    def take_photo(self, channel=0, verbose=False, group_by_timestamp=False):
        """
        Take a photo from the camera at the specified channel number, and save it to the output directory at self.output_dir.

        Params
            group_by_timestamp (bool): Set to true to save all channels under the same directory. Leave as False for default behavior (save in separate directories).
            file_name (str): Custom filename to save the photo. Leave as None for default behavior.
            channel (int): Channel number.
                           An individual camera is always channel 0. A camera connected to a 16-channel NVR may be on channel 0-15.
            verbose (bool): Print status messages.
        Returns
            str: Path to the saved image.
        """
        if self.token is None:
            raise TokenError("No active token. Please call login() first.")
        if datetime.now() > self.token_expiry:
            raise TokenError("Token expired. Please call login() again.")

        random_string = "flsYJfZgM6RTB_os"  # TODO: generate random sequence
        url = f"https://{self.ip}/cgi-bin/api.cgi?cmd=Snap&channel={channel}&rs={random_string}&token={self.token}"
        # TODO: send token as parameter instead of in the URL

        response = requests.post(url, verify=False)

        if response.status_code == 200:
            # Write the image to a file
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

            # Save image to photos directory

            # If the camera has a label, use it - otherwise use the channel number
            label = (
                self.labels[channel]
                if (channel in self.labels) and (self.labels[channel])
                else f"channel{channel}"
            )

            # If group_by_timestamp, save under timestamp directory - otherwise group by camera label / channel number
            if group_by_timestamp:
                save_dir = (
                    # "sync" label used for syncing the timestamp if necessary
                    os.path.join(self.output_dir, self.labels["sync"])
                    if ("sync" in self.labels) and (self.labels["sync"])
                    else os.path.join(self.output_dir, timestamp)
                )
                save_path = os.path.join(save_dir, f"{label}.jpg")
            else:
                save_dir = os.path.join(self.output_dir, label)
                save_path = os.path.join(save_dir, f"{label}_{timestamp}.jpg")

            # Create photos directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)

            with open(save_path, "wb") as file:
                file.write(response.content)
            if verbose:
                print(f"Image saved as {save_path}")

            return save_path
        else:
            raise TokenError(
                f"Failed to take photo, status code {response.status_code}"
            )

    def reset_token(self):
        """
        Clear token and set token_expiry to a time in the past.
        """
        self.token = None
        self.token_expiry = datetime.now() - timedelta(
            seconds=1
        )  # token is expired by default


class TokenError(Exception):
    """
    Exception raised when a token is missing or invalid, or cannot be generated.
    """

    pass


# Quick test - take photo on channel 0
if __name__ == "__main__":
    ip = "192.168.1.148"
    username = "admin"
    password = "627rocks"
    camera = Camera(ip, username, password)
    camera.login()
    output_path = camera.take_photo(channel=0)
    print("Check image at", output_path)
