# CISC 500: 360 Camera
Inverted 360 camera mosaic using openCV.

## Usage
1. Install all **Requirements**
2. Clone this repository into a local folder
3. Connect NVR to computer and ensure it is reachable via IP
4. Run `server.py`
5. Go to `http://127.0.0.1:5000/` and log in to the NVR
6. Click `Take All Photos`, then `Detect All Chessboards`
7. Under `Calibration`, enter the corresponding image numbers for each side, ensuring they are consistent. You should have 4 images per side.
8. Click `Calibrate!`
9. After succesful calibration, some `.pickle` files will be created containing calibration data
10. Run `Mosaicing.py` and follow the prompts. This will use your captured images and newly created pickle files.
11. Done!

## Requirements
- `cv2`
- `flask`
- `tqdm`
- `pyyaml`

## Other
  * [Link to Photos](https://queensuca-my.sharepoint.com/:f:/g/personal/20omha_queensu_ca/En7uACwYI1JDgKaHjc6D180BSr_ItMArLyjzeZceP9AZmA?email=christian.muise%40queensu.ca&e=vzrkr0)
