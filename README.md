# CISC 500: 360 Camera
Inverted 360 camera mosaic using openCV.

## Usage
1. Install all **Requirements**
2. Clone this repository into a local folder
3. Move app/photo-scheduler module from https://github.com/QuMuLab/plutus-cameras into `360-camera` directory
4. Connect NVR to computer and ensure it is reachable via IP
5. Run `server.py`
6. Go to `http://127.0.0.1:5000/` and log in to the NVR
7. Click `Take All Photos`, then `Detect All Chessboards`
8. Under `Calibration`, enter the corresponding image numbers for each side, ensuring they are consistent. You should have 4 images per side.
9. Click `Calibrate!`
10. After succesful calibration, some `.pickle` files will be created containing calibration data
11. Run `Mosaicing.py` and follow the prompts. This will use your captured images and newly created pickle files.
12. Done!

## Requirements
- `cv2`
- `flask`
- `tqdm`
- `pyyaml`

## Other
  * [Link to Photos](https://queensuca-my.sharepoint.com/:f:/g/personal/20omha_queensu_ca/En7uACwYI1JDgKaHjc6D180BSr_ItMArLyjzeZceP9AZmA?email=christian.muise%40queensu.ca&e=vzrkr0)
