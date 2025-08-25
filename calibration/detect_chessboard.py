import glob
import os.path
import cv2
import numpy as np
import multiprocessing
import random


def show_img(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1600, 1200)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_pixels_no_transparency(image):
    """
    Reads cv image and returns HSV channels as numpy array
    :param image: numpy.ndarray: the image (use cv2.imread)
    :return: numpy.ndarray: array of pixels without transparency
    """
    # Convert image to HSV space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Remove all transparent pixels (V = 255)
    pixels = hsv.reshape(-1, 3)
    i = np.where(pixels[:, 2] != 255)
    pixels_no_transparency = pixels[i]

    # return channels each in own array
    return pixels_no_transparency


def get_channels(image):
    """
    Reads cv image and returns HSV channels as numpy array
    :param image: numpy.ndarray: the image (use cv2.imread)
    :return: [numpy.ndarray]: HSV channels as list of numpy arrays
    """
    # Convert image to HSV space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Remove all transparent pixels (V = 255)
    pixels = hsv.reshape(-1, 3)
    i = np.where(pixels[:, 2] != 255)
    pixels_no_transparency = pixels[i]

    # return channels each in own array
    return pixels_no_transparency[:, 0], pixels_no_transparency[:, 1], pixels_no_transparency[:, 2]


def populate_bin_tensor(bin_tensor, images_total, images_subset):
    """
    :param bin_tensor: BinTensor: an empty BinTensor
    :param images_total: list: list of directories to training_images
    :param images_subset: list: list of directories containing subset training_images of pixels from the training_images in images_total.
                                The indices of these training_images should correspond to the total image from which the subset was extracted.
    """
    for img_i in range(len(images_total)):
        # Get total and subset pixels, removing transparency
        pixels_total = get_pixels_no_transparency(cv2.imread(images_total[img_i]))
        pixels_sub = get_pixels_no_transparency(cv2.imread(images_subset[img_i]))

        # Mean and standard deviation of H/S/V of total pixels
        means = np.mean(pixels_total, axis=0)
        stds = np.std(pixels_total, axis=0)

        # Add normalized subset pixel to bin tensor
        for pix in pixels_sub:
            bin_tensor.add_point((pix[0] - means[0]) / stds[0], (pix[1] - means[1]) / stds[1],
                                 (pix[2] - means[2]) / stds[2], suppress_warnings=True)


def get_mask_function_generator(bin_tensor, threshold=0.01):
    """
    Generate a mask function (takes H,S,V values and returns True/False) using a bin tensor and training_images for training
    :param bin_tensor: BinTensor: an empty BinTensor
    :param threshold: float: default=.01: smaller threshold values means higher positive rate
    :return: int, int -> Boolean: the mask generator function (takes mean and standard deviation and returns the mask)
    """

    def get_mask_function(img_means, img_stds):
        def fn(h, s, v):
            bindices = bin_tensor.get_bindex((h - img_means[0]) / img_stds[0],
                                             (s - img_means[1]) / img_stds[1],
                                             (v - img_means[2]) / img_stds[2])
            if bindices:
                v_bin = bin_tensor.data[bindices]
                # return random() < ((v_bin / bin_tensor.data[bin_tensor.maximum]) * tolerance)
                return v_bin / bin_tensor.data[bin_tensor.maximum] > threshold
            else:  # this point is out of the range of bin_tensor
                return 0

        return fn

    return get_mask_function


def get_mask(image, mask_f, display_mask=True, crop_x=0.0, crop_y=0.0):
    """
    Perform color-segmentation using HSV channels pixel distances
    :param image: numpy.ndarray: image to be processed (use img = cv2.imread(path_to_image))
    :param mask_f: int, int, int -> bool: mask function to be used - should take parameters h,s,v and return True/False
    :param display_mask: bool: default=True: display mask after applying color segmentation
    :param crop_x: float: default = 0: percentage of width to crop
    :param crop_y: float: default = 0: percentage of height to crop
    :return: mask: numpy.ndarray
    """
    assert 0 <= crop_x <= 1
    assert 0 <= crop_y <= 1

    # Color-segmentation using HSV to get binary mask
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    msk = np.zeros(h.shape, dtype=np.uint8)
    # Iterate over each pixel and apply the custom function
    for i in range(int(h.shape[0] * (crop_x / 2)), int(h.shape[0] * (1 - crop_x / 2))):  # Loop over rows
        for j in range(int(h.shape[1] * (crop_y / 2)), int(h.shape[1] * (1 - crop_y / 2))):  # Loop over columns
            if mask_f(h[i, j], s[i, j], v[i, j]):
                msk[i, j] = 255  # Set to 255 (white) if condition is true
            else:
                msk[i, j] = 0  # Set to 0 (black) if condition is false

    if display_mask:
        show_img("msk", msk)

    return msk


def lut_mask(image, window_size=15, ignore_percentage=0.1, peak_offset=0, curvature=70, apply_mask=False):
    """
    Perform color-segmentation using RGB channels distribution
    :param image: numpy.ndarray: image to be processed (use img = cv2.imread(path_to_image))
    :param window_size: int: default=15: hyperparameter; size of window for smoothing RGB curve with moving average
    :param ignore_percentage: float: default=0.1: hyperparameter; percentage of RGB values to ignore when locating peak
    :param peak_offset: int: default=0: hyperparameter; experimental value to manually shift RGB peak
    :param curvature: int: default=70: hyperparameter; curvature value used when applying RGB LUT
    :param apply_mask: bool or int: default=False: set as integer for a hard cutoff for white pixels, or False
    :return: mask: numpy.ndarray
    """
    # Get RGB Histogram
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(image_rgb)
    r_hist = cv2.calcHist([r], [0], None, [256], [0, 256])
    g_hist = cv2.calcHist([g], [0], None, [256], [0, 256])
    b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist = r_hist + g_hist + b_hist

    # Smooth with moving average
    i = 0
    moving_averages = []
    while i < len(hist) - window_size + 1:
        window_average = round(np.sum(hist[i:i + window_size]) / window_size, 2)
        moving_averages.append(window_average)
        i += 1

    # Peak is defined as first minima (ignoring first ignore_percentage%)
    peak = 0
    for peak in range(int(len(moving_averages) * ignore_percentage), len(moving_averages)):
        # if previous RGB value is greater, and next is smaller
        if (moving_averages[peak] - moving_averages[peak - 1]) > 0 > (
                moving_averages[peak] - moving_averages[peak + 1]):
            break
    peak += peak_offset

    # Finally, apply LUT
    lut_in = [peak, peak + curvature, 255]
    lut_out = [0, 255, 255]
    lut_8u = np.interp(np.arange(0, 256), lut_in, lut_out).astype(np.uint8)
    msk = cv2.LUT(image, lut_8u)

    # Apply mask keeping only near-white pixels
    if apply_mask:
        lwr = np.array([255 - apply_mask, 255 - apply_mask, 255 - apply_mask], dtype=np.int32)
        upr = np.array([255, 255, 255], dtype=np.int32)
        return cv2.inRange(msk, lwr, upr)
    else:
        return msk


def grayscale_mask(image, grayscale_tolerance=20):
    """
    Remove (set as black) non-grayscale pixels within a tolerance
    :param image: numpy.ndarray: image to be processed (use img = cv2.imread(path_to_image))
    :param grayscale_tolerance: maximum grayscale distance (0 means only keep grayscale)
    :return: mask: numpy.ndarray
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(rgb_image)

    # Iterate over each pixel and remove pixels above threshold
    for i in range(r.shape[0]):  # Loop over rows
        for j in range(r.shape[1]):  # Loop over columns
            pixel_r = int(r[i, j])
            pixel_g = int(g[i, j])
            pixel_b = int(b[i, j])
            if (pixel_r - pixel_g) ** 2 + (pixel_g - pixel_b) ** 2 + (
                    pixel_b - pixel_r) ** 2 > grayscale_tolerance ** 2:
                r[i, j] = 0
                g[i, j] = 0
                b[i, j] = 0

    result = cv2.merge((b, g, r))
    return result


def detect_chessboard(image, mask_generator, pattern_size=(4, 4), crop_x=0.0, crop_y=0.0, display=True, save_path=None):
    """
    Detect chessboard given image and mask generator
    :param image: numpy.ndarray: image to be processed (use img = cv2.imread(path_to_image))
    :param mask_generator: function: mask generator function (use get_mask_function_generator(bin_tensor))
    :param pattern_size: tuple: default=(4,4): pattern size to detect
    :param crop_x: float: default = 0: percentage of width to crop before detecting chessboard
    :param crop_y: float: default = 0: percentage of height to crop before detecting chessboard
    :param display: bool: default=True: display mask after applying color segmentation
    :param save_path: str: default=None: path to save mask image. Leave as None for no saving.
    """
    assert 0 <= crop_x <= 1
    assert 0 <= crop_y <= 1

    # Get image means and stds
    pixels = get_pixels_no_transparency(image)
    p_means = np.mean(pixels, axis=0)
    p_stds = np.std(pixels, axis=0)

    # get mask
    msk = get_mask(image, mask_generator(p_means, p_stds), crop_x=crop_x, crop_y=crop_y, display_mask=display)

    # detect chessboard
    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
    dlt = cv2.dilate(msk, krn, iterations=5)
    res = 255 - cv2.bitwise_and(dlt, msk)
    res = np.uint8(res)
    ret, corners = cv2.findChessboardCorners(res, pattern_size,
                                             flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    # display and print if display=True
    if ret:
        fnl = cv2.drawChessboardCorners(image, pattern_size, corners, ret)  # final image with chessboard detected
        if display:
            print(ret, corners)
            show_img("img", fnl)
        if save_path:
            print(save_path)
            cv2.imwrite(save_path, fnl)

    return ret, corners


def detect_chessboard_lut(image, pattern_size, crop_x=0.0, crop_y=0.0, apply_grayscale=False, window_size=15,
                          ignore_percentage=0.1, peak_offset=0, curvature=70, apply_mask=False):
    """
    Detect chessboard using LUT mask
    :param image: numpy.ndarray: image to be processed (use img = cv2.imread(path_to_image))
    :param pattern_size: tuple: size of chessboard pattern (x, y)
    :param crop_x: float: default = 0: percentage of width to crop
    :param crop_y: float: default = 0: percentage of height to crop
    :param apply_grayscale: bool: default=False: filter out non-grayscale pixels before detecting chessboard
    :param window_size: int: default=15: hyperparameter; size of window for smoothing RGB curve with moving average
    :param ignore_percentage: float: default=0.1: hyperparameter; percentage of RGB values to ignore when locating peak
    :param peak_offset: int: default=0: hyperparameter; experimental value to manually shift RGB peak
    :param curvature: int: default=70: hyperparameter; curvature value used when applying RGB LUT
    :param apply_mask: bool or int: default=False: set as integer for a hard cutoff for white pixels, or False
    """
    assert 0 <= crop_x <= 1
    assert 0 <= crop_y <= 1

    if apply_grayscale:
        mask = grayscale_mask(image, grayscale_tolerance=80)
    else:
        mask = image

    crop = mask[int(mask.shape[0] * (crop_y / 2)): int(mask.shape[0] * (1 - crop_y / 2)),
           int(mask.shape[1] * (crop_x / 2)): int(mask.shape[1] * (1 - crop_x / 2))]

    mask2 = lut_mask(crop, window_size=window_size, ignore_percentage=ignore_percentage, peak_offset=peak_offset,
                     curvature=curvature, apply_mask=apply_mask)

    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    dlt = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, krn, iterations=1)
    res = 255 - cv2.bitwise_and(dlt, mask2)
    res = np.uint8(res)
    ret, corners = cv2.findChessboardCorners(res, pattern_size,
                                             flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        # Correct coordinates of corners due to cropping
        for c in corners:
            c[0][0] += (crop_x / 2) * image.shape[1]
            c[0][1] += (crop_y / 2) * image.shape[0]

    return ret, corners


def worker(i, p):
    if not os.path.exists("choke_signal"):
        r =  detect_chessboard_lut(p[0], p[1], crop_x=p[2], crop_y=p[3], apply_grayscale=p[4],
                                        peak_offset=p[5], curvature=p[6], apply_mask=p[7])
        if r[0]:
            open("choke_signal", 'w+').close()
        return i, r
    else:
        return i, (None, None)


def detect_chessboard_lut_search(image_path, parameters, pattern_size, crop_x, crop_y, save_image=False):
    """
    Call detect_chessboard_lut with multiple parameters (very high success rate)
    :param image_path: str: path to image to be processed
    :param parameters: list[tuple]: []
    :param pattern_size: tuple: size of chessboard pattern (x, y)
    :param crop_x: float: percentage of width to crop
    :param crop_y: float: percentage of height to crop
    :param save_image: bool: default=False: overwrite image_path with illustrated chessboard corners
    :return: status: either returns False if no chessboard is detected, or returns tuple:
                     (True, np.array of corners, tuple of parameters that succeeded: apply_grayscale, curvature, peak offset, apply strict mask)
    """
    assert 0 <= crop_x <= 1
    assert 0 <= crop_y <= 1
    assert not os.path.exists("choke_signal")

    img = cv2.imread(image_path)

    n = multiprocessing.cpu_count()
    p = [(i, (img, pattern_size, crop_x, crop_y) + parameters[i]) for i in range(len(parameters))]

    with multiprocessing.Pool(processes=n) as pool:
        result = pool.starmap(worker, p)

    try:
        os.remove("choke_signal")
    except OSError:
        pass

    _, r = zip(*result)
    ret, corners = next((x for x in r if x[0]), (False, None))
    if ret and save_image:
        cv2.imwrite(image_path, cv2.drawChessboardCorners(cv2.imread(image_path), pattern_size, corners, ret))
    return ret, corners


# Same as detect_chessboard_lut_search, but with no multi-threading (kept for testing purposes)
def detect_chessboard_lut_search_non_parallel(image_path, pattern_size, crop_x, crop_y, save_image=False):
    for apply_grayscale in (False, True):
        for peak_offset in (0, -30, -2, 10, -20, -10, 20):
            for apply_strict_mask in (False, True):
                for curvature in (70, 40, 80, 3, 24, 0, 6, 60, 12, 50):
                    ret, corners = detect_chessboard_lut(cv2.imread(image_path), pattern_size,
                                                         crop_x=crop_x, crop_y=crop_y,
                                                         apply_grayscale=apply_grayscale, peak_offset=peak_offset,
                                                         curvature=curvature, apply_mask=apply_strict_mask)
                    if ret:
                        if save_image:
                            cv2.imwrite(image_path,
                                        cv2.drawChessboardCorners(cv2.imread(image_path), pattern_size, corners, ret))
                        return True, corners, (apply_grayscale, curvature, peak_offset, apply_strict_mask)


def calibrate(image_corners, pattern_size, column, image_size, focal_length):
    """
    Calibrate camera
    :param image_corners: np.array: corners as detected by cv2.findChessboardCorners
    :param pattern_size: tuple: size of chessboard pattern (x, y) - must be same as detected pattern
    :param column: [float]: list of chessboard column coordinates (row coordinates are [1,2,...,n_rows])
    :param image_size: 2-tuple (int, int): image pixel resolution (w, h)
    :param focal_length: float: focal length of camera
    """
    assert pattern_size[1] == len(column)

    # Row y coordinates
    rows = range(1, pattern_size[0] + 1)

    # Generate list of planar coordinates with z = 0 (corrected later on)
    objp = []
    for r in rows:
        for c in column:
            objp.append([c, r, 0])
    objp = np.asarray([objp], dtype=np.float32)
    objpoints = [objp] * len(image_corners)

    camera_mat = np.asarray([[focal_length, 0, image_size[0] // 2],
                             [0, focal_length, image_size[1] // 2],
                             [0, 0, 1]], dtype=np.float32)

    return cv2.calibrateCamera(objpoints, image_corners, image_size, camera_mat, None)


# Generate parameters
def default_parameters():
    params = []
    for gray in (False, True):
        for peak_o in (0, -30, -2, 10, -20, -10, 20):
            for strict_mask in (False, True):
                for curv in (70, 40, 80, 3, 24, 0, 6, 60, 12, 50):
                    if gray:
                        if random.random() < 0.5:  # use half of parameters for apply_grayscale for performance
                            params.append((gray, peak_o, curv, strict_mask))
                    else:
                        params.append((gray, peak_o, curv, strict_mask))
    return params


if __name__ == '__main__':
    # Demo for detecting multiple chessboard patterns in a directory
    path = input("Enter glob pattern for .jpg image(s): ")
    chess_pattern_size = (4, 4)

    for f in glob.glob(path):
        status = detect_chessboard_lut_search(f, default_parameters(), chess_pattern_size, 0.3, 0.3)

        if not status[0]:
            print(f"{os.path.basename(f)} Failed")
        else:  # success
            print(f"{os.path.basename(f)} Success")
            show_img("img", cv2.drawChessboardCorners(cv2.imread(f), chess_pattern_size, status[1], True))

            corners_save_path = os.path.join(os.path.dirname(path), "numpy-corners")
            if not os.path.exists(corners_save_path):
                os.makedirs(corners_save_path)
            np.save(os.path.join(corners_save_path, os.path.basename(f) + ".npy"), status[1])
