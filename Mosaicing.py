import cv2
import pickle
from os import path
import numpy as np
import math
import glob


def show_img(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1600, 1200)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def zoom_around_center(image, center, zoom_factor):
    (h, w) = image.shape[:2]
    zoom_matrix = cv2.getRotationMatrix2D(center, 0, zoom_factor)
    zoomed_image = cv2.warpAffine(image, zoom_matrix, (w, h), flags=cv2.INTER_LINEAR)

    return zoomed_image


def apply_transformations(image, theta, theta_x, theta_y, s_x, s_y, n=10):
    # Rotation (theta)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -theta, 1)  # Negative for clockwise rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

    # X distortion (theta_x)
    m_width = np.float32([[1, np.tan(np.radians(theta_x)), 0],
                          [0, 1, 0]])
    distorted_width_image = cv2.warpAffine(rotated_image, m_width, (w, h))

    # Y distortion (theta_y)
    m_height = np.float32([[1, 0, 0],
                           [np.tan(np.radians(theta_y)), 1, 0]])
    distorted_image = cv2.warpAffine(distorted_width_image, m_height, (w, h))

    # Correct s_x and s_y
    if s_x < 0:
        s_x = -s_x
    if s_x < 1:
        s_x = 1 - (1 - s_x) ** n
    if s_y < 0:
        s_y = -s_y
    if s_y < 1:
        s_y = 1 - (1 - s_y) ** n

    stretched_width_image = cv2.resize(distorted_image, None, fx=s_x, fy=1)
    stretched_image = cv2.resize(stretched_width_image, None, fx=1, fy=s_y)
    return stretched_image


def parse_pickles(pickles, base_path, num_images=16):
    """
    Parse calibration data into list of n images
    :param pickles: list of pickle file names
    :param base_path: base path of raw jpg files
    :param num_images: int: number of images in pickles
    :return: list[tuple] of length num_images: (fname, mtx, dist, rvec, tvec, chess_center)
    :return: mean_x, mean_y, mean_z: average translation values in x,y,z directions
    """
    data = [None] * num_images
    mean_x = 0
    mean_y = 0
    mean_z = 0

    for p in pickles:
        fs = p.split(',')
        fs[0] = fs[0].split('(')[-1]
        fs[-1] = fs[-1].split(')')[0]

        with open(p, 'rb') as f:
            x = pickle.load(f)  # ret, mtx, dist, rvecs, tvecs, image_corners
        for i in range(len(fs)):
            data[int(fs[i])] = (path.join(base_path, fs[i]+'.jpg'),
                                x[1],
                                x[2],
                                x[3][i],
                                x[4][i],
                                np.mean(x[5][i], axis=0))
            mean_x += x[4][i][0][0]
            mean_y += x[4][i][1][0]
            mean_z += x[4][i][2][0]

    mean_x /= num_images
    mean_y /= num_images
    mean_z /= num_images

    return data, mean_x, mean_y, mean_z


def get_mosaic(pickles, base_path):
    """
    Parse calibration data into mosaic
    :param pickles: list of pickle file names
    :param base_path: base path of raw jpg files
    :return: int: status
    """
    data, mean_x, mean_y, mean_z = parse_pickles(pickles, base_path)
    if None in data:
        return -1

    for img_num in range(len(data)):
        filename, intrinsic, distortion, rotation_vec, translation_vec, chess_center = data[img_num]
        img = cv2.imread(filename)
        zoom = float(translation_vec[2][0] / mean_z)
        coord = (int(chess_center[0][0]), int(chess_center[0][1]))

        centered_img = zoom_around_center(img, coord, zoom)
        undistorted_img = cv2.undistort(centered_img, intrinsic, distortion, None, None)
        rot_matrix, _ = cv2.Rodrigues(rotation_vec)
        sy = math.sqrt(rot_matrix[0, 0] * rot_matrix[0, 0]
                       + rot_matrix[1, 0] * rot_matrix[1, 0])
        singular = sy < 1e-6  # singularity check

        if not singular:
            # No singularity
            theta_y = math.atan2(rot_matrix[2, 1], rot_matrix[2, 2])
            theta_x = math.atan2(-rot_matrix[2, 0], sy)
            theta = math.atan2(rot_matrix[1, 0], rot_matrix[0, 0])
        else:
            # Singular case (gimbal lock)
            theta_y = math.atan2(-rot_matrix[1, 2], rot_matrix[1, 1])
            theta_x = math.atan2(-rot_matrix[2, 0], sy)
            theta = 0.0

        # Correct in opposite direction
        theta = -theta
        theta_x = -theta_x
        theta_y = -theta_y

        # Stretch x and y
        s_x = mean_x / translation_vec[0][0]
        s_y = mean_y / translation_vec[1][0]

        # Warp image
        # show_img(f"und {img_num}", undistorted_img)
        warped_image = apply_transformations(undistorted_img, theta, theta_x, theta_y, s_x, s_y)
        (h, w) = warped_image.shape[:2]

        # Crop to remove black borders
        new_width = w // 2
        new_height = h // 2
        x1 = new_width - new_width // 2
        x2 = new_width + new_width // 2
        y1 = new_height - new_height // 2
        y2 = new_height + new_height // 2
        cropped = warped_image[y1:y2, x1:x2]

        show_img(f"final {img_num}", cropped)

    return 0


if __name__ == '__main__':
    # Base path
    bases = glob.glob("front/static/capture/*")
    print("Pick the capture you'd like to mosaic:")
    for idx, base in enumerate(bases):
        print(f"[{idx}] {base}")
    idx = input(">>> ")
    base = bases[int(idx)]

    pickle_paths = glob.glob(path.basename(base)+"*")
    status = get_mosaic(pickle_paths, base)
    if status == -1:
        print("Run calibration first from web console")
