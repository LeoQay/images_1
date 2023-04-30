import functools
import numpy as np
import cv2

import constants


def apply_funcs(original_img, funcs):
    current_img = original_img
    for func in funcs:
        current_img = func(current_img)
    return current_img


def opening(n):
    kernel = np.ones((n, n), dtype=np.uint8)

    def wrapper(img):
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    return wrapper


def closing(n):
    kernel = np.ones((n, n), dtype=np.uint8)

    def wrapper(img):
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return wrapper


def dilate(n):
    kernel = np.ones((n, n), dtype=np.uint8)

    def wrapper(img):
        return cv2.dilate(img, kernel, iterations=1)

    return wrapper


def erode(n):
    kernel = np.ones((n, n), dtype=np.uint8)

    def wrapper(img):
        return cv2.erode(img, kernel, iterations=1)

    return wrapper


def threshold(low, high, is_invert=False):
    mode = cv2.THRESH_BINARY_INV if is_invert else cv2.THRESH_BINARY

    def wrapper(img):
        _, img_thr = cv2.threshold(img, low, high, mode)
        return img_thr

    return wrapper


def change_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def bgr_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def bgr_to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def gauss_blur(n, sigma=0):
    def wrapper(img):
        return cv2.GaussianBlur(img, (n, n), sigma)

    return wrapper


def kirshe_operator(img):
    def rotate(arr):
        n = arr.shape[0]
        res = np.array(arr)
        res[0, :n - 1] = arr[0, 1:n]
        res[1:n, 0] = arr[:n - 1, 0]
        res[n - 1, 1:n] = arr[n - 1, :n - 1]
        res[:n - 1, n - 1] = arr[1:n, n - 1]
        return res

    kernel = np.array([
        [-3, -3, 5],
        [-3, 0, 5],
        [-3, -3, 5]
    ])
    kernels = []
    for _ in range(8):
        kernels.append(kernel)
        kernel = rotate(kernel)

    imgs = [np.abs(cv2.filter2D(img, -1, kernel)) for kernel in kernels]
    return functools.reduce(np.maximum, imgs, np.zeros(img.shape, dtype=np.uint8))


def xiufu(img_gray, img):
    _, mask = cv2.threshold(img_gray, 175, 255, cv2.THRESH_BINARY)
    res_ = cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_CUBIC)
    dst = cv2.inpaint(res_, mask, 10, cv2.INPAINT_TELEA)

    return dst


def bitwise_or(imgs):
    res = imgs[0]
    for img in imgs[1:]:
        res = cv2.bitwise_or(res, img)
    return res


def bitwise_and(imgs):
    res = imgs[0]
    for img in imgs[1:]:
        res = cv2.bitwise_and(res, img)
    return res


def canny(low, high):
    def wrapper(img):
        return cv2.Canny(img, low, high)

    return wrapper


def get_masked_image(img, contour, erosion=None, is_convex=True):
    mask = np.zeros(img.shape, dtype=np.uint8)
    if is_convex:
        cv2.fillConvexPoly(mask, contour, (255, 255, 255))
    else:
        cv2.fillPoly(mask, contour, (255, 255, 255))
    if erosion is not None:
        mask = erode(erosion)(mask)
    return cv2.bitwise_and(img, mask)


def mask_all_contours(img, contours, eroding_val=15):
    return bitwise_or([get_masked_image(img, contour, eroding_val) for contour in contours])


def draw_text(img, text, coords, thickness=2, color=(0, 255, 0), fontScale=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_copy = img.copy()
    cv2.putText(img_copy, text, coords, font, fontScale, color, thickness, cv2.LINE_AA)
    return img_copy


def find_blue(img, contours=None):
    hsv = bgr_to_hsv(img)
    mask = cv2.inRange(hsv, constants.BLUE_LOWER, constants.BLUE_UPPER)
    if contours is not None:
        mask = mask_all_contours(mask, contours, 0)
    return mask


def find_yellow(img, contours=None):
    hsv = bgr_to_hsv(img)
    mask = cv2.inRange(hsv, constants.YELLOW_LOWER, constants.YELLOW_UPPER)
    if contours is not None:
        mask = mask_all_contours(mask, contours, 0)
    return mask


def find_red(img, contours=None):
    hsv = bgr_to_hsv(img)
    mask = cv2.inRange(hsv, constants.RED_LOWER, constants.RED_UPPER)
    if contours is not None:
        mask = mask_all_contours(mask, contours, 0)
    return mask


def draw_contours(img_color, contours):
    image_copy = img_color.copy()
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    return image_copy


##########################################################

white_figures = [
    gauss_blur(5),
    threshold(100, 255, True),
]

##########################################################
##########################################################

