import numpy as np
import cv2
from sklearn.cluster import KMeans

import constants
import image_processing


def find_contours(img):
    contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    return contours, hierarchy


def get_convex_contours(contours):
    return [cv2.convexHull(contour, False) for contour in contours]


def get_rect(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.int0(box)


def leave_only_main_contours(args):
    contours, hierarchy = args
    return [contour for i, contour in enumerate(contours) if hierarchy[0, i, 3] == -1]


def leave_only_area_threshold_contours(threshold, inv=False):
    def wrapper(contours):
        if not inv:
            return list(filter(lambda x: cv2.contourArea(x) >= threshold, contours))
        else:
            return list(filter(lambda x: cv2.contourArea(x) < threshold, contours))

    return wrapper


def leave_only_hexagon_similar_contours(contours):
    return contours


def leave_only_big_contours(contours):
    areas = np.array([cv2.contourArea(contour) for contour in contours])
    threshold = constants.LEAVE_ONLY_BIG_CONTOURS * areas.max()
    inds = np.arange(len(contours))[areas >= threshold]
    return [contours[idx] for idx in inds]


def distance(X, Y):
    answer = np.linalg.norm(X, axis=-1)[:, None] ** 2 + \
             np.linalg.norm(Y, axis=-1)[None, :] ** 2 - (2.0 * X) @ Y.T
    answer[answer < 0] = 0
    return np.sqrt(answer)


def get_simplified_hexagon(contour):
    contour = contour.reshape(-1, 2)
    N = contour.shape[0]
    if N < 6:
        return None
    sharp_angles_far_180 = []

    for i in range(-2, N - 2):
        c1, c2, c3 = contour[i], contour[i + 1], contour[i + 2]
        v1 = np.array(c1 - c2, dtype=float)
        v2 = np.array(c3 - c2, dtype=float)
        arccos = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        if np.abs(180 / np.pi * arccos - 180) > constants.HEXAGON_ANGLE_EPSILON:
            sharp_angles_far_180.append(c2)

    X = np.array(np.hstack(sharp_angles_far_180).reshape(-1, 2), dtype=float)
    points = KMeans(n_clusters=6, n_init=10).fit(X).cluster_centers_

    dists = distance(points, X)
    points = np.array(points, dtype=np.int32)
    indices = dists.argmin(axis=1).argsort()
    points = points[indices, :]

    return np.array(points.reshape(-1, 1, 2))


def get_blue_contours(img):
    contours = image_processing.apply_funcs(image_processing.bgr_to_gray(img), hexagons)
    blue = image_processing.find_blue(img, contours)
    blue = image_processing.apply_funcs(blue, [
        image_processing.closing(7)
    ])
    threshold = constants.HEXAGON_PARTICLE_THRESHOLD * np.array([
        cv2.contourArea(contour) for contour in contours
    ]).mean()
    blue_contours = image_processing.apply_funcs(blue, convex_figures + [
        leave_only_area_threshold_contours(threshold)
    ])

    return blue_contours


def get_yellow_contours(img):
    contours = image_processing.apply_funcs(image_processing.bgr_to_gray(img), hexagons)
    yellow = image_processing.find_yellow(img, contours)
    yellow = image_processing.apply_funcs(yellow, [
        image_processing.closing(7)
    ])
    threshold = constants.HEXAGON_PARTICLE_THRESHOLD * np.array([
        cv2.contourArea(contour) for contour in contours
    ]).mean()
    yellow_contours = image_processing.apply_funcs(yellow, convex_figures + [
        leave_only_area_threshold_contours(threshold)
    ])

    return yellow_contours


def get_red_contours(img):
    contours = image_processing.apply_funcs(image_processing.bgr_to_gray(img), hexagons)
    red = image_processing.find_red(img, contours)
    red = image_processing.apply_funcs(red, [
        image_processing.closing(7)
    ])
    threshold = constants.HEXAGON_PARTICLE_THRESHOLD * np.array([
        cv2.contourArea(contour) for contour in contours
    ]).mean()
    red_contours = image_processing.apply_funcs(red, convex_figures + [
        leave_only_area_threshold_contours(threshold)
    ])

    return red_contours


def DrawPolygon(ImShape, Polygon, Color):
    Im = np.zeros(ImShape, np.uint8)
    cv2.fillConvexPoly(Im, Polygon, Color)
    return Im


def Get2PolygonIntersectArea(ImShape, Polygon1, Polygon2):
    Im1 = DrawPolygon(ImShape[:-1], Polygon1, 122)
    Im2 = DrawPolygon(ImShape[:-1], Polygon2, 133)
    Im = Im1 + Im2
    _, OverlapIm = cv2.threshold(Im, 200, 255, cv2.THRESH_BINARY)

    IntersectArea = np.sum(np.greater(OverlapIm, 0))
    contours, _ = cv2.findContours(OverlapIm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return 0.0, None

    contourArea = cv2.contourArea(contours[0])
    perimeter = cv2.arcLength(contours[0], True)

    return IntersectArea, perimeter, OverlapIm, contourArea


##########################################################

convex_figures = [
    find_contours,
    leave_only_main_contours,
    get_convex_contours
]

hexagons = image_processing.white_figures + convex_figures + [
    leave_only_big_contours,
    leave_only_hexagon_similar_contours
]

##########################################################
