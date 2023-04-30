import numpy as np
import cv2

import constants
import contour_processing
import image_processing


def get_mean_perimeter(counters):
    return np.array([cv2.arcLength(counter['hex'], True) for counter in counters]).mean()


def get_mean_area(counters):
    return np.array([cv2.contourArea(counter['hex']) for counter in counters]).mean()


def try_to_make_better_color(contours, hex_contour):
    areas = np.array([cv2.contourArea(contour) for contour in contours])
    inds = areas.argsort()
    areas = areas[inds]
    contours = [contours[idx] for idx in inds]
    if len(contours) >= 2:
        contours = [
            contour
            for idx, contour in enumerate(contours)
            if areas[idx] / areas[-1] > 1 / constants.BIG_COMPONENTS_RATIO
        ]
        if areas[0] / areas[1] <= 1 / constants.BIG_COMPONENTS_RATIO:
            contours = contours[1:]
    return contours


def get_counters(img):
    hex_contours = image_processing.apply_funcs(image_processing.bgr_to_gray(img), contour_processing.hexagons)
    blue = contour_processing.get_blue_contours(img)
    yellow = contour_processing.get_yellow_contours(img)
    red = contour_processing.get_red_contours(img)

    def find_best(contour):
        over = np.array([
            contour_processing.Get2PolygonIntersectArea(img.shape, h, contour)[0]
            for h in hex_contours
        ])
        index = over.argmax()
        if np.nonzero(over[index]):
            return index
        return None

    result = [
        {'hex': hex_contour, 'blue': [], 'yellow': [], 'red': []}
        for hex_contour in hex_contours
    ]

    for color_contour in blue:
        arg = find_best(color_contour)
        if arg is not None:
            result[arg]['blue'].append(color_contour)
    for color_contour in yellow:
        arg = find_best(color_contour)
        if arg is not None:
            result[arg]['yellow'].append(color_contour)
    for color_contour in red:
        arg = find_best(color_contour)
        if arg is not None:
            result[arg]['red'].append(color_contour)

    for counter in result:
        for col in ['blue', 'yellow', 'red']:
            counter[col] = try_to_make_better_color(counter[col], counter['hex'])

    return result


def repair_color_by_2_others(state, missed_color):
    def dist_mod_6(a, b):
        return min(abs(a - b), 6 - abs(a - b))

    fulled = set(range(6))
    not_missed_coords = []
    not_missed_colors = []
    for col, val in state.items():
        if col != missed_color:
            not_missed_colors.append(col)
            not_missed_coords.append(val[1:])
            fulled.discard(val[1])
            fulled.discard(val[2])
    fulled = sorted(fulled)

    if len(fulled) != 2:
        return

    dist = dist_mod_6(*fulled)
    if dist == 1:
        state[missed_color] = ['full'] + fulled
    elif dist == 2:
        mid = (fulled[0] + 1) % 6 if fulled[1] < 4 else (fulled[1] + 1) % 6
        for col, val in state.items():
            if col != missed_color:
                if mid in val[1:]:
                    mid = col
        mode = 'full' if state[mid][0] == 'crossed' else 'crossed'
        state[missed_color] = [mode] + fulled
    elif dist_mod_6(*not_missed_coords[0]) == 1:
        state[missed_color] = ['full'] + fulled
    elif state[not_missed_colors[0]][0] == 'crossed' and state[not_missed_colors[1]][0] == 'crossed':
        state[missed_color] = ['full'] + fulled
    else:
        state[missed_color] = ['crossed'] + fulled


def determine_counter_state(counter):
    simple_hex = contour_processing.get_simplified_hexagon(counter['hex'])
    if simple_hex is None:
        raise NotImplementedError('Less than 6 points')
    simple_hex = simple_hex.reshape(-1, 2)
    hex_means = np.array(simple_hex, dtype=float)
    hex_means[:-1, :] += simple_hex[1:, :]
    hex_means[-1, :] += simple_hex[0, :]
    hex_means /= 2

    state = {}
    missed_colors = []
    for col in ['blue', 'yellow', 'red']:
        blocks = counter[col]
        if len(blocks) == 0 or len(blocks) >= 3:
            missed_colors.append(col)
            state[col] = None
            continue
        elif len(blocks) == 1:
            rect = contour_processing.get_rect(blocks[0]).reshape(-1, 2)
            len1 = np.linalg.norm(rect[0] - rect[1])
            len2 = np.linalg.norm(rect[0] - rect[3])
            if len1 < len2:
                rect = rect[[3, 0, 1, 2], :]
            mean1 = (rect[1] + rect[2]) / 2
            mean2 = (rect[0] + rect[3]) / 2
            state[col] = ['full']
        else:
            rect1 = contour_processing.get_rect(blocks[0]).reshape(-1, 2)
            rect2 = contour_processing.get_rect(blocks[1]).reshape(-1, 2)
            mean1 = rect1.mean(axis=0)
            mean2 = rect2.mean(axis=0)
            state[col] = ['crossed']
        arg1 = np.linalg.norm(hex_means - mean1[None, :], axis=1).argmin()
        arg2 = np.linalg.norm(hex_means - mean2[None, :], axis=1).argmin()
        state[col] += sorted([arg1, arg2])

    if len(missed_colors) >= 2:
        raise NotImplementedError('I am unsure in two or more colors')
    elif len(missed_colors) == 1:
        repair_color_by_2_others(state, missed_colors[0])

    return state, np.array(hex_means, dtype=np.int32)


def determine_counter_type(state):
    if state['blue'] is None or state['yellow'] is None or state['red'] is None:
        return -1

    unique = set()
    for val in state.values():
        unique.add(val[1])
        unique.add(val[2])
    if sorted(unique) != list(range(6)):
        return -1

    permutation = list(range(6))

    def do_permutation():
        return {
            col: [value[0]] + sorted([permutation[value[1]], permutation[value[2]]])
            for col, value in state.items()
        }

    def do_shift():
        save = permutation[0]
        permutation[:-1] = permutation[1:]
        permutation[-1] = save

    for name, counter_type in constants.COUNTER_TYPES.items():
        for _ in range(6):
            cur = do_permutation()
            if cur == counter_type:
                return name, counter_type
            do_shift()

    return -1


def get_counter_type(counter):
    try:
        state, _ = determine_counter_state(counter)
        t = {'default': determine_counter_type(state)}
    except NotImplementedError:
        t = {'default': -1}

    if t['default'] == -1:
        for col in ['blue', 'yellow', 'red']:
            save = counter[col]
            counter[col] = []
            name = 'del_{}'.format(col)
            try:
                state, _ = determine_counter_state(counter)
            except NotImplementedError:
                t[name] = -1
            else:
                t[name] = determine_counter_type(state)
            if t[name] != -1:
                return t[name]
            counter[col] = save

    return t['default']
