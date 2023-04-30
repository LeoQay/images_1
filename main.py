import sys
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import constants
import image_processing
import counter_detector


def show(img, show_=True):
    max_edge = 8.0
    alpha = max_edge / max(img.shape[0], img.shape[1])
    size = (alpha * img.shape[1], alpha * img.shape[0])
    if show_:
        plt.figure(figsize=size)
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(image_processing.change_rgb(img))
    if show_:
        plt.show()


def load_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)

    n, m, _ = img.shape
    alpha = np.sqrt(constants.IMAGE_SIZE / (n * m))

    width = round(img.shape[1] * alpha)
    height = round(img.shape[0] * alpha)
    scale = (width, height)

    return cv2.resize(img, scale, interpolation=cv2.INTER_CUBIC)


def single_mode(filename):
    img = load_image(filename)

    counters = counter_detector.get_counters(img)
    types = [counter_detector.get_counter_type(counter) for counter in counters]

    if len(types) == 0:
        print('I don\'t recognized any counter')
    elif len(types) == 1:
        print('Counter type: {}'.format(types[0][0]))
        print('Repr: {}'.format(types[0][1]))
    else:
        print('Something strange, but here {} counters have been detected:'.format(len(types)))
        for t in types:
            print('Counter type: {}'.format(t[0]))
            print('Repr: {}'.format(t[1]))


def group_mode(filename, result_filename, show_=False):
    img = load_image(filename)

    n, m, _ = img.shape

    counters = counter_detector.get_counters(img)
    types = [counter_detector.get_counter_type(counter) for counter in counters]

    bugs = []

    perimeter = counter_detector.get_mean_perimeter(counters)
    side = round(perimeter / 6)

    for counter, t in zip(counters, types):
        if t == -1:
            bugs.append(counter)
            continue
        M = cv2.moments(counter['hex'])
        coords = (
            round(M['m10'] / M['m00']) - round(side * np.sqrt(3) / 2),
            round(M['m01'] / M['m00'])
        )
        img = image_processing.draw_text(
            img, '{}'.format(t[0]), coords, thickness=3, fontScale=1.65, color=(255, 255, 255))

    img = cv2.resize(img, (m, n), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(result_filename, img)
    if show_:
        show(img)


def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode', type=str, choices=['single', 'group'])
    parser.add_argument('--files', dest='files', nargs='*')
    parser.add_argument('--show', dest='show', type=bool, required=False, default=False)
    args = parser.parse_args(args=argv[1:]).__dict__
    if args['mode'] == 'single' and len(args['files']) != 1:
        raise ValueError('For single mode I need only ONE FILE PATH specified')
    if args['mode'] == 'group' and len(args['files']) != 2:
        raise ValueError('For group mode I need only TWO FILE PATHS specified')
    return args


def main(argv):
    args = get_args(argv)
    if args['mode'] == 'group':
        group_mode(*args['files'], args['show'])
    else:
        single_mode(args['files'][0])


if __name__ == "__main__":
    main(sys.argv)
