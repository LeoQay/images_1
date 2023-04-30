"""MAGIC CONSTANTS"""

import numpy as np


HEXAGON_PARTICLE_THRESHOLD = 0.012

YELLOW_LOWER = np.array([17, 155, 155])
YELLOW_UPPER = np.array([35, 255, 255])

RED_LOWER = np.array([0, 153, 134])
RED_UPPER = np.array([9, 255, 255])

BLUE_LOWER = np.array([25, 0, 0])
BLUE_UPPER = np.array([255, 73, 200])

HEXAGON_ANGLE_EPSILON = 5

LEAVE_ONLY_BIG_CONTOURS = 0.2

BIG_COMPONENTS_RATIO = 6

IMAGE_SIZE = 1300000

##########################################################

COUNTER_TYPES = {
    '1': {
        'blue': ['crossed', 0, 2],
        'red': ['full', 1, 3],
        'yellow': ['full', 4, 5]
    },
    '2': {
        'blue': ['full', 0, 3],
        'red': ['full', 4, 5],
        'yellow': ['full', 1, 2]
    },
    '3': {
        'blue': ['full', 0, 5],
        'red': ['full', 3, 4],
        'yellow': ['full', 1, 2]
    },
    '4': {
        'blue': ['crossed', 0, 3],
        'red': ['full', 2, 4],
        'yellow': ['crossed', 1, 5]
    },
    '5': {
        'blue': ['full', 4, 5],
        'red': ['full', 0, 3],
        'yellow': ['full', 1, 2]
    },
    '6': {
        'blue': ['full', 2, 4],
        'red': ['crossed', 1, 5],
        'yellow': ['crossed', 0, 3]
    },
    '7': {
        'blue': ['full', 4, 5],
        'red': ['full', 1, 3],
        'yellow': ['crossed', 0, 2]
    },
    '8': {
        'blue': ['full', 4, 5],
        'red': ['crossed', 0, 2],
        'yellow': ['full', 1, 3]
    },
    '9': {
        'blue': ['full', 2, 4],
        'red': ['crossed', 0, 3],
        'yellow': ['crossed', 1, 5]
    },
    '10': {
        'blue': ['full', 1, 3],
        'red': ['crossed', 0, 2],
        'yellow': ['full', 4, 5]
    }
}

