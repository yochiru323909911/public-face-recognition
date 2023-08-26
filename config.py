NUM_FEATURES = 23
THRESHOLD = 20000  # The threshold for two person
NUM_POINTS = 8  # Number of neighboring points to compare
RADIUS = 1  # Radius of the circular neighborhood
RADIUS_COLOR = 5
IRIS_TOLARENCE = 20
SHINE_THRESHOLD = 25
IMAGE_WIDTH = 500
LEFT = 0
RIGHT = 1
T_MIN = 0  # For normalize the features
T_MAX = 100  # For normalize the features

FACIAL_LANDMARKS_IDXS = {
    "mouth": (48, 68),
    "right_eyebrow": (17, 22),
    "left_eyebrow": (22, 27),
    "right_eye": (36, 42),
    "left_eye": (42, 48),
    "nose": (27, 36),
    "jaw": (0, 17)
}

FEATURES_RANGE = [
    {"min": 0, "max": 255},
    {"min": 0, "max": 255},
    {"min": 0, "max": 255},
    {"min": 0, "max": 255},
    {"min": 1, "max": 2},
    {"min": 25, "max": 300},
    {"min": 100, "max": 200},
    {"min": 60, "max": 200},
    {"min": 20, "max": 150},
    {"min": 110, "max": 300},
    {"min": 65, "max": 200},
    {"min": 45, "max": 200},
    {"min": 5, "max": 70},
    {"min": 0, "max": 100},
    {"min": 10, "max": 100},
    {"min": 130, "max": 300},
    {"min": 55, "max": 200},
    {"min": 135, "max": 250},
    {"min": 0.75, "max": 0.9},
    {"min": -20, "max": 50},
    {"min": 0, "max": 200},
    {"min": -7, "max": 50},
    {"min": 120, "max": 250},
    {"min": 130, "max": 250}
]