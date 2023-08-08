NUM_FEATURES = 23
THRESHOLD = 21000  # The threshold for two person
NUM_POINTS = 8  # Number of neighboring points to compare
RADIUS = 1  # Radius of the circular neighborhood
RADIUS_COLOR = 5
IRIS_TOLARENCE = 20
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
    {"min": 0.8, "max": 1.5},
    {"min": 50, "max": 210},
    {"min": 110, "max": 150},
    {"min": 75, "max": 170},
    {"min": 35, "max": 70},
    {"min": 150, "max": 215},
    {"min": 70, "max": 110},
    {"min": 50, "max": 100},
    {"min": 10, "max": 38},
    {"min": 8, "max": 32},
    {"min": 30, "max": 55},
    {"min": 0, "max": 255},
    {"min": 0, "max": 255},
    {"min": 130, "max": 400},
    {"min": 65, "max": 400},
    {"min": 0.5, "max": 5},
    {"min": 2, "max": 200},
    {"min": 0.1, "max": 40},
    {"min": 140, "max": 175},
    {"min": 115, "max": 170},
    {"min": 45, "max": 300},
]