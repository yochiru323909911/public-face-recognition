import cv2
import numpy as np
import dlib
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
from face_align.app import align

# ======================================================================================================================
THRESHOLD = 100 #the threshold with two images of the same person
WEIGHT_HISTOGRAM = 1000
NUM_POINTS = 8  # Number of neighboring points to compare
RADIUS = 1  # Radius of the circular neighborhood
NUM_FEATURES = 256 + 10 + 2  # The histogram and the geometric features and the nose AB color
IRIS_TOLARENCE = 20
IMAGE_WIDTH = 500
LEFT = 0
RIGHT = 1
T_MIN = 0  # for normalize the features
T_MAX = 100  # for normalize the features
allvectors = [[] for i in range(NUM_FEATURES)]
FEATURES_RANGE = [
    {"min": 0, "max": 255},
    {"min": 0, "max": 255},
    {"min": 40, "max": 200},
    {"min": 110, "max": 150},
    {"min": 80, "max": 155},
    {"min": 20, "max": 75},
    {"min": 140, "max": 210},
    {"min": 70, "max": 110},
    {"min": 8, "max": 40},
    {"min": 4, "max": 30},
    {"min": 22, "max": 55}
]
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# ======================================================================================================================


# Function to compute LBP value for a pixel
def compute_lbp(pixel, neighbors):
    lbp = 0
    for i, neighbor in enumerate(neighbors):
        if neighbor >= pixel:
            lbp |= 1 << i
    return lbp
# ======================================================================================================================


# Calculate LBP histogram
def calculate_lbp_histogram(image):
    lbp_image = np.zeros_like(image)
    rows, cols = image.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center = image[i, j]
            neighbors = [image[i + int(RADIUS * np.sin(2 * np.pi * k / NUM_POINTS)),
                              j + int(RADIUS * np.cos(2 * np.pi * k / NUM_POINTS))] for k in range(NUM_POINTS)]
            lbp_value = compute_lbp(center, neighbors)
            lbp_image[i, j] = lbp_value

    hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=[0, 256])
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Normalize the histogram

    return hist * WEIGHT_HISTOGRAM
# ======================================================================================================================


def find_landmarks(img, hog_face_detector, dlib_facelandmark):
    faces = hog_face_detector(img)
    if len(faces) == 0:
        print("No detection!☹")
        return None
    for face in faces:
        face_landmarks = dlib_facelandmark(img, face)
    landmarks = []
    for n in range(0, 68):
        x = face_landmarks.part(n).x
        y = face_landmarks.part(n).y
        landmarks.append((x, y))
    return landmarks
# ======================================================================================================================


def resize_face(colorfull, gray, hog_face_detector, face_landmarks):

    landmark_coords = find_landmarks(gray, hog_face_detector, face_landmarks)

    # Get the bounding box around the face
    x, y, w, h = cv2.boundingRect(np.array(landmark_coords))
    # Crop the image with the face
    face_gray = gray[y:y + h, x:x + w]
    face_colorfull = colorfull[y:y + h, x:x + w]
    height, width = face_gray.shape[:2]
    # Set the new width and calculate the new height to maintain aspect ratio
    new_height = int((height / width) * IMAGE_WIDTH)
    # Resize the image to the new dimensions
    resized_gray = cv2.resize(face_gray, (IMAGE_WIDTH, new_height))
    resized_colorfull = cv2.resize(face_colorfull, (IMAGE_WIDTH, new_height))
    landmark_coords = find_landmarks(resized_gray, hog_face_detector, face_landmarks)

    return resized_gray, resized_colorfull, landmark_coords
# ======================================================================================================================


# Detect facial features
def detect_features(landmark_coords):
    # Initialize empty lists for each feature
    eyes = [0, 0]
    nose = []
    mouth = []
    jaw = []
    eyebrow = [0, 0]
    FACIAL_LANDMARKS_IDXS = {
        "mouth": (48, 68),
        "right_eyebrow": (17, 22),
        "left_eyebrow": (22, 27),
        "right_eye": (36, 42),
        "left_eye": (42, 48),
        "nose": (27, 36),
        "jaw": (0, 17)
    }

    mouth = landmark_coords[FACIAL_LANDMARKS_IDXS["mouth"][0]:FACIAL_LANDMARKS_IDXS["mouth"][1]]
    eyes[LEFT] = landmark_coords[FACIAL_LANDMARKS_IDXS["left_eye"][0]:FACIAL_LANDMARKS_IDXS["left_eye"][1]]
    eyes[RIGHT] = landmark_coords[FACIAL_LANDMARKS_IDXS["right_eye"][0]:FACIAL_LANDMARKS_IDXS["right_eye"][1]]
    eyebrow[0] = landmark_coords[FACIAL_LANDMARKS_IDXS["left_eyebrow"][0]:FACIAL_LANDMARKS_IDXS["left_eyebrow"][1]]
    eyebrow[1] = landmark_coords[FACIAL_LANDMARKS_IDXS["right_eyebrow"][0]:FACIAL_LANDMARKS_IDXS["right_eyebrow"][1]]
    nose = landmark_coords[FACIAL_LANDMARKS_IDXS["nose"][0]:FACIAL_LANDMARKS_IDXS["nose"][1]]
    jaw = landmark_coords[FACIAL_LANDMARKS_IDXS["jaw"][0]:FACIAL_LANDMARKS_IDXS["jaw"][1]]
    return eyes, nose, mouth, eyebrow, jaw
# ======================================================================================================================


# Extract feature vectors
def extract_features(eyes, nose, mouth, eyebrow, jaw):

    features = []

    lips_ellipse = cv2.fitEllipse(np.array(mouth, dtype=np.float32))
    features.append(abs(lips_ellipse[1][0] - lips_ellipse[1][1]))  # extract the height (length of major axis)

    # Extract the distance between the eyes
    eye1, eye2 = eyes[RIGHT], eyes[LEFT]
    distance = euclidean(eye1[3], eye2[0])
    features.append(distance)

    # Extract the distance from the nose to the chin
    mouth_x, mouth_y = mouth[9]
    jaw_x, jaw_y = jaw[8]
    distance = np.sqrt((mouth_x - jaw_x) ** 2 + (mouth_y - jaw_y) ** 2)
    features.append(distance)

    # extract the distance between the nose and mouth
    nose_x, nose_y = nose[6]
    mouth_x, mouth_y = mouth[3]
    distance = np.sqrt((nose_x - mouth_x) ** 2 + (nose_y - mouth_y) ** 2)
    features.append(distance)

    # extract length of nose
    features.append(euclidean(nose[6], nose[0]))

    # extract width of nose
    features.append(euclidean(nose[4], nose[8]))

    # extract the thickness of the lips
    features.append(euclidean(mouth[2], mouth[13]))  # the thickness of the thick part the upper lip
    features.append(euclidean(mouth[3], mouth[14]))  # the thickness of the thin part the upper lip
    features.append(euclidean(mouth[8], mouth[18]))  # the thickness of the thick part the downer lip
    return features
# ======================================================================================================================


def get_outline(grayscale_image, points):
    # Create a mask image with the same size as the input image, filled with zeros (black)
    mask = np.zeros_like(grayscale_image)
    # Create a contour from the points using the dimensions of the image
    contour = np.array(points, np.int32).reshape((-1, 1, 2))
    # Draw the contour on the mask image
    cv2.drawContours(mask, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)
    # Use bitwise_and to extract only the region of the image defined by the contour
    organ_region = cv2.bitwise_and(grayscale_image, mask)
    # Use Canny edge detection to detect the edges of the lips region
    outline = cv2.Canny(organ_region, 50, 200)
    return outline


# ======================================================================================================================


# Compute the Euclidean distance between the feature vectors
def compare_features(f1, f2):
    distance = np.sqrt(np.sum([(f1[i] - f2[i]) ** 2 for i in range(len(f1))]))
    return distance
# ======================================================================================================================


def find_skin_color(image, nose, eyebrow, jaw):
    # right cheek color
    right_cheek = find_cheek_color(image, nose[2], jaw[1])
    # left cheek color
    left_cheek = find_cheek_color(image, jaw[-2], nose[2])
    # forehead color
    forehead = find_forehead_color(image, eyebrow[RIGHT], eyebrow[LEFT])
# ======================================================================================================================


def find_cheek_color(image, organ1, organ2):
    # find the color of the right chick
    # noseJawDistance_x, noseJawDistance_y = (organ1[0]+organ2[0])//2, abs(organ1[1]+organ2[1])//2
    # cheekCenter = [(organ2[0])+int(noseJawDistance_x//2), organ2[1]+int(noseJawDistance_y//2)]
    cheek_center = [(organ1[0]+organ2[0])//2, (organ1[1]+organ2[1])//2]
    return calculateBGR(image, cheek_center[0], cheek_center[1])
# ======================================================================================================================


def find_forehead_color(image, rightEyebrow, leftEyebrow):
    y = max(rightEyebrow[2][1], leftEyebrow[2][1]) - 10
    eyebrows_dis = (leftEyebrow[0][0]-rightEyebrow[-1][0])//2
    x = rightEyebrow[-1][0]+eyebrows_dis-2
    return calculateBGR(image, x, y)
# ======================================================================================================================


def calculateBGR(image, x, y):
    b, g, r = image[y, x]
    b, g, r = np.int32(b), np.int32(g), np.int32(r)
    for i in range(5):
        for j in range(5):
            b1, g1, r1 = image[y + j, x + i]
            b += b1
            g += g1
            r += r1
            image[y + j, x + i] = (1, 255, 1)
    b /= 25
    g /= 25
    r /= 25
    return b, g, r
# ======================================================================================================================


def findEyeColor(image, eye, features):
    # Load the iris region (with pupil) as an RGB image
    iris_region = image[int(eye[1][1]):int(eye[5][1]), int(eye[1][0]):int(eye[2][0])]
    if iris_region.size == 0:
        print("Open the eyes!😠")
        return
    # Convert the image to LAB color space
    lab = cv2.cvtColor(iris_region, cv2.COLOR_BGR2LAB)

    # Split the LAB image into its channels
    l, a, b = cv2.split(lab)

    # Create a binary mask where the L channel values are greater than 20
    mask = (l > np.mean(l)-IRIS_TOLARENCE).astype(np.uint8)

    # Calculate the mean A and B values using the mask
    a_mean = np.mean(a * mask)
    b_mean = np.mean(b * mask)

    # features.append(a_mean)
    # features.append(b_mean)
    # features.append(euclidean(a_mean, b_mean))

# ======================================================================================================================


# move on all the vectors of features
# take one feature from all the vectors and send it to normalize
def normalize_res(features):
    normalize = []
    for feature in range(len(features)):  # loop on specific feature
        min_val, max_val = FEATURES_RANGE[feature]["min"], FEATURES_RANGE[feature]["max"]
        normalize.append((features[feature] - min_val) / (max_val - min_val) * (T_MAX - T_MIN) + T_MIN)
    return normalize
# ======================================================================================================================

"""
def get_nose_mean_color(image, nose):
    # Extract the second and third points of the nose
    nose_pt_2 = nose[1]  # Second point
    nose_pt_3 = nose[2]  # Third point

    # Define the ROI rectangle around the nose points
    roi_x = int(nose_pt_2[0] - 2.5)  # Left x-coordinate of ROI
    roi_y = int(min(nose_pt_2[1], nose_pt_3[1]))  # Top y-coordinate of ROI
    roi_width = 5  # Width of the ROI
    roi_height = int(abs(nose_pt_3[1] - nose_pt_2[1]))  # Height of the ROI

    # Extract the ROI
    roi = image[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]

    # Convert ROI to LAB color space
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

    # Compute the mean color of the ROI in LAB
    mean_lab = np.mean(roi_lab, axis=(0, 1))

    return mean_lab
"""
def get_nose_mean_color(image, nose):
    # Extract the second and third points of the nose
    nose_pt_2 = nose[1]  # Second point
    nose_pt_3 = nose[2]  # Third point

    # Define the ROI rectangle around the nose points
    roi_x = int(nose_pt_2[0] - 2.5)  # Left x-coordinate of ROI
    roi_y = int(min(nose_pt_2[1], nose_pt_3[1]))  # Top y-coordinate of ROI
    roi_width = 5  # Width of the ROI
    roi_height = int(abs(nose_pt_3[1] - nose_pt_2[1]))  # Height of the ROI

    # Extract the ROI
    roi = image[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]

    # Convert ROI to LAB color space
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

    # Compute the mean of A and B channels
    mean_a = np.mean(roi_lab[:, :, 1])
    mean_b = np.mean(roi_lab[:, :, 2])

    return [mean_a, mean_b]
# ======================================================================================================================


# Main function
def recognize_face(image_name, hog_face_detector, face_landmarks):

    rotated_gray, rotated_color = align(image_name, 4)

    final_gray, final_color, landmarks_coords = resize_face(rotated_color, rotated_gray, hog_face_detector, face_landmarks)

    hist = calculate_lbp_histogram(final_gray)

    # Detect facial features
    eyes, nose, mouth, eyebrow, jaw = detect_features(landmarks_coords)

    features = get_nose_mean_color(final_color, nose)

    features += extract_features(eyes, nose, mouth, eyebrow, jaw)

    normal = normalize_res(features)

    for row, feature in zip(allvectors, normal):  # every row is a specific feature
        row.append(feature)

    # normal is the normalize features, and features is not normal
    return np.concatenate((hist, features))  # Concatenate the histogram with the existing vector

# ======================================================================================================================


def recognize(hog_face_detector, dlib_facelandmark, user1):

    v = []
    with open('symmetry_faces/{}.txt'.format(user1), 'r') as f:
        image_paths = ["symmetry_faces/{}/".format(user1) + line.strip() + ".jpg" for line in f]
    for image_path in image_paths:
        print(image_path)
        v.append(recognize_face(image_path, hog_face_detector, dlib_facelandmark))
    return v

# ======================================================================================================================


def permotations_vectors(person1, person2, is_the_same):
    v = []
    for i in range(len(person1)):
        start = i+1 if is_the_same else 0
        for j in range(start, len(person2)):
            v.append((compare_features(person1[i], person2[j])))
    return v
# ======================================================================================================================


def show_plt(f1, f2):
    plt.scatter(f1, np.zeros_like(f1), c='b', label="user1")
    plt.scatter(f2, np.zeros_like(f2), c='r', label="user2")
    plt.legend()
    plt.show()

# ======================================================================================================================


"""
The user want to register and send some images.
"""


def register(images):
    images_features = []
    for img in images:
        images_features.append(recognize_face(img, hog_face_detector, dlib_facelandmark))
    features = [0 for i in range(len(images_features[0]))]
    for feature in range(len(images_features[0])):
        for img_features in images_features:
            features[feature] += img_features
        features[feature] /= len(images_features)
    return features
# ======================================================================================================================


"""
The user is already register and want to open a locked application
"""


def unlock_ask(user_features, image):
    features_vec = recognize_face(image, hog_face_detector, dlib_facelandmark)
    if compare_features(user_features, features_vec) > THRESHOLD:
        return False, user_features
    return True, [(user_features[i] + features_vec[i]) / 2 for i in range(len(features_vec))]
# ======================================================================================================================







