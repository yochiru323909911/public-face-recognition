import cv2
import numpy as np
import dlib
from scipy.spatial.distance import euclidean
from face_align.app import align
import face_symmetry as fs
import math
from config import NUM_FEATURES, THRESHOLD, NUM_POINTS, RADIUS, RADIUS_COLOR, IRIS_TOLARENCE, IMAGE_WIDTH, \
    LEFT, RIGHT, T_MIN, T_MAX, FACIAL_LANDMARKS_IDXS, FEATURES_RANGE
# ======================================================================================================================


hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# ======================================================================================================================


def calculate_roundness_index(jowl_end_points):
    """
    Calculate the roundness index of a closed shape defined by a list of points.

    Args:
        jowl_end_points (list): List of points defining the closed shape.

    Returns:
        roundness_index (float): The calculated roundness index.
    """

    # Ensure the jowl_end_points form a closed shape
    if jowl_end_points[0] != jowl_end_points[-1]:
        jowl_end_points.append(jowl_end_points[0])

    # Calculate the perimeter
    perimeter = 0
    for i in range(len(jowl_end_points) - 1):
        x1, y1 = jowl_end_points[i]
        x2, y2 = jowl_end_points[i + 1]
        perimeter += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Calculate the area using the shoelace formula
    area = 0
    for i in range(len(jowl_end_points) - 1):
        x1, y1 = jowl_end_points[i]
        x2, y2 = jowl_end_points[i + 1]
        area += (x1 * y2 - x2 * y1)
    area = abs(area) / 2

    # Calculate the roundness index
    roundness_index = (4 * math.pi * area) / (perimeter ** 2)
    return roundness_index
# ======================================================================================================================


def compute_texture(image, radius=RADIUS, neighbors=NUM_POINTS):
    """
   Compute the texture histogram using Local Binary Pattern (LBP) for a given image.

   Args:
       image (numpy.ndarray): The input grayscale image.
       radius (int, optional): The radius of the circular neighborhood for LBP.
       neighbors (int, optional): The number of neighboring points for LBP.

   Returns:
       hist (numpy.ndarray): The histogram of values.
   """
    # Determine the dimensions of the input image.
    ysize, xsize = image.shape
    # define circle of symetrical neighbor points
    angles_array = 2 * np.pi / neighbors
    alpha = np.arange(0, 2 * np.pi, angles_array)
    # Determine the sample points on circle with radius R
    s_points = np.array([-np.sin(alpha), np.cos(alpha)]).transpose()
    s_points *= radius
    # s_points is a 2d array with 2 columns (y,x) coordinates for each cicle neighbor point
    # Determine the boundaries of s_points wich gives us 2 points of coordinates
    # gp1(min_x,min_y) and gp2(max_x,max_y), the coordinate of the outer block
    # that contains the circle points
    min_y = min(s_points[:, 0])
    max_y = max(s_points[:, 0])
    min_x = min(s_points[:, 1])
    max_x = max(s_points[:, 1])
    # Block size, each LBP code is computed within a block of size bsizey*bsizex
    # so if radius = 1 then block size equal to 3*3
    bsizey = np.ceil(max(max_y, 0)) - np.floor(min(min_y, 0)) + 1
    bsizex = np.ceil(max(max_x, 0)) - np.floor(min(min_x, 0)) + 1
    # Coordinates of origin (0,0) in the block
    origy = int(0 - np.floor(min(min_y, 0)))
    origx = int(0 - np.floor(min(min_x, 0)))
    # Minimum allowed size for the input image depends on the radius of the used LBP operator.
    if xsize < bsizex or ysize < bsizey:
        raise Exception('Too small input image. Should be at least (2*radius+1) x (2*radius+1)')
    # Calculate dx and dy: output image size
    # for exemple, if block size is 3*3 then we need to substract the first row and the last row which is 2 rows
    # so we need to substract 2, same analogy applied to columns
    dx = int(xsize - bsizex + 1)
    dy = int(ysize - bsizey + 1)
    # Fill the center pixel matrix C.
    C = image[origy:origy + dy, origx:origx + dx]
    # Initialize the result matrix with zeros.
    result = np.zeros((dy, dx), dtype=np.float32)
    for i in range(s_points.shape[0]):
        # Get coordinate in the block:
        p = s_points[i][:]
        y, x = p + (origy, origx)
        # Calculate floors, ceils and rounds for the x and ysize
        fx = int(np.floor(x))
        fy = int(np.floor(y))
        cx = int(np.ceil(x))
        cy = int(np.ceil(y))
        rx = int(np.round(x))
        ry = int(np.round(y))
        D = [[]]
        if np.abs(x - rx) < 1e-6 and np.abs(y - ry) < 1e-6:
            # Interpolation is not needed, use original datatypes
            N = image[ry:ry + dy, rx:rx + dx]
            D = (N >= C).astype(np.uint8)
        else:
            # interpolation is needed
            # compute the fractional part.
            ty = y - fy
            tx = x - fx
            # compute the interpolation weight.
            w1 = (1 - tx) * (1 - ty)
            w2 = tx * (1 - ty)
            w3 = (1 - tx) * ty
            w4 = tx * ty
            # compute interpolated image:
            N = w1 * image[fy:fy + dy, fx:fx + dx]
            N = np.add(N, w2 * image[fy:fy + dy, cx:cx + dx], casting="unsafe")
            N = np.add(N, w3 * image[cy:cy + dy, fx:fx + dx], casting="unsafe")
            N = np.add(N, w4 * image[cy:cy + dy, cx:cx + dx], casting="unsafe")
            D = (N >= C).astype(np.uint8)
        # Update the result matrix.
        v = 2 ** i
        result += D * v
        hist, _ = np.histogram(result.astype(np.uint8).ravel(), bins=256, range=[0, 256])
        return hist
# ======================================================================================================================


def find_landmarks(img, hog_face_detector, dlib_facelandmark):
    """
    Detect and retrieve facial landmarks using dlib's face detector and shape predictor.

    Args:
        img (numpy.ndarray): The input image containing a face.
        hog_face_detector (dlib.fhog_object_detector): HOG-based face detector from dlib.
        dlib_facelandmark (dlib.shape_predictor): Shape predictor for facial landmarks.

    Returns:
        landmarks (list): List of tuples representing detected facial landmark coordinates.
    """

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


def resize_face(colorfull, gray):
    """
    Resize and crop a face image while maintaining aspect ratio.

    Args:
        colorfull (numpy.ndarray): The input color face image.
        gray (numpy.ndarray): The corresponding grayscale face image.

    Returns:
        resized_gray (numpy.ndarray): Resized grayscale face image.
        resized_colorfull (numpy.ndarray): Resized color face image.
        landmark_coords (list): List of facial landmark coordinates in the resized image.
    """

    landmark_coords = find_landmarks(gray, hog_face_detector, dlib_facelandmark)

    if landmark_coords is None:
        return None, None, None

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
    landmark_coords = find_landmarks(resized_gray, hog_face_detector, dlib_facelandmark)

    return resized_gray, resized_colorfull, landmark_coords
# ======================================================================================================================


# Detect facial features
def detect_features(landmark_coords):
    """
    Detect and extract specific facial features from a set of facial landmarks.

    Args:
        landmark_coords (numpy.ndarray): An array of facial landmark coordinates.

    Returns:
        tuple: A tuple containing lists of detected facial features, including eyes, nose, mouth, eyebrows, and jaw.

    This function detects and extracts specific facial features from a given array of facial landmark coordinates.
    It categorizes the landmarks into different regions: eyes, nose, mouth, eyebrows, and jaw.

    The function then returns a tuple containing lists of the detected facial features, allowing further processing
    and analysis on each feature category.
    """

    # Initialize empty lists for each feature
    eyes = [0, 0]
    eyebrows = [0, 0]
    mouth = landmark_coords[FACIAL_LANDMARKS_IDXS["mouth"][0]:FACIAL_LANDMARKS_IDXS["mouth"][1]]
    eyes[LEFT] = landmark_coords[FACIAL_LANDMARKS_IDXS["left_eye"][0]:FACIAL_LANDMARKS_IDXS["left_eye"][1]]
    eyes[RIGHT] = landmark_coords[FACIAL_LANDMARKS_IDXS["right_eye"][0]:FACIAL_LANDMARKS_IDXS["right_eye"][1]]
    eyebrows[LEFT] = landmark_coords[FACIAL_LANDMARKS_IDXS["left_eyebrow"][0]:FACIAL_LANDMARKS_IDXS["left_eyebrow"][1]]
    eyebrows[RIGHT] = landmark_coords[FACIAL_LANDMARKS_IDXS["right_eyebrow"][0]:FACIAL_LANDMARKS_IDXS["right_eyebrow"][1]]
    nose = landmark_coords[FACIAL_LANDMARKS_IDXS["nose"][0]:FACIAL_LANDMARKS_IDXS["nose"][1]]
    jaw = landmark_coords[FACIAL_LANDMARKS_IDXS["jaw"][0]:FACIAL_LANDMARKS_IDXS["jaw"][1]]
    return eyes, nose, mouth, eyebrows, jaw
# ======================================================================================================================


def extract_features(eyes, nose, mouth, eyebrow, jaw):
    """
    Extract facial features from facial landmarks.

    Args:
        eyes (dict): A dictionary containing landmarks of the left and right eyes.
        nose (list): A list of landmarks representing the nose.
        mouth (list): A list of landmarks representing the mouth.
        eyebrow (dict): A dictionary containing landmarks of the left and right eyebrows.
        jaw (list): A list of landmarks representing the jaw.

    Returns:
        list: A list of extracted facial features.

    This function extracts various facial features from the given facial landmarks.
    The extracted features include the aspect ratio of the jaw, lips ratio, distance between eyes,
    distance from nose to chin, distance between nose and mouth, length and width of nose,
    distance between eyebrows, and thickness of the lips.

    The features are calculated based on the provided facial landmarks using mathematical calculations
    such as Euclidean distance and geometric properties.
    """

    # Find the aspect ratio
    features = [euclidean(jaw[1], jaw[-1]) / euclidean(jaw[1], jaw[8])]

    # Find the lips ratio
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

    # Extract the distance between the nose and mouth
    nose_x, nose_y = nose[6]
    mouth_x, mouth_y = mouth[3]
    distance = np.sqrt((nose_x - mouth_x) ** 2 + (nose_y - mouth_y) ** 2)
    features.append(distance)

    # Extract length of nose
    features.append(euclidean(nose[6], nose[0]))

    # Extract width of nose
    features.append(euclidean(nose[4], nose[8]))

    # Extract the distance between the eyebrows
    features.append(euclidean(eyebrow[RIGHT][-1], eyebrow[LEFT][0]))

    # Extract the thickness of the lips
    features.append(euclidean(mouth[2], mouth[13]))
    features.append(euclidean(mouth[3], mouth[14]))
    features.append(euclidean(mouth[8], mouth[18]))
    return features
# ======================================================================================================================


def compare_features(f1, f2):
    """
    Compute the Euclidean distance between two feature vectors.

    Args:
        f1 (list): The first feature vector.
        f2 (list): The second feature vector.

    Returns:
        float: The Euclidean distance between the two feature vectors.

    This function calculates the Euclidean distance between two feature vectors 'f1' and 'f2'.
    The feature vectors are assumed to be lists of numerical values.

    The Euclidean distance is computed by iterating over the elements of both feature vectors and
    calculating the squared difference between corresponding elements. The square root of the sum
    of squared differences is then returned as the Euclidean distance.
    """

    distance = np.sqrt(np.sum([(f1[i] - f2[i]) ** 2 for i in range(len(f1))]))
    return distance

# ======================================================================================================================


def find_skin_color(image, nose, eyebrow, jaw):
    """
    Find the average 'A' and 'B' color channel values in the LAB color space for the skin region.

    Args:
        image (numpy.ndarray): The input image in BGR color space.
        nose (list): A list of nose landmarks.
        eyebrow (dict): A dictionary containing right and left eyebrow landmarks.
        jaw (list): A list of jaw landmarks.

    Returns:
        tuple: A tuple containing the average 'A' and 'B' color channel values in the LAB color space of the skin.

    This function calculates the average 'A' and 'B' color channel values in the LAB color space for the skin region.
    The skin region is divided into different subregions: right cheek, left cheek, forehead, and nose.

    The function calculates the average color for each subregion using the respective functions, such as 'find_cheek_color'
    for the cheeks, 'find_forehead_color' for the forehead, and 'get_nose_mean_color' for the nose.

    The final skin color is determined by averaging the color values of the right cheek, left cheek, forehead, and nose..
    """

    # Right cheek average color
    right_cheek_a, right_cheek_b = find_cheek_color(image, nose[2], jaw[1])

    # Left cheek average color
    left_cheek_a, left_cheek_b = find_cheek_color(image, jaw[-2], nose[2])

    # Forehead average color
    forehead_a, forehead_b = find_forehead_color(image, eyebrow[RIGHT], eyebrow[LEFT])

    # Nose average color
    nose_a, nose_b = get_nose_mean_color(image, nose)

    # Calculate the average skin color
    skin_color_a = (right_cheek_a + left_cheek_a + forehead_a + nose_a) // 4
    skin_color_b = (right_cheek_b + left_cheek_b + forehead_b + nose_b) // 4

    return [skin_color_a, skin_color_b]
# ======================================================================================================================


def find_cheek_color(image, organ1, organ2):
    """
    Find the average 'A' and 'B' color channel values in the LAB color space of the cheek region.

    Args:
        image (numpy.ndarray): The input image in BGR color space.
        organ1 (tuple): The coordinates of the first landmark representing a cheek-related feature.
        organ2 (tuple): The coordinates of the second landmark representing a cheek-related feature.

    Returns:
        tuple: A tuple containing the average 'A' and 'B' color channel values in the LAB color space of the cheek.

    This function calculates the average 'A' and 'B' color channel values in the LAB color space for the cheek region.
    The region is determined by the coordinates of two landmarks representing features associated with the cheeks.

    The function calculates the center point of the cheek region by finding the midpoint between the two provided landmarks.
    The 'calculate_lab' function is then used to calculate the average 'A' and 'B' values for the specified cheek region.
    """

    cheek_center = [(organ1[0] + organ2[0]) // 2, (organ1[1] + organ2[1]) // 2]
    return calculate_lab(image, cheek_center[0], cheek_center[1])
# ======================================================================================================================


def find_forehead_color(image, right_eyebrow, left_eyebrow):
    """
    Find the average 'A' and 'B' color channel values in the LAB color space of the forehead region.

    Args:
        image (numpy.ndarray): The input image in BGR color space.
        right_eyebrow (list): A list of landmarks representing the right eyebrow.
        left_eyebrow (list): A list of landmarks representing the left eyebrow.

    Returns:
        tuple: A tuple containing the average 'A' and 'B' color channel values in the LAB color space of the forehead.

    This function calculates the average 'A' and 'B' color channel values in the LAB color space for the forehead region.
    The region is determined by the coordinates of the right and left eyebrows' outermost points.

    The function calculates the starting point (x, y) for the forehead region, which is a few pixels above the higher
    point of the eyebrows. The width of the forehead region is determined as half of the distance between the outermost
    points of the eyebrows. The 'calculate_lab' function is then used to calculate the average 'A' and 'B' values
    for the specified forehead region.
    """

    y = max(right_eyebrow[2][1], left_eyebrow[2][1]) - 10
    eyebrows_dis = (left_eyebrow[0][0] - right_eyebrow[-1][0]) // 2
    x = right_eyebrow[-1][0] + eyebrows_dis - 2
    return calculate_lab(image, x, y)
# ======================================================================================================================


def calculate_lab(image, x, y):
    """
    Calculate the average 'A' and 'B' color channel values in the LAB color space around a given pixel.

    Args:
        image (numpy.ndarray): The input image in BGR color space.
        x (int): The x-coordinate of the central pixel.
        y (int): The y-coordinate of the central pixel.

    Returns:
        tuple: A tuple containing the average 'A' and 'B' color channel values in the LAB color space.

    This function calculates the average 'A' and 'B' color channel values in the LAB color space for a
    region around a specified central pixel. The region is defined by a square with a side length of 'RADIUS_COLOR',
    centered at the given pixel coordinates (x, y).

    The function iterates through the neighboring pixels within the defined region, extracts their 'A' and 'B'
    values in the LAB color space, and accumulates these values. After iterating through all the pixels,
    the average 'A' and 'B' values are computed by dividing the accumulated values by the number of pixels
    in the region.
    """
    # Convert BGR to LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Extract LAB values of the central pixel
    l, a, b = lab_image[y, x]
    l, a, b = np.int32(l), np.int32(a), np.int32(b)

    for i in range(RADIUS_COLOR):
        for j in range(RADIUS_COLOR):
            # Extract LAB values for the neighboring pixel
            l1, a1, b1 = lab_image[y + j, x + i]
            a += a1
            b += b1
            # Change the pixel value to (1, 128, 128) in the LAB color space
            lab_image[y + j, x + i] = (1, 128, 128)

    # Calculate the average LAB values
    a /= RADIUS_COLOR**2
    b /= RADIUS_COLOR**2

    return a, b
# ======================================================================================================================


def find_eye_color(image, eye):
    """
    Find the average color of the iris region in an eye image.

    Args:
        image (numpy.ndarray): The input image in BGR color space.
        eye (list): A list of eye landmarks.

    Returns:
        tuple or None: A tuple containing the mean 'A' and 'B' color channel values in LAB color space,
                       or None if the iris region is not found.

    This function extracts the iris region from the eye image using the provided eye landmarks.
    It converts the iris region to the LAB color space and calculates the mean 'A' and 'B' color
    channel values using a binary mask based on the L channel values.

    If the iris region is not found (e.g., when the eyes are closed), the function returns None.
    """
    # Load the iris region (with pupil) as an RGB image
    iris_region = image[int(eye[1][1]):int(eye[5][1]), int(eye[1][0]):int(eye[2][0])]
    if iris_region.size == 0:
        print("Open the eyes!😠")
        return None

    # Convert the image to LAB color space
    lab = cv2.cvtColor(iris_region, cv2.COLOR_BGR2LAB)

    # Split the LAB image into its channels
    l, a, b = cv2.split(lab)

    # Create a binary mask where the L channel values are greater than a threshold
    mask = (l > np.mean(l) - IRIS_TOLARENCE).astype(np.uint8)

    # Calculate the mean A and B values using the mask
    a_mean = np.mean(a * mask)
    b_mean = np.mean(b * mask)

    return a_mean, b_mean
# ======================================================================================================================


def normalize_res(features):
    """
    Normalize a list of features using a predefined range and target range.

    Args:
        features (list): A list of features to be normalized.

    Returns:
        list: A list of normalized features.

    This function normalizes a list of features using a predefined range for each feature and scaling
    the normalized values to a specified target range. The input 'features' list contains the feature
    values to be normalized.

    For each feature, the minimum and maximum values from the predefined range are extracted. The feature
    value is then scaled using the formula: normalized_value = (feature - min_val) / (max_val - min_val)
    * (T_MAX - T_MIN) + T_MIN, where T_MIN and T_MAX are the target range values.
    """
    normalize = []
    for feature in range(NUM_FEATURES):  # Loop through specific features
        min_val = FEATURES_RANGE[feature]["min"]
        max_val = FEATURES_RANGE[feature]["max"]
        normalized_value = (features[feature] - min_val) / (max_val - min_val) * (T_MAX - T_MIN) + T_MIN
        normalize.append(normalized_value)
    return normalize

# ======================================================================================================================


def get_nose_mean_color(image, nose):
    """
    Calculate the mean color of the nose region in an image.

    Args:
        image (numpy.ndarray): The input image in BGR color space.
        nose (list): A list of nose landmarks.

    Returns:
        tuple: A tuple containing the mean values of the 'A' and 'B' channels in LAB color space.

    This function extracts a region of interest (ROI) around the nose points from the given image.
    It then converts the ROI to the LAB color space and computes the mean values of the 'A' and 'B'
    color channels. The calculated means represent the average color of the nose region.

    Note: The constants 'RADIUS_COLOR' and 'cv2.COLOR_BGR2LAB' are assumed to be defined elsewhere in the codebase.
    """
    # Extract the second and third points of the nose
    nose_pt_2 = nose[1]  # Second point
    nose_pt_3 = nose[2]  # Third point

    # Define the ROI rectangle around the nose points
    roi_x = int(nose_pt_2[0] - 2.5)  # Left x-coordinate of ROI
    roi_y = int(min(nose_pt_2[1], nose_pt_3[1]))  # Top y-coordinate of ROI
    roi_width = RADIUS_COLOR  # Width of the ROI
    roi_height = int(abs(nose_pt_3[1] - nose_pt_2[1]))  # Height of the ROI

    # Extract the ROI
    roi = image[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]

    # Convert ROI to LAB color space
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

    # Compute the mean of A and B channels
    mean_a = np.mean(roi_lab[:, :, 1])
    mean_b = np.mean(roi_lab[:, :, 2])

    return mean_a, mean_b
# ======================================================================================================================


def calculate_angle(points_list):
    """
    Calculate the angle between three points in a two-dimensional space.

    Args:
        points_list (list of tuples): A list containing three tuples, each representing a point (x, y).

    Returns:
        float: The calculated angle in degrees between the line segments connecting the three points.

    This function calculates the angle formed by three points in a two-dimensional space.
    The input 'points_list' is a list of three tuples, where each tuple represents a point's (x, y) coordinates.

    The angle calculation involves calculating vectors AB and BC using the given points, calculating the dot
    product of these vectors, and then using the dot product to determine the angle between the vectors.
    The calculated angle is returned in degrees.
    """
    a, b, c = points_list

    # Calculate vectors AB and BC
    ab = (b[0] - a[0], b[1] - a[1])
    bc = (c[0] - b[0], c[1] - b[1])

    # Calculate the dot product of AB and BC
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]

    # Calculate the magnitudes of AB and BC
    magnitude_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    # Calculate the angle in radians using the arccos function
    angle_radians = math.acos(dot_product / (magnitude_ab * magnitude_bc))

    # Convert the angle from radians to degrees
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees
# ======================================================================================================================


def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points in a two-dimensional space.

    Args:
        point1 (tuple): Coordinates of the first point (x, y).
        point2 (tuple): Coordinates of the second point (x, y).

    Returns:
        float: The calculated Euclidean distance between the two points.

    This function calculates the Euclidean distance between two points in a two-dimensional space.
    The input tuples 'point1' and 'point2' contain the (x, y) coordinates of the respective points.

    The Euclidean distance is computed using the formula: distance = sqrt((x2 - x1)^2 + (y2 - y1)^2),
    where (x1, y1) and (x2, y2) are the coordinates of the two points.
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
# ======================================================================================================================


def angle_between_three_points(p1, p2, p3):
    """
    Calculate the angle between three points in a two-dimensional plane.

    Args:
        p1 (tuple): Coordinates of the first point (x, y).
        p2 (tuple): Coordinates of the second point (x, y), usually the vertex of the angle.
        p3 (tuple): Coordinates of the third point (x, y).

    Returns:
        float: The calculated angle in degrees between the lines formed by the three points.

    This function calculates the angle formed by three points in a two-dimensional plane. The angle
    is determined by the lines connecting the second point (usually the vertex) to the other two points.
    The input tuples 'p1', 'p2', and 'p3' contain the (x, y) coordinates of the respective points.

    The angle calculation is based on the difference in the arctangent (atan2) angles between the lines
    connecting the vertex point 'p2' to the other two points 'p1' and 'p3'.
    """

    angle_radians = math.atan2(p3[1] - p2[1], p3[0] - p2[0]) - math.atan2(p1[1] - p2[1], p1[0] - p2[0])
    return math.degrees(angle_radians)
# ======================================================================================================================


def calculate_nasolabial_angle(nose, mouth):
    """
     Calculate the angle between the nose, upper lip, and the bottom of the nose.

     Args:
         nose (list): A list of nose landmarks.
         mouth (list): A list of mouth landmarks.

     Returns:
         float: The calculated angle between the nasal tip, upper lip midpoint, and the bottom of the nose.

     This function computes the angle formed by three landmarks: the nasal tip, the midpoint of the upper lip,
     and the bottom of the nose. The provided nose and mouth landmark data is used to determine these points.
     The resulting angle represents the angle between the upper lip and the bottom of the nose, known as
     the nasolabial angle.
     """

    nasal_tip = nose[3]
    upper_lip_midpoint = ((mouth[2][0] + mouth[3][0]) // 2, (mouth[2][1] + mouth[3][1]) // 2)
    bottom_of_nose = nose[6]

    # Calculate the angle
    nasolabial_angle = angle_between_three_points(nasal_tip, upper_lip_midpoint, bottom_of_nose)
    return nasolabial_angle
# ======================================================================================================================


def calculate_brow_angle(eyebrows, nose):
    """
    Calculate the angle between the eyebrows and the bridge of the nose.

    Args:
        eyebrows (list): A list of eyebrow landmarks.
        nose (list): A list of nose landmarks.

    Returns:
        float: The calculated angle between the inner eyebrow corner, bridge of the nose, and outer eyebrow corner.

    This function calculates the angle formed by three landmarks: the inner corner of the eyebrow,
    the bridge of the nose, and the outer corner of the eyebrow. The provided eyebrow and nose landmark
    data is used to determine these points. The resulting angle represents the inclination or tilt of
    the eyebrows relative to the nose.
    """
    # Extract landmarks of interest
    inner_eyebrow_corner = eyebrows[RIGHT][-1]
    outer_eyebrow_corner = eyebrows[LEFT][0]
    bridge_of_nose = nose[0]

    # Calculate the angle
    brow_angle = angle_between_three_points(inner_eyebrow_corner, bridge_of_nose, outer_eyebrow_corner)
    return brow_angle
# ======================================================================================================================


def extract_angles(features, eyes, nose, mouth, eyebrow, jaw):
    """
    Extract various facial angle-based features from facial landmarks.

    Args:
        features (list): A list to store the extracted facial angle features.
        eyes (list): A list of eye landmarks.
        nose (list): A list of nose landmarks.
        mouth (list): A list of mouth landmarks.
        eyebrow (list): A list of eyebrow landmarks.
        jaw (list): A list of jaw landmarks.

    This function computes and appends several facial angle-based features to the 'features' list.
    The angles and measurements calculated represent different aspects of the face's structure
    and shape. The input landmarks for eyes, nose, mouth, eyebrow, and jaw are used to calculate
    these features.

    The computed features include the distance between eyes and the above of eyebrows, the average
    of eyes width, the roundness of the jaw, the shape of eyes, the roundness of the nose tip, the
    nasolabial angle, angles of the upper and under lip sides, and the brow angle.
    """
    # Distance between eyes and the above of eyebrows
    features.append((euclidean(eyes[RIGHT][3], eyebrow[LEFT][4]) + euclidean(eyes[LEFT][0], eyebrow[LEFT][0])) / 2)

    # Average of eyes width
    features.append(((euclidean(eyes[RIGHT][0], eyes[RIGHT][3]) + euclidean(eyes[LEFT][0], eyes[LEFT][3])) / 2))

    # Roundness of jaw
    features.append(calculate_roundness_index(jaw))

    # Shape of eyes
    features.append(eyes[RIGHT][0][1] - eyes[RIGHT][3][1])

    # Roundness of nose tip
    features.append(calculate_angle(nose[5:-1]))

    features.append(calculate_nasolabial_angle(nose, mouth))

    # Angle of the upper lip side
    features.append((calculate_angle([mouth[2], mouth[0], mouth[12]]) +
                     calculate_angle([mouth[4], mouth[6], mouth[16]])) / 2)

    # Angle of the under lip side
    features.append((calculate_angle([mouth[11], mouth[0], mouth[12]]) +
                     calculate_angle([mouth[7], mouth[6], mouth[16]])) / 2)

    features.append(calculate_brow_angle(eyebrow, nose))
# ======================================================================================================================


def recognize_face(image):
    """
    Recognize a face in the given image using a combination of features and comparisons.

    Args:
        image_name (str): The filename of the input image containing a face.

    Returns:
        result_features (numpy.ndarray or None): Combined feature vector and histogram for the face.
                                                None if face detection or landmark extraction fails.
    """

    rotated_gray, rotated_color = align(hog_face_detector, dlib_facelandmark, image, 4)

    if rotated_gray is None:
        return None

    final_gray, final_color, landmarks_coords = resize_face(rotated_color, rotated_gray)

    if landmarks_coords is None:
        return None

    hist = compute_texture(final_gray)

    # Detect facial features
    eyes, nose, mouth, eyebrow, jaw = detect_features(landmarks_coords)

    features = find_skin_color(final_color, nose, eyebrow, jaw)

    left_eye_a, left_eye_b = find_eye_color(final_color, eyes[LEFT])
    right_eye_a, right_eye_b = find_eye_color(final_color, eyes[RIGHT])
    features.append((left_eye_a + right_eye_a) // 2)
    features.append((left_eye_b + right_eye_b) // 2)

    features += extract_features(eyes, nose, mouth, eyebrow, jaw)

    extract_angles(features, eyes, nose, mouth, eyebrow, jaw)

    normal_features = normalize_res(features)

    normal_features.append(fs.calculate_symmetry_score(landmarks_coords))

    return np.concatenate((hist, normal_features))  # Concatenate the histogram with the existing vector
# ======================================================================================================================


def unlock_ask(user_features, image_path):
    """
    Determine whether the provided user features match the features extracted from the image.

    Args:
        user_features (list): List of user's features for comparison.
        image_path (str): The filename of the input image containing a face.

    Returns:
        is_unlocked (bool): True if the features match and the user is unlocked, False otherwise.
        updated_user_features (list): Updated user features after averaging with new features.
    """
    new_features = recognize_face(image_path)
    if compare_features(list(new_features), list(user_features)) > THRESHOLD:
        return False, user_features
    return True, [(user_features[i] + new_features[i]) / 2 for i in range(len(new_features))]
# ======================================================================================================================


def register(images):
    """
  Process a list of images to extract and compute median features from recognized faces.

  Args:
      images (list): A list of images containing faces to be processed.

  Returns:
      list or None: A list containing computed median features for recognized faces,
                    or None if no features were extracted from any image.

  This function loops through a list of images, extracts features from recognized faces using
  the 'recognize_face' function, and then calculates the median value for each feature across
  all the recognized faces. The result is a list of median features. If no features were
  extracted from any image, the function returns None.
  """
    images_features = []  # Initialize an empty list to store extracted features from images

    # Loop through each image in the 'images' list
    for img in images:
        res = recognize_face(img)  # Call the 'recognize_face' function to extract features from the image

        # Check if features were successfully extracted from the image
        if res is not None:
            images_features.append(res)  # If features were extracted, add them to the 'images_features' list

    # Initialize a list 'features' to hold the processed feature data
    features = [[] for _ in range(len(images_features[0]))]

    # Loop through each feature index
    for feature in range(len(images_features[0])):
        # Loop through each set of image features
        for img_features in images_features:
            features[feature].append(img_features[feature])  # Collect the corresponding feature value for each image
        features[feature] = np.median(features[feature])  # Calculate the median of collected feature values

    # Check if no features were extracted from any image
    if len(features) == 0:
        return None  # If no features were extracted, return None

    return features  # Return the processed feature data




































































