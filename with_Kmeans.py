from collections import OrderedDict
import cv2
import numpy as np
import dlib
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
import os
# ======================================================================================================================
NUM_FEATURES = 12
IRIS_TOLARENCE = 20
LEFT = 0
RIGHT = 1
T_MIN = 0  # for normalize the features
T_MAX = 100  # for normalize the features
allvectors = [[] for i in range(NUM_FEATURES)]
normalize_features = [[] for i in range(NUM_FEATURES)]
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


# Detect facial features
def detect_features(gray, hog_face_detector, face_landmarks):
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
    landmark_coords = find_landmarks(gray, hog_face_detector, face_landmarks)

    # Get the bounding box around the face
    x, y, w, h = cv2.boundingRect(np.array(landmark_coords))
    # Crop the image with the face
    face_img = gray[y:y + h, x:x + w]
    height, width = face_img.shape[:2]
    # Set the new width and calculate the new height to maintain aspect ratio
    new_width = 500
    new_height = int((height / width) * new_width)
    # Resize the image to the new dimensions
    resized_img = cv2.resize(face_img, (new_width, new_height))
    for i in range(len(landmark_coords)):
        landmark_coords[i] = ((landmark_coords[i][0] - x) * 500 / w, (landmark_coords[i][1] - y) * int(h * (500 / w)) / h)

    mouth = landmark_coords[FACIAL_LANDMARKS_IDXS["mouth"][0]:FACIAL_LANDMARKS_IDXS["mouth"][1]]
    eyes[LEFT] = landmark_coords[FACIAL_LANDMARKS_IDXS["left_eye"][0]:FACIAL_LANDMARKS_IDXS["left_eye"][1]]
    eyes[RIGHT] = landmark_coords[FACIAL_LANDMARKS_IDXS["right_eye"][0]:FACIAL_LANDMARKS_IDXS["right_eye"][1]]
    eyebrow[0] = landmark_coords[FACIAL_LANDMARKS_IDXS["left_eyebrow"][0]:FACIAL_LANDMARKS_IDXS["left_eyebrow"][1]]
    eyebrow[1] = landmark_coords[FACIAL_LANDMARKS_IDXS["right_eyebrow"][0]:FACIAL_LANDMARKS_IDXS["right_eyebrow"][1]]
    nose = landmark_coords[FACIAL_LANDMARKS_IDXS["nose"][0]:FACIAL_LANDMARKS_IDXS["nose"][1]]
    jaw = landmark_coords[FACIAL_LANDMARKS_IDXS["jaw"][0]:FACIAL_LANDMARKS_IDXS["jaw"][1]]
    return eyes, nose, mouth, eyebrow, jaw, resized_img
# ======================================================================================================================


# Extract feature vectors
def extract_features(eyes, nose, mouth, eyebrow, jaw):

    # Find the aspect ratio
    features = [euclidean(jaw[1], jaw[-1]) / euclidean(jaw[1], jaw[8])]
    print("face aspect ratio: ", features[0])

    lips_ellipse = cv2.fitEllipse(np.array(mouth, dtype=np.float32))
    features.append(euclidean(lips_ellipse[1][0], lips_ellipse[1][1]))  # extract the height (length of major axis)
    print("the major and minor axis aspect ratio: ", features[-1])

    # Extract the distance between the eyes
    eye1, eye2 = eyes[RIGHT], eyes[LEFT]
    distance = euclidean(eye1[3], eye2[0])
    # distance = np.sqrt((eye1[3][0] - eye2[0][0]) ** 2 + (eye1[3][1] - eye2[0][1]) ** 2)
    features.append(distance)
    print("eyes distance = ", distance)

    # Extract the distance from the nose to the chin
    mouth_x, mouth_y = mouth[9]
    jaw_x, jaw_y = jaw[8]
    distance = np.sqrt((mouth_x - jaw_x) ** 2 + (mouth_y - jaw_y) ** 2)
    print("distance mouth-chin = ", distance)
    features.append(distance)

    # extract the distance between the nose and mouth
    nose_x, nose_y = nose[6]
    mouth_x, mouth_y = mouth[3]
    distance = np.sqrt((nose_x - mouth_x) ** 2 + (nose_y - mouth_y) ** 2)
    print("distance nose-mouth = ", distance)
    features.append(distance)

    # extract length of nose
    print("nose length = ", euclidean(nose[6], nose[0]))
    features.append(euclidean(nose[6], nose[0]))

    # extract width of nose
    print("nose width = ", euclidean(nose[4], nose[8]))
    features.append(euclidean(nose[4], nose[8]))

    # extract the shape of the eyebrows
    print("left eyebrow = ", end=" ")
    for index in range(len(eyebrow[LEFT])-1):
        # features.append(euclidean(eyebrow[LEFT][index], eyebrow[LEFT][index+1]))
        print(euclidean(eyebrow[LEFT][index], eyebrow[LEFT][index+1]), end=", ")

    print("right eyebrow = ", end=" ")
    for index in range(len(eyebrow[RIGHT]) - 1):
        # features.append(euclidean(eyebrow[RIGHT][index], eyebrow[RIGHT][index + 1]))
        print(euclidean(eyebrow[RIGHT][index], eyebrow[RIGHT][index+1]), end=", ")

    # Extract the distance between the eyebrows
    features.append(euclidean(eyebrow[RIGHT][-1], eyebrow[LEFT][0]))
    print("eyebrows distance = ", euclidean(eyebrow[RIGHT][-1], eyebrow[LEFT][0]))

    # extract the thickness of the lips
    features.append(euclidean(mouth[2], mouth[13]))  # the thickness of the thick part the upper lip
    print("thick part the upper lip = ", euclidean(mouth[2], mouth[13]))
    features.append(euclidean(mouth[3], mouth[14]))  # the thickness of the thin part the upper lip
    print("thin part the upper lip = ", euclidean(mouth[3], mouth[14]))
    features.append(euclidean(mouth[8], mouth[18]))  # the thickness of the thick part the downer lip
    print("thick part the downer lip = ", euclidean(mouth[8], mouth[18]))
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
    """"
    distance = 0
    for feature in normalize_features:
        print(feature)
        distance += np.sqrt(np.sum([(feature[0] - feature[1])**2]))
    """
    # distance = np.sqrt(np.sum([(f1[i] - f2[i]) ** 2 for i in range(len(f1))]))
    distance = np.sum([euclidean(f1[i], f2[i]) for i in range(len(f1))]) #לזכור לשנות את הלן לקבוע נאם אוף פיצ'רס
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


def findEyesColor(image, eye, features):
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

    features.append(a_mean)
    features.append(b_mean)

    print("A: orginal = {}, after = {}".format(np.mean(a), a_mean))
    print("B: orginal = {}, after = {}".format(np.mean(b), b_mean))


# ======================================================================================================================


# move on all the vectors of features
# take one feature from all the vectors and send it to normalize
def normalizeRes():
    for feature in range(len(allvectors)):  # loop on specific feature
        min_val, max_val = min(feature), max(feature)
        normalize_features[feature] = [(i - min_val) / (max_val - min_val) * (T_MAX - T_MIN) + T_MIN for i in feature]
# ======================================================================================================================


# Main function
def recognize_face(image_name, hog_face_detector, face_landmarks):

    image = cv2.imread(image_name)
    # Pre-process the images
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect facial features
    eyes, nose, mouth, eyebrow, jaw, gray = detect_features(gray, hog_face_detector, face_landmarks)

    features = extract_features(eyes, nose, mouth, eyebrow, jaw)

    # extract the skin color
    # find_skin_color(image, nose, eyebrow, jaw)

    # extract the eyes color
    findEyesColor(image, eyes[LEFT], features)

    for row, feature in zip(allvectors, features):  # every row is a specific feature
        row.append(feature)

    # normalizeRes()

    return features

# ======================================================================================================================


def recognize(hog_face_detector, dlib_facelandmark, user1, user2):

    f1 = []
    print("******************** first *********************************")
    with open('faces/{}.txt'.format(user1), 'r') as f:
        image_paths = ["faces/{}/".format(user1) + line.strip() + ".jpg" for line in f]
    for image_path in image_paths:
        print("-------------------------------------------------------")
        print(image_path)
        f1.append(recognize_face(image_path, hog_face_detector, dlib_facelandmark))

    print("******************** second *********************************")
    f2 = []
    with open('faces/{}.txt'.format(user2), 'r') as f:
        image_paths1 = ["faces/{}/".format(user2) + line.strip() + ".jpg" for line in f]
    for image_path in image_paths1:
        print("-------------------------------------------------------")
        print(image_path)
        f2.append(recognize_face(image_path, hog_face_detector, dlib_facelandmark))

    print("******************** comparing: *********************")
    print("first:")
    for i in range(len(f1) - 1):
        print(compare_features(f1[i], f1[i + 1]))
    print("second:")
    for i in range(len(f2) - 1):
        print(compare_features(f2[i], f2[i + 1]))
    print("first vs second:")
    for i in range(min(len(f1), len(f2))):
        print(compare_features(f1[i], f2[i]))

    # k-means:
    # Convert data to a NumPy array for easier processing
    feature = 1
    data = np.array(allvectors[feature])
    # Initialize centroids randomly
    centroids = np.random.choice(data, size=2)

    # Assign data points to nearest centroid
    labels = np.argmin(np.abs(data[:, np.newaxis] - centroids), axis=1)

    # Split data into two clusters based on labels
    cluster1 = data[labels == 0]
    cluster2 = data[labels == 1]

    # Plot the two clusters and centroids
    plt.scatter(cluster1, np.zeros_like(cluster1), c='b', label='Cluster 1')
    plt.scatter(cluster2, np.zeros_like(cluster2), c='r', label='Cluster 2')

    # Set x-tick labels to image names
    plt.xticks(data, [os.path.basename(path)[:-4] for path in image_paths + image_paths1], rotation=90)

    # mark the first half of the feature list (the first user):
    values1 = allvectors[feature][:len(image_paths)]
    for val in values1:
        plt.axvline(val, color='orange', linestyle='--')

    plt.title("feature " + str(feature) + ": " + user1 + " - " + user2)
    plt.legend()
    plt.show()

    # Clear all sublists in allvectors
    for sublist in allvectors:
        sublist.clear()

# ======================================================================================================================


if __name__ == "__main__":

    hog_face_detector = dlib.get_frontal_face_detector()
    dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    recognize(hog_face_detector, dlib_facelandmark, "l", "b")
    recognize(hog_face_detector, dlib_facelandmark, "l", "e")
    recognize(hog_face_detector, dlib_facelandmark, "l", "y")
    recognize(hog_face_detector, dlib_facelandmark, "l", "t")
    recognize(hog_face_detector, dlib_facelandmark, "b", "e")
    recognize(hog_face_detector, dlib_facelandmark, "b", "t")
    recognize(hog_face_detector, dlib_facelandmark, "b", "y")
    recognize(hog_face_detector, dlib_facelandmark, "e", "t")
    recognize(hog_face_detector, dlib_facelandmark, "e", "y")
    recognize(hog_face_detector, dlib_facelandmark, "t", "y")

