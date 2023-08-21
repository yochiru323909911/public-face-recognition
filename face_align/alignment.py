import cv2, dlib, argparse
from face_align.utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix


def refind_landmarks(shape, M):
    """
    Re-calculate the landmarks' positions on a rotated image using the given rotation matrix.

    Args:
        shape (dlib.full_object_detection): A dlib shape object representing facial landmarks on the original image.
        M (numpy.ndarray): The 2x3 rotation matrix used to align the image.

    Returns:
        list: A list of tuples containing the new (x, y) coordinates of the re-calculated landmarks.

    This function takes a dlib shape object representing facial landmarks and a rotation matrix used to align
    the image to a rotated position. It calculates the new positions of the landmarks on the rotated image
    by applying the same rotation transformation to each landmark point.

    The landmarks' coordinates are transformed using the given rotation matrix M, ensuring that the landmarks'
    positions on the rotated image correspond to their positions on the original image.
    """
    new_landmarks = []
    for i in range(shape.num_parts):
        x, y = shape.part(i).x, shape.part(i).y
        new_x = int(M[0, 0] * x + M[0, 1] * y + M[0, 2])
        new_y = int(M[1, 0] * x + M[1, 1] * y + M[1, 2])
        new_landmarks.append((new_x, new_y))
    return new_landmarks
# ======================================================================================================================


def align(detector, predictor, input_image, scale):
    """
   Align a face in an input image using facial landmarks.

   Args:
       detector (dlib.fhog_object_detector): A face detector object from dlib.
       predictor (dlib.shape_predictor): A facial landmark predictor object from dlib.
       input_image (str): Path to the input image.
       scale (int): Scale factor for resizing the image.

   Returns:
       rotated_gray: The gray rotated image.
       rotated_color: The color rotated image.
       new_landmarks: The landmarks of the rotated images.

       Or None if a face didn't detect in the image.


   This function aligns a face in an input image using facial landmarks.
   It takes a dlib face detector, a facial landmark predictor, the path to the input image,
   and a scale factor for image resizing.

   The function performs the following steps:
   1. Load the input image and convert it to grayscale.
   2. Resize the grayscale and color images using the specified scale factor.
   3. Detect faces using the provided face detector.
   4. For each detected face, extract left and right eye centers using the landmark predictor.
   5. Calculate a rotation matrix using the extracted eye centers.
   6. Apply the rotation matrix to both the grayscale and color images.
   7. Return the aligned grayscale and color images.

   If no face is detected, a message is printed, and None is returned for both aligned images.

   Note: The functions 'extract_left_eye_center', 'extract_right_eye_center', and 'get_rotation_matrix'
   are assumed to be defined elsewhere in the codebase.
   """

    color = cv2.imread(input_image)
    img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)

    height, width = img.shape[:2]
    s_height, s_width = height // scale, width // scale
    img = cv2.resize(img, (s_width, s_height))
    color = cv2.resize(color, (s_width, s_height))

    dets = detector(img, 1)

    for i, det in enumerate(dets):
        shape = predictor(img, det)
        left_eye = extract_left_eye_center(shape)
        right_eye = extract_right_eye_center(shape)

        M = get_rotation_matrix(left_eye, right_eye)
        rotated_gray = cv2.warpAffine(img, M, (s_width, s_height), flags=cv2.INTER_CUBIC)
        rotated_color = cv2.warpAffine(color, M, (s_width, s_height), flags=cv2.INTER_CUBIC)

        # Re-calculate the landmarks with the rotation matrix
        new_landmarks = refind_landmarks(shape, M)

        return rotated_gray, rotated_color, new_landmarks

    print("No detection!☹")
    return None, None, None
