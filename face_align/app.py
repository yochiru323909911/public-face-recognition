import cv2, dlib, argparse
from face_align.utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix


def align(input_image, scale):
    """parser = argparse.ArgumentParser(description='Align faces in image')
    parser.add_argument('input', type=str, help='')
    parser.add_argument('output', type=str, help='')
    parser.add_argument('--scale', metavar='S', type=int, default=4, help='an integer for the accumulator')"""

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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

        return rotated_gray, rotated_color

    return None, None
