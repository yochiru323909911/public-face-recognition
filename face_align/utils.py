import numpy as np
import cv2

LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
# ======================================================================================================================


def rect_to_tuple(rect):
    """
    Convert a dlib rectangle object to a tuple of left, top, right, and bottom coordinates.

    Args:
        rect (dlib.rectangle): A dlib rectangle object representing a bounding box.

    Returns:
        tuple: A tuple containing the left, top, right, and bottom coordinates of the rectangle.
    """
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return left, top, right, bottom
# ======================================================================================================================


def extract_eye(shape, eye_indices):
    """
    Extract eye landmarks from a shape object using specified eye indices.

    Args:
        shape (dlib.full_object_detection): A dlib shape object representing facial landmarks.
        eye_indices (list): A list of indices corresponding to the desired eye landmarks.

    Returns:
        list: A list of dlib points representing the extracted eye landmarks.
    """
    points = map(lambda i: shape.part(i), eye_indices)
    return list(points)
# ======================================================================================================================


def extract_eye_center(shape, eye_indices):
    """
    Extract the center point of an eye from a shape object using specified eye indices.

    Args:
        shape (dlib.full_object_detection): A dlib shape object representing facial landmarks.
        eye_indices (list): A list of indices corresponding to the desired eye landmarks.

    Returns:
        tuple: A tuple containing the x and y coordinates of the extracted eye center.
    """
    points = extract_eye(shape, eye_indices)
    xs = map(lambda p: p.x, points)
    ys = map(lambda p: p.y, points)
    return sum(xs) // 6, sum(ys) // 6
# ======================================================================================================================


def extract_left_eye_center(shape):
    """
    Extract the center point of the left eye from a shape object.

    Args:
        shape (dlib.full_object_detection): A dlib shape object representing facial landmarks.

    Returns:
        tuple: A tuple containing the x and y coordinates of the extracted left eye center.
    """
    return extract_eye_center(shape, LEFT_EYE_INDICES)
# ======================================================================================================================


def extract_right_eye_center(shape):
    """
    Extract the center point of the right eye from a shape object.

    Args:
        shape (dlib.full_object_detection): A dlib shape object representing facial landmarks.

    Returns:
        tuple: A tuple containing the x and y coordinates of the extracted right eye center.
    """
    return extract_eye_center(shape, RIGHT_EYE_INDICES)
# ======================================================================================================================


def angle_between_2_points(p1, p2):
    """
    Calculate the angle between two points.

    Args:
        p1 (tuple): A tuple containing the x and y coordinates of the first point.
        p2 (tuple): A tuple containing the x and y coordinates of the second point.

    Returns:
        float: The angle in degrees between the two points.
    """
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))
# ======================================================================================================================


def get_rotation_matrix(p1, p2):
    """
    Calculate the rotation matrix for aligning points p1 and p2.

    Args:
        p1 (tuple): A tuple containing the x and y coordinates of the first point.
        p2 (tuple): A tuple containing the x and y coordinates of the second point.

    Returns:
        numpy.ndarray: The 2x3 rotation matrix.
    """
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M
# ======================================================================================================================


def crop_image(image, det):
    """
    Crop the specified region from the input image based on the given rectangle.

    Args:
        image (numpy.ndarray): The input image.
        det (dlib.rectangle): A dlib rectangle representing the region to be cropped.

    Returns:
        numpy.ndarray: The cropped region of the input image.
    """
    left, top, right, bottom = rect_to_tuple(det)
    return image[top:bottom, left:right]
