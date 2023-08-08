import math

def calculate_symmetry_score(landmarks):
    """
    Calculate the symmetry score of a face based on its landmarks.

    The symmetry score measures the level of symmetry between two halves of the face.
    The landmarks should be ordered in a consistent manner.

    Args:
        landmarks (list): List of tuples representing facial landmark coordinates.

    Returns:
        symmetry_score (float): The calculated symmetry score of the face.
    """
    # Calculate the distance between the outermost landmarks to estimate the face size
    x_coords, y_coords = zip(*landmarks)
    face_size = math.hypot(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords))

    # Define indices for the landmarks on the left and right halves of the face
    left_landmark_indices = [7, 6, 5, 4, 3, 2, 1, 0, 17, 18, 19, 20, 21, 39, 38, 37, 36, 41, 40, 31, 32, 48, 49, 59, 58, 67]
    right_landmark_indices = [9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 23, 22, 42, 43, 44, 45, 46, 47, 35, 34, 53, 54, 55, 56, 65]

    # Extract landmarks for the left and right halves
    left_landmarks = [landmarks[i] for i in left_landmark_indices]
    right_landmarks = [landmarks[i] for i in right_landmark_indices]

    mid_x = round(sum(landmarks[i][0] for i in range(28, 31)) / 3)

    mass2 = [(2 * mid_x - int(x), y) for x, y in left_landmarks]

    symmetry_score = 0
    for i in range(len(left_landmarks)):
        dx = left_landmarks[i][0] - mass2[i][0]
        dy = left_landmarks[i][1] - right_landmarks[i][1]
        normalized_symmetry = math.hypot(dx, dy) / face_size
        symmetry_score += normalized_symmetry

    return symmetry_score
