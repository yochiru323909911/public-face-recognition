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
    left_landmark_indices = [39, 36, 0, 21, 31, 32, 50, 48, 58]
    right_landmark_indices = [42, 45, 16, 22, 35, 34, 52, 54, 56]

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
