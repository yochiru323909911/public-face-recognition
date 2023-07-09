import cv2
import skimage
import matplotlib.pyplot as plt
import dlib
import numpy as np
import math
from matplotlib import pyplot as plt
from face_align.app import align


def find_landmarks(img, hog_face_detector, dlib_facelandmark):
    faces = hog_face_detector(img)
    if len(faces) == 0:
        return None
    for face in faces:
        face_landmarks = dlib_facelandmark(img, face)
    landmarks = []
    for n in range(0, 68):
        x = face_landmarks.part(n).x
        y = face_landmarks.part(n).y
        landmarks.append((x, y))
    return landmarks


def resize_face(colorfull, gray, hog_face_detector, dlib_facelandmark):

    landmark_coords = find_landmarks(gray, hog_face_detector, dlib_facelandmark)
    if landmark_coords is None:
        return None, None, None
    x, y, w, h = cv2.boundingRect(np.array(landmark_coords))
    face_gray = gray[y:y + h, x:x + w]
    face_colorfull = colorfull[y:y + h, x:x + w]
    height, width = face_gray.shape[:2]
    new_height = int((height / width) * 500)
    resized_gray = cv2.resize(face_gray, (500, new_height))
    resized_colorfull = cv2.resize(face_colorfull, (500, new_height))
    landmark_coords = find_landmarks(resized_gray, hog_face_detector, dlib_facelandmark)
    return resized_gray, resized_colorfull, landmark_coords


def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def returnMass(image_path):
    rotated_gray, rotated_color = align(image_path, 4)
    if rotated_gray is None:
        return 0

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    final_gray, final_color, landmarks = resize_face(rotated_color, rotated_gray, detector, predictor)

    if landmarks is None:
        return 0

    mass = []
    mass1 = []

    k = [7, 6, 5, 4, 3, 2, 1, 0, 17, 18, 19, 20, 21, 39, 38, 37, 36, 41, 40, 31, 32, 48, 49, 59, 58, 67]
    m = [9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 23, 22, 42, 43, 44, 45, 46, 47, 35, 34, 53, 54, 55, 56, 65]

    for n in k:
        x = landmarks[n][0]
        y = landmarks[n][1]
        cv2.circle(final_color, (x, y), 3, (0, 255, 0), -1)
        # cv2.putText(image, str(n), (x+5,y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        mass.append((x, y))

    for n in m:
        x = landmarks[n][0]
        y = landmarks[n][1]
        cv2.circle(final_color, (x, y), 3, (255, 0, 100), -1)
        # cv2.putText(image, str(n), (x+5,y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        mass1.append((x, y))

    for n in [28, 29, 30]:
        x = landmarks[n][0]
        y = landmarks[n][1]
        cv2.circle(final_color, (x, y), 3, (0, 0, 255), -1)
    # cv2.putText(image, str(n), (x+5,y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    mid_x = round((landmarks[28][0] + landmarks[29][0] + landmarks[30][0]) / 3)

    # cv2.putText(image, "Press ESC to close image", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    # viewImage(image,"faces_detected")
    mass2 = []
    for i in range(0, len(mass)):
        mass2.append(2 * mid_x - int(mass[i][0]))

    sum = 0
    for i in range(0, len(mass)):
        sum = sum + ((mass1[i][0] - mass2[i]) ** 2 + (mass[i][1] - mass1[i][1]) ** 2) ** .5

    print(image_path[:-4], end=": ")
    cv2.imwrite(image_path[:-4] + 'worked.jpg', final_color)
    # viewImage(image,"faces_detected")
    return (sum)

def permotations_images(user):
    vector = []
    print("******************** {} *********************************".format(user))
    with open('symmetry_faces/{}.txt'.format(user), 'r') as f:
        image_paths = ["symmetry_faces/{}/".format(user) + line.strip() + ".jpg" for line in f]
    for image_path in image_paths:
        res = returnMass(image_path)
        if res != 0:
            vector.append(res)
            print(res)
    return vector

i = permotations_images("o")
if len(i) != 0:
    print("average:", sum(i) / len(i), ", min:", min(i), ", max:", max(i))
i = permotations_images("b")
if len(i) != 0:
    print("average:", sum(i) / len(i), ", min:", min(i), ", max:", max(i))
i = permotations_images("e")
if len(i) != 0:
    print("average:", sum(i) / len(i), ", min:", min(i), ", max:", max(i))
i = permotations_images("l")
if len(i) != 0:
    print("average:", sum(i) / len(i), ", min:", min(i), ", max:", max(i))
i = permotations_images("t")
if len(i) != 0:
    print("average:", sum(i) / len(i), ", min:", min(i), ", max:", max(i))
i = permotations_images("y")
if len(i) != 0:
    print("average:", sum(i) / len(i), ", min:", min(i), ", max:", max(i))


