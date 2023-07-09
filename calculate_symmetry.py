import cv2
import skimage
import matplotlib.pyplot as plt
import dlib
import numpy as np
import math
from matplotlib import pyplot as plt

def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def returnMass(image_path):
    global checker
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('C:/Users/lysyi/Desktop/symmetry_face/shape_pred_68.dat')

    faces = detector(gray)
    faces_detected = "אנשים נמצאו: " + format(len(faces))
    print(faces_detected)
    mass = []
    mass1 = []

    x1 = faces[0].left()
    y1 = faces[0].top()
    x2 = faces[0].right()
    y2 = faces[0].bottom()

    # cv2.circle(image, (x1, y1), 3, (255, 0, 0), -1)
    # cv2.circle(image, (x2, y2), 3, (255, 0, 0), -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    landmarks = predictor(gray, faces[0])

    k = [7, 6, 5, 4, 3, 2, 1, 0, 17, 18, 19, 20, 21, 39, 38, 37, 36, 41, 40, 31, 32, 48, 49, 59, 58, 67]
    m = [9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 23, 22, 42, 43, 44, 45, 46, 47, 35, 34, 53, 54, 55, 56, 65]

    for n in k:
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        # cv2.putText(image, str(n), (x+5,y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        mass.append((x, y))

    for n in m:
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 3, (255, 0, 100), -1)
        # cv2.putText(image, str(n), (x+5,y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        mass1.append((x, y))

    for n in [28, 29, 30]:
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
    # cv2.putText(image, str(n), (x+5,y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    mid_x = round((landmarks.part(28).x + landmarks.part(29).x + landmarks.part(30).x) / 3)

    # cv2.putText(image, "Press ESC to close image", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    # viewImage(image,"faces_detected")
    mass2 = []
    for i in range(0, len(mass)):
        mass2.append(2 * mid_x - int(mass[i][0]))

    sum = 0
    for i in range(0, len(mass)):
        sum = sum + ((mass1[i][0] - mass2[i]) ** 2 + (mass[i][1] - mass1[i][1]) ** 2) ** .5

    print(image_path[:-4])
    cv2.imwrite(image_path[:-4] + 'worked.jpg', image)
    # viewImage(image,"faces_detected")
    return (sum)