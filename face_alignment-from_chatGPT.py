# one algorithm:
import numpy as np
import open3d as o3d
import dlib

# Load the face detector and landmark predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the two face images
image1 = o3d.io.read_image('face1.jpg')
image2 = o3d.io.read_image('face2.jpg')

# Detect the face landmarks in both images
landmarks1 = detector(image1, 1)[0]
landmarks2 = detector(image2, 1)[0]
landmarks1 = predictor(image1, landmarks1)
landmarks2 = predictor(image2, landmarks2)

# Convert the 2D landmarks to 3D points
def convert_landmarks_to_points(landmarks):
    points = []
    for i in range(68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        points.append([x, y, 0])
    return np.asarray(points)

points1 = convert_landmarks_to_points(landmarks1)
points2 = convert_landmarks_to_points(landmarks2)

# Estimate the initial transformation using an affine transform
T = o3d.registration.TransformationEstimationAffine()
trans_init = T.compute_transformation(o3d.geometry.PointCloud(points1), o3d.geometry.PointCloud(points2))

# Run the ICP algorithm to refine the transformation
icp = o3d.registration.registration_icp(o3d.geometry.PointCloud(points1), o3d.geometry.PointCloud(points2), 1.0, trans_init,
                                          o3d.registration.TransformationEstimationPointToPoint())
# Apply the transformation to the second image to align it with the first
aligned_image2 = image2.transform(icp.transformation)

# Visualize the aligned images
o3d.visualization.draw_geometries([image1, aligned_image2])











# more good:
import numpy as np
from skimage import transform as trans
import alignmention

# Load the face alignment model
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

# Define the source and target images
src_img = np.array(...)  # 2D numpy array containing the first face image
tgt_img = np.array(...)  # 2D numpy array containing the second face image

# Detect the facial landmarks for each image
src_landmarks = fa.get_landmarks(src_img)[0]
tgt_landmarks = fa.get_landmarks(tgt_img)[0]

# Compute the similarity transformation that maps the source landmarks to the target landmarks
tform = trans.SimilarityTransform()
tform.estimate(tgt_landmarks, src_landmarks)

# Apply the similarity transformation to the source image to align it with the target image
aligned_src = trans.warp(src_img, tform.inverse, output_shape=tgt_img.shape)
