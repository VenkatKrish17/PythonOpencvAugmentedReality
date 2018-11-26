
from objloader_simple import *
import cv2
import numpy as np
import math
def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """

    # h_len = len(hex_color)
    # return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

obj=OBJ('../models/effiel.obj', swapyz=False)
camera_parameters = np.array([[200, 0, 0], [0, 200, 200], [1, 0, 1]])
homography = np.array([[1,0,0],[0,1,0],[0,0,1]])
faces=obj.faces
vertices = obj.vertices
print(len(faces))
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 2)
w=400
h=400
model = []
color = False
scale=0.2
translate=100
# print(cv2.findHomography(cv2.imread("aruco_marks.png"),cv2.imread("aruco.png")))


# points = points.reshape(-1, 1, 3)
# dst=cv2.getAffineTransform(points,projection)
# print(points.shape)
# # print(points)
# dst = cv2.getPerspectiveTransform(points.reshape(-1, 1, 3),projection )
# # print(dst)
# print(dst)
def rotation():
    global homography
    # homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # print(col_3)
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    translation = [translation[0] + 200, translation[1]+200,translation[2]]
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    # print(rot_1,rot_2,rot_3)
    projection = np.stack((rot_1,rot_2, rot_3, translation)).T
    projection = np.dot(camera_parameters, projection)
    print(projection)
    return projection
rotation()


while True:
    # read the current frame
    ret, frame = cap.read()
    for face in faces:
        # print(face)
        face_vertices = face[0]
        # print(face_vertices)
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        # print(points)
        points = np.array([[p[0]*0.005+200,p[1]*0.005+200] for p in points])
        from numpy import sin, cos
        theta = np.radians(90)
        rotated_points=[]
        for point in points:
            p_x=point[0]
            p_y=point[1]
            p_rotated_x = p_x * sin(theta) - p_y * sin(theta)
            p_rotated_y = p_x * sin(theta) + p_y * cos(theta)
            rotated_points.append([p_rotated_x,p_rotated_y])
        rotated_points = np.array([[p[0],p[1]] for p in rotated_points])

        pts = np.int32(rotated_points)
        points = np.array([[p[0]+200,p[1]+200] for p in pts])
        print(points)
        if color is False:
            cv2.fillConvexPoly(frame, np.int32(points), (10, 10, 10))
        else:
            # color = hex_to_rgb(face[-1])
            # color = color[::-1]  # reverse
            color=tuple(face[-1])
            cv2.fillConvexPoly(frame, pts, (face[-1][0]/16,face[-1][1]/16,face[-1][2]/16))
    cv2.imshow("loaded",frame)
    cv2.waitKey(0)
