
# Useful links
# http://www.pygame.org/wiki/OBJFileLoader
# https://rdmilligan.wordpress.com/2015/10/15/augmented-reality-using-opencv-opengl-and-blender/
# https://clara.io/library

# TODO -> Implement command line arguments (scale, model and object to be projected)
#      -> Refactor and organize code (proper funcition definition and separation, classes, error handling...)

import argparse

import cv2
import numpy as np
import math
import os
from objloader import *
import sys
from visualizer import display
from threading import Thread
sys.path.append('model/model1')

# Minimum number of matches that have to be found
# to consider the recognition valid
MIN_MATCHES = 250

def clahe(img, clip_limit=2.0, grid_size=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)
def main():
    """
    This functions loads the target surface image,
    """
    homography = None
    # matrix of camera parameters (made up but works quite well for me)
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    # create ORB keypoint detector
    orb = cv2.ORB_create()
    # create BFMatcher object based on hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # load the reference surface that will be searched in the video stream
    dir_name = "/".join(os.getcwd().split("\\")[:-1])
    # model = cv2.imread(os.path.join(dir_name, 'reference/aruco_marks.png'), 0)
    # Compute model keypoints and its descriptors

    # Load 3D model from OBJ file
    # print(dir_name)
    # sys.path.append(dir_name+'/model/model2')
    # print(sys.path)
    # print(os.path.join(dir_name, 'model/model2/OBJ2.obj'))
    # sys.path.append('/model/model2')
    obj = OBJ('model/model2/OBJ2.obj',with_gl=False, swapyz=True)
    # init video capture
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FPS, 2)
    models = []
    while True:
        # read the current frame
        ret, frame = cap.read()
        # print(ret)    
        # print(frame)
        if not ret:
            print ("Unable to capture video")
            return
        # find and draw the keypoints of the frame
        kp_frame, des_frame = orb.detectAndCompute(frame, None)

        if(len(models)==0):
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # # gray = cv2.bilateralFilter(gray, 11, 17, 17)
            # edged = cv2.Canny(gray, 100, 255)
            # (__,cnts, _) = cv2.findContours(edged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.imshow("Canny",edged)
            # c = max(cnts, key = cv2.contourArea)
            # c=np.array(c)
            # c1=cv2.boundingRect(c)
            # print(c1)
            # np.expand_dims(contours,-1)
            # cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
            #
            #
            # c=cv2.selectROI(frame)
            # x,y,w,h = cv2.boundingRect(c)
            # x,y,w,h = c
            # print(x,y,w,h)
            # # print(screenCnt)
            #
            # frame
            # match frame descriptors with model descriptors
            # model = frame[y:y+h,x:x+w]
            model1=cv2.imread(os.path.join(dir_name,"references/ref1/1.jpeg"),0)
            model2=cv2.imread(os.path.join(dir_name,"references/ref1/2.jpeg"),0)
            model3=cv2.imread(os.path.join(dir_name,"references/ref1/3.jpeg"),0)
            model1=cv2.resize(model1,(400,400))
            model2=cv2.resize(model2,(400,400))
            model3=cv2.resize(model3,(400,400))
            #cv2.imshow("model",model)
        #print(model)
            # cv2.imshow("model",model)
        models=[model1,model2,model3]
        kp_model1, des_model1 = orb.detectAndCompute(model1, None)
        kp_model2, des_model2 = orb.detectAndCompute(model2, None)
        kp_model3, des_model3 = orb.detectAndCompute(model3, None)
        # cv2.imshow("out",des_model)
        kp_models=[kp_model1,kp_model2,kp_model3]
        matches1 = bf.match(des_model1, des_frame)
        matches2 = bf.match(des_model2, des_frame)
        matches3 = bf.match(des_model3, des_frame)
        all_matches=[matches1,matches2,matches3]
        all_matches_index=[len(matches1),len(matches2),len(matches3)]
        print(all_matches_index)
        matches=all_matches[all_matches_index.index(sorted(all_matches_index)[2])]
        kp_model=kp_models[all_matches_index.index(sorted(all_matches_index)[2])]
        model=models[all_matches_index.index(sorted(all_matches_index)[2])]
        matches = sorted(matches, key=lambda x: x.distance)
        print(len(matches))

        if len(matches) > MIN_MATCHES:
            print("found")
            # differenciate between source points and destination points
            src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # compute Homography
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if args.rectangle:
                # Draw a rectangle that marks the found model in the frame
                h, w = model.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                # project corners into frame

                dst = cv2.perspectiveTransform(pts,np.array([[1.5,0,0],[0,1.5,0],[0,0,1.5]]))
                # connect them with lines
                frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            # if a valid homography matrix was found render cube on model plane
            if homography is not None:
                print("homo not none")
                try:
                    # obtain 3D projection matrix from homography matrix and camera parameters
                    # print("proj mat")
                    # print(len(projection_matrix(camera_parameters, homography)))
                    # print("estimating projection")
                    projection = projection_matrix(camera_parameters, homography)
                    # print(projection)
                    # print(len(projection_matrix(camera_parameters, homography)))
                    # project cube or model
                    # print(type(projection))
                    print(len(render(frame, obj, projection, model, False)))
                    frame = render(frame, obj, projection, model, False)
                    #frame = render(frame, model, projection)
                except Exception as e:
                    print(e)
                    pass
            # draw first 10 matches.
            if args.matches or True:
                frame = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:10], 0, flags=2)
            # show result
            cv2.imshow('frame', frame)
            print(len(os.listdir("../model"))+1)
            for mod_len in range(1,4):
                path='../model/model'+str(mod_len)
                print(mod_len)
                sys.path.append(path)
                print(path)
                #display("../model/model"+str(mod_len)+"/OBJ.obj")
                os.system("python visualizer.py "+'../model/model'+str(mod_len)+"/OBJ"+str(mod_len)+".obj")
                # sys.path.append('../model/model2')
                # display("../model/model2/OBJ.obj")
            cv2.waitKey(1)
            # break;


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            print( "Not enough matches found - %d/%d" % (len(matches), MIN_MATCHES))

    cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
    return 0

def render(img, obj, projection, model, color=True):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 8
    print("h,w")
    print(model.shape)
    w,h = model.shape

    # print("faces")
    for face in obj.faces:
        face_vertices = face[0]
        # print(face)
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        # print(points)
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        # print(dst)
        #print(dst)
        imgpts = np.int32(dst)
        #print(imgpts)
        color=False
        # print(face[-1])

        if color is False:
            cv2.fillConvexPoly(img, imgpts, (10, 10, 10))
        else:
            # color = hex_to_rgb(face[-1])
            # color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, (face[-1][0],face[-1][1],face[-1][2]))

    return img

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters),np.array([[1,0,0],[0,1,0],[0,0,1]]))
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    # print(rot_1,rot_2,rot_3)
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))


# Command line argument parsing
# NOT ALL OF THEM ARE SUPPORTED YET
parser = argparse.ArgumentParser(description='Augmented reality application')

parser.add_argument('-r','--rectangle', help = 'draw rectangle delimiting target surface on frame', action = 'store_true')
parser.add_argument('-mk','--model_keypoints', help = 'draw model keypoints', action = 'store_true')
parser.add_argument('-fk','--frame_keypoints', help = 'draw frame keypoints', action = 'store_true')
parser.add_argument('-ma','--matches', help = 'draw matches between keypoints', action = 'store_true')
# TODO jgallostraa -> add support for model specification
#parser.add_argument('-mo','--model', help = 'Specify model to be projected', action = 'store_true')

args = parser.parse_args()

if __name__ == '__main__':
    main()
