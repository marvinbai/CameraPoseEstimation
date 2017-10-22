import numpy as np
import cv2  # To use SIFT descriptor, opencv-contrib-python is needed instead of opencv-python.
import glob
from math import atan2, asin

# Read Images.
patPic = cv2.imread('../data/pattern.png', cv2.IMREAD_GRAYSCALE)
img_names = glob.glob('../data/IMG_*.JPG')
pics = []   # List of np array.
for img_name in img_names:
    pics.append(cv2.imread(img_name, 0))

# Detect key points in original pattern using SIFT descriptor.    
sift = cv2.xfeatures2d.SIFT_create()
kp1 = sift.detect(patPic, None)
cv2.imwrite('keyPoints.png', cv2.drawKeypoints(patPic, kp1, patPic))    # Visualize key points on original pattern.

# Key points matching.
kp1, des1 = sift.detectAndCompute(patPic, None)
bf = cv2.BFMatcher()    # BFMatcher with default params. Since the image is not large, brutal force matcher is implmented.
                        # For large images, use FLANN based matcher instead.

# Store info for camera calibration.
points1 = [];   # Object points in global coordinates. List of np arrays.
points2 = [];   # Image points in image coordinates. List of np arrays.
                        
for cnt in range(len(pics)):    # Iterate over all photos taken by iphone 6.
    pic = pics[cnt]
    kp2, des2 = sift.detectAndCompute(pic, None)    # Key points and descriptors from SIFT.
    matches = bf.knnMatch(des1, des2, k = 2)    # Match points from SIFT of two figures.
    good = []   # Matches which satisfy the threshold. List of DMatch.
    pt1 = []    # Coordinates of Keypoints of good match in original pattern.
    pt2 = []    # Coordinates of Keypoints of good match in taken photo.

    # Use a threshold to filter the matches.
    for i in range(len(matches)):
        m, n = matches[i]
        if m.distance < 0.35 * n.distance:
            good.append([m])
            pt1.append([kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]])
            pt2.append([kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]])
    print('Number of good match for picture[', cnt, '] is', len(good))
    cv2.imwrite('matching_' + str(cnt) + '.png', cv2.drawMatchesKnn(patPic, kp1, pic, kp2, good, None, flags = 2))
    
    # Use the object points in global coordinates and image points for camera calibration.
    pts1 = np.zeros((len(good), 3), dtype = np.float32)
    pts2 = np.zeros((len(good), 2), dtype = np.float32)
    for i in range(len(good)):
        pts1[i, :] = pt1[i][0]*8.8/patPic.shape[0], pt1[i][1]*8.8/patPic.shape[1], 0  # Object points in [cm] in global coordinates.
        pts2[i, :] = pt2[i][0], pt2[i][1]    
    points1.append(pts1)
    points2.append(pts2)

# Camera calibration.
_, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(points1, points2, pics[0].shape, None, None)

pics = []   # Reset to load color pictures for demonstration.
for img_name in img_names:
    pics.append(cv2.imread(img_name, cv2.IMREAD_COLOR))

# Draw the axis in images for demonstration.
for cnt in range(len(pics)):
    print()
    print('Calculating for pic[' + str(cnt) + ']...')
    _, rvec, tvec, inliers = cv2.solvePnPRansac(points1[cnt], points2[cnt], mtx, dist) # Translation and rotation vector. 
    
    # Project the global coordinates to the image points.
    source, _ = cv2.projectPoints(np.float32([0, 0, 0]).reshape(1,-1), rvec, tvec, mtx, dist)
    axis_x, _ = cv2.projectPoints(np.float32([0, 10, 0]).reshape(1,-1), rvec, tvec, mtx, dist)
    axis_y, _ = cv2.projectPoints(np.float32([10, 0, 0]).reshape(1,-1), rvec, tvec, mtx, dist)
    axis_z, _ = cv2.projectPoints(np.float32([0, 0, -10]).reshape(1,-1), rvec, tvec, mtx, dist)
    pic = pics[cnt]
    pic = cv2.line(pic, tuple(source[0][0]), tuple(axis_x[0][0]), (255, 0, 0), 5)
    pic = cv2.line(pic, tuple(source[0][0]), tuple(axis_y[0][0]), (0, 255, 0), 5)
    pic = cv2.line(pic, tuple(source[0][0]), tuple(axis_z[0][0]), (0, 0, 255), 5)
    cv2.imwrite('visualize_' + str(cnt) + '.png', pic)
    
    # Calculate camera position and roll, pitch and yaw.
    Rt, _ = cv2.Rodrigues(rvec.reshape(1,-1))
    R = Rt.transpose()
    pos = -np.dot(R, tvec)
    print('Camera position for pic[' + str(cnt) + '] is x: ' + str(pos[0][0]) + ', y: ' + str(pos[1][0]) + ', z: ' + str(pos[2][0]) + '(cm).')
    roll = atan2(-R[2][1], R[2][2])
    pitch = asin(R[2][0])
    yaw = atan2(-R[1][0], R[0][0])
    print('roll =', roll)
    print('pitch =', pitch)
    print('yaw =', yaw)
