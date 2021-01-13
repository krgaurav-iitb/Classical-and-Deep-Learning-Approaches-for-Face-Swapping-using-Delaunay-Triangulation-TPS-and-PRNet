#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:18:38 2020

@author: kumar
"""





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import math
import numpy as np
import dlib
from imutils import face_utils
import imutils
import matplotlib.pyplot as plt
import sys
import copy
from demo import main1


# In[2]:


#videoFile = '../Data/gaurav.mp4'
#imageFile = '../Data/deepti.jpeg'
videoFile = '../Data/TestSet_P2/Test1.mp4'
imageFile = '../Data/TestSet_P2/Rambo.jpg'
resultFile = '../Data/Results/'
detector = dlib.get_frontal_face_detector()
shapePredictor = '../Data/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(shapePredictor)
plt.rcParams["figure.figsize"] = (10,6)


# In[22]:


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def detectFacialFiducials(frame, detector, predictor, method = 'TPS'):    
    orig = copy.deepcopy(frame)
    rects = detector(orig, 1)
    shape = []
    if(len(rects) == 0):
        #gray = changeBrightness(orig, 30)
        rects = detector(orig, 2)
        if(len(rects) == 0):
            rects = detector(orig, 3)
            shape=[]
            subdiv=[]
            return shape, subdiv
            
        
    #shape_list = []
    for (i, rect) in enumerate(rects):
        shape = predictor(orig, rect)
        shape = face_utils.shape_to_np(shape)
        if method == 'DT':
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            rect_s = (x, y, w, h)
            subdiv = cv2.Subdiv2D((x, y, w, h))#changed
            bx = x + w
            by = y + h#changed
            x_start=x
            y_start=y
            for (x, y) in shape:
                if y_start < y < by and x_start< x < bx:#changed
                    subdiv.insert((x, y))
        """
        Display intermediate output
        gray_plt = copy.deepcopy(frame)
        cv2.rectangle(gray_plt, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(gray_plt, "Face #{}".format(i + 1), (x - 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        for (x, y) in shape:
            cv2.circle(gray_plt, (x, y), 1, (0, 255, 255), -1)
        plt.imshow(gray_plt), plt.show()
        """           
    if method == 'DT':
        print(len(shape))
        return shape, subdiv
    else:
        return shape

def draw_delaunay(face_to_place_on, points_face_to_place_on, face_to_copy, points_face_to_copy, subdiv):
    triangleList = subdiv.getTriangleList()
    delaunay_color = (255, 255, 255)
    traingle_num = []
    actualA = []
    coor = []
    baryB = []
    size = face_to_place_on.shape
    r = (0, 0, size[1], size[0])

    for( k, t) in enumerate(triangleList) :
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5]) 
        if (rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3)):
            index_pt1 = np.where((points_face_to_place_on == pt1).all(axis = 1))
            index_pt2 = np.where((points_face_to_place_on == pt2).all(axis = 1))
            index_pt3 = np.where((points_face_to_place_on == pt3).all(axis = 1))
            #print( len(index_pt1[0]))
            #print('index_pt2', index_pt2)
            #print('index_pt3', index_pt3)
            if (len(index_pt1[0])!=0) and (len(index_pt2[0])!=0) and (len(index_pt3[0])!=0) :#changed
                traingle_num.append((index_pt1[0][0],index_pt2[0][0],index_pt3[0][0]))
                
    #             cv2.line(face_to_place_on, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
    #             cv2.line(face_to_place_on, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
    #             cv2.line(face_to_place_on, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)
                baryB1 = []
                B = np.array([[t[0], t[2], t[4]],[t[1], t[3], t[5]],[1, 1, 1]])
                x_min = min([t[0], t[2], t[4]])
                x_max = max([t[0], t[2], t[4]])
                y_min = min([t[1], t[3], t[5]])
                y_max = max([t[1], t[3], t[5]])
                B_inv = np.linalg.inv(B)
                
                for x in range(int(x_min), int(x_max)):
                    for y in range(int(y_min), int(y_max)):
                        alpha, beta, gamma = np.matmul(B_inv, np.transpose(np.array([x, y, 1])))
    
                        if 0<=alpha<=1 and 0<=beta<=1 and 0<=gamma<=1 and (0<alpha+beta+gamma<=1.0000000000001):
                            coor.append([x, y])
                            baryB1.append([alpha, beta, gamma])
    
                if(len(baryB1) != 0):
                    baryB.append(baryB1)
                else:
                    baryB.append([])
    #print(points_face_to_place_on)
    for (k,ind) in enumerate(traingle_num):
        pt1 = tuple(points_face_to_copy[ind[0]])
        pt2 = tuple(points_face_to_copy[ind[1]])
        pt3 = tuple(points_face_to_copy[ind[2]])
#         cv2.line(face_to_copy, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
#         cv2.line(face_to_copy, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
#         cv2.line(face_to_copy, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)
        A = np.array([[pt1[0], pt2[0], pt3[0]],[pt1[1], pt2[1], pt3[1]],[1, 1, 1]])
        for c in range(len(baryB[k])):             
            temp = np.dot(A,np.transpose(baryB[k][c]))
            actualA.append([temp[0]/temp[2], temp[1]/temp[2], 1])
            
    return coor, actualA   

def swapUsingDelaunay(points_face_to_place_on, points_face_to_copy, face_to_place_on, face_to_copy, subdiv):
    mask = createMask(face_to_place_on.shape[:-1], points_face_to_place_on)
    #plt.imshow(mask), plt.show()
    rect_s1 = cv2.boundingRect(cv2.convexHull(points_face_to_place_on))
    centre= (int(rect_s1[0] + (rect_s1[2] / 2)), int(rect_s1[1] + (rect_s1[3] / 2)))
    gray1 = copy.deepcopy(face_to_place_on)
    gray2 = copy.deepcopy(face_to_place_on)
    orig = copy.deepcopy(face_to_copy)
    coor, actualA = draw_delaunay(face_to_place_on, points_face_to_place_on, face_to_copy, points_face_to_copy, subdiv)
    for i in range(0,len(actualA)):
                gray1[coor[i][1],coor[i][0],2] = orig[int(actualA[i][1]),int(actualA[i][0]),2]
                gray1[coor[i][1],coor[i][0],1] = orig[int(actualA[i][1]),int(actualA[i][0]),1]
                gray1[coor[i][1],coor[i][0],0] = orig[int(actualA[i][1]),int(actualA[i][0]),0]
    
    mask_img = cv2.bitwise_and(gray1, gray1, mask)
    #plt.imshow(mask_img), plt.show()
    clone_img = cv2.seamlessClone(mask_img, gray2, mask, centre, cv2.MIXED_CLONE)
    return clone_img 

def U(r):
    return r*r*math.log(r*r)

def thinPlateSpline(points_source, points_dest):
    p = points_source.shape[0]
    K = np.zeros((p, p))
    for j in range(p):
        K[j] = [U(np.linalg.norm((-points_source[j]+points_source[i]),ord =2)+sys.float_info.epsilon) for i in range(p)]
    P = np.append(points_source, np.ones([p, 1]), axis = 1)
    O = np.zeros((3,3))
    KPPtO = np.vstack([np.hstack([K, P]), np.hstack([P.T, O])])
    Lambda = 5
    I = np.identity(p + 3)
    temp = KPPtO + Lambda * I
    temp_inverse = np.linalg.inv(temp)
    V = np.concatenate([np.array(points_dest), np.zeros((3))])
    V.resize(V.shape[0], 1)
    weights = np.matmul(temp_inverse,V)
    return weights, K

def fxy(points1, points2, weights):
    K = np.zeros([points2.shape[0], 1])
    for i in range(points2.shape[0]):
        K[i] = U(np.linalg.norm((points2[i] - points1), ord =2)+sys.float_info.epsilon)
    f = weights[-1] + weights[-3] * points1[0] + weights[-2] * points1[1] + np.matmul(K.T, weights[0:-3])
    return f

def createMask(size, points):
    radius = 10  # kernel size
    kernel = np.ones((radius, radius), np.uint8)
    mask = np.zeros(size, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    #mask = cv2.erode(mask, kernel, iterations = 1)
    return mask

def warpUsingTPS(source_image, points_source, dest_image, points_dest, mask, weights_X, weights_Y):
    #Define bounding box
    xy1_min = np.float32([min(points_source[:,0]),min(points_source[:,1])])
    xy1_max = np.float32([max(points_source[:,0]),max(points_source[:,1])])
    x = np.arange(xy1_min[0],xy1_max[0]).astype(int)
    y = np.arange(xy1_min[1],xy1_max[1]).astype(int)
    X,Y = np.mgrid[x[0]:x[-1] + 1, y[0]:y[-1] + 1]
    pts_src = np.vstack((X.ravel(),Y.ravel()))
    xy = pts_src.T
    u = np.zeros_like(xy[:,0])
    v = np.zeros_like(xy[:,0])

    for i in range(xy.shape[0]):
        u[i] = fxy(xy[i,:], points_source, weights_X)
        v[i] = fxy(xy[i,:], points_source, weights_Y)

    #Place warped face onto source face
    warped_img = source_image.copy()
    mask_warped_img = np.zeros_like(warped_img[:,:,0])
    #print(u.shape[0])
    for i in range(u.shape[0]):
        if mask.shape[0] - v[i] <= 0:
            v[i] = mask.shape[0]-1
        if mask.shape[1] - u[i] <= 0:
            u[i] = mask.shape[1]-1
        if mask[v[i],u[i]] > 0:
            warped_img[xy[i, 1], xy[i, 0], :] = dest_image[v[i], u[i], :]
            mask_warped_img[xy[i, 1], xy[i, 0]] = 255
            
    return warped_img, mask_warped_img

def blendingPoisson(dest_img, warped_mask, warped_img):
    r = cv2.boundingRect(warped_mask)
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    output = cv2.seamlessClone(warped_img, dest_img, warped_mask, center, cv2.NORMAL_CLONE)
    return output

def changeBrightness(img, increase = True, value = 30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if(increase):
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        v[v < value] = 0
        v[v >= value] -= value
    finalHSV = cv2.merge((h, s, v))
    img = cv2.cvtColor(finalHSV, cv2.COLOR_HSV2BGR)
    return img

def swapUsingTPS(points_face_to_place_on, points_face_to_copy, face_to_place_on, face_to_copy):
    weights_X, K = thinPlateSpline(points_face_to_place_on, points_face_to_copy[:,0])
    weights_Y, K = thinPlateSpline(points_face_to_place_on, points_face_to_copy[:,1])
    mask = createMask(face_to_place_on.shape[:-1], points_face_to_copy)
    frame_orig = copy.deepcopy(face_to_place_on)
    warped_img, mask_warped_img = warpUsingTPS(face_to_place_on, points_face_to_place_on, face_to_copy, points_face_to_copy, mask, weights_X, weights_Y)
    result = blendingPoisson(frame_orig, mask_warped_img, warped_img)
    return result


# In[23]:


cap = cv2.VideoCapture(videoFile)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
ret, frame = cap.read()
vw = cv2.VideoWriter(resultFile + "rambo_TPS.avi", fourcc, 30, (frame.shape[1], frame.shape[0]))

#Read the image containing face that would be copied
face_to_copy = cv2.cvtColor(cv2.imread(imageFile), cv2.COLOR_BGR2RGB)
#points_face_to_copy = detectFacialFiducials(face_to_copy, detector, predictor)
data = np.loadtxt("/home/kumar/Desktop/MSCS/AdvanceCV/faceSwap/PRNet/faceDataOutput/Scarlett_kpt.txt").astype(int)
points_face_to_copy=data[:,0:2]
#points_face_to_copy = main1()
method = 'TPS'
frameCount = 1
#face_to_copy = changeBrightness(face_to_copy, False, 30)

#flag = 1
#while(flag):
while(cap.isOpened()): 
    ret, frame = cap.read()
    if frame is not None:#changed
        frameCount+=1
        print frameCount
        face_to_place_on = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(face_to_place_on, cv2.COLOR_RGB2GRAY)
        result = copy.deepcopy(face_to_place_on)
            
        if method == 'DT':
            points_face_to_place_on, subdiv = detectFacialFiducials(face_to_place_on, detector, predictor, method)
            if(len(points_face_to_place_on)!=0):
                result = swapUsingDelaunay(points_face_to_place_on, points_face_to_copy, face_to_place_on, face_to_copy, subdiv)
            
                
        
        else:
            points_face_to_place_on = detectFacialFiducials(face_to_place_on, detector, predictor)
            result = swapUsingTPS(points_face_to_place_on, points_face_to_copy, face_to_place_on, face_to_copy)
        
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        vw.write(result)
    else:#changed
        break#changed
    #plt.imshow(result), plt.show()
    #flag = flag-1
    
vw.release()
cap.release()


# In[18]:


plt.scatter(points_face_to_place_on[:,1], points_face_to_place_on[:,0])


# In[16]:


plt.imshow(result), plt.show()


# In[ ]:





