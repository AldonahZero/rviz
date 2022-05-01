#! /usr/bin/env python3.8
import numpy as np
from collections import deque
import os

corners_3d_velo = np.array([
  [15.12424719, 16.06928938, 12.27779014, 11.33274795, 15.10334837, 16.04839056,
  12.25689133, 11.31184914],
  [ 6.49985556,  4.9406537,  2.64287284,  4.2020747, 6.47872485, 4.91952299,
   2.62174213, 4.18094399],
  [-1.70710774, -1.71370686, -1.7776097,  -1.77101058,  0.29267156,  0.28607244,
   0.2221696,   0.22876872]])

corners_3d_velo = corners_3d_velo.T
ego_car = np.array([[2.15, 0.9, -1.73], [2.15, -0.9, -1.73], [-1.95, -0.9, -1.73],[-1.95,0.9,-1.73], [2.15,0.9,-0.23], [2.15,-0.9,-0.23],[-1.95,-0.9,-0.23],[-1.95,0.9,-0.23]]
                   )


def compute_3d_box_cam2(h, w, l, x, y, z, yaw):
    """
    Return : 3xn in cam2 coordinate
    """
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d_cam2[0,:] += x
    corners_3d_cam2[1,:] += y
    corners_3d_cam2[2,:] += z
    return corners_3d_cam2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def distance_point_to_segment(P,A,B):
  """
  calculates the min distance of point P to a segment AB.
  return min distance and point q
  """

  AP = P-A
  BP = P-B
  AB = B-A
  # 锐角，投影点在线段上
  if np.dot(AB,AP)>=0 and np.dot(-AB,BP)>=0:
    return np.abs(np.cross(AP,AB))/np.linalg.norm(AB), np.dot(AP,AB)/np.dot(AB,AB)*AB+A
  # 否则线段外
  d_PA = np.linalg.norm(AP)
  d_PB = np.linalg.norm(BP)
  if d_PA < d_PB:
    return d_PA, A
  return d_PB, B

# 计算两个3d框的最短距离

def min_distance_cuboids(cub1,cub2):
  """
  compute min dist between two non-overlapping cuboids of shape (8,4)
  """

  minD = 1e5
  for i in range(4):
    for j in range(4):
      d, Q = distance_point_to_segment(cub1[i,:2], cub2[j,:2], cub2[j+1,:2])
      if d < minD:
        minD = d
        minP = ego_car[i,:2]
        minQ = Q
  for i in range(4):
    for j in range(4):
      d, Q = distance_point_to_segment(cub1[i,:2], cub2[j,:2], cub2[j+1,:2])
      if d < minD:
        minD = d
        minP = corners_3d_velo[i,:2]
        minQ = Q
  return minP, minQ, minD

class Object():
    #trajectory
    def __init__(self, center, max_length):
        # 所有过去的位置
        self.locations = deque(maxlen=max_length) # save loc
        self.locations.appendleft(center)
        self.max_length = max_length


    def update(self, center, displacement, yaw):
        """
        Update the center of the object, and calculates the velocity
        """
        for i in range(len(self.locations)):
            x0, y0 = self.locations[i]
            x1 = x0 * np.cos(yaw) + y0 * np.sin(yaw) - displacement
            y1 = -x0 * np.sin(yaw) + y0 * np.cos(yaw)
            self.locations[i] = np.array([x1, y1])

        if center is not None:
            self.locations.appendleft(center)

    def reset(self):
        self.locations = deque(maxlen=self.max_length)
