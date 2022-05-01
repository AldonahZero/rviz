#! /usr/bin/env python3.8
import rospy
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix
from geometry_msgs.msg import Point
import sensor_msgs.point_cloud2 as pcl2
from cv_bridge import CvBridge
import tf
import  cv2
import numpy as np

FRAME_ID = 'map'
DETECTION_COLOR_MAP = {'Car': (255,255,0), 'Pedestrian': (0, 226, 255), 'Cyclist': (141, 40, 255)} # color for detection, in format bgr
RATE = 10
LIFETIME = 1.0/RATE # 1/rate

# connect vertic
LINES = [[0, 1], [1, 2], [2, 3], [3, 0]] # lower face
LINES+= [[4, 5], [5, 6], [6, 7], [7, 4]] # upper face
LINES+= [[4, 0], [5, 1], [6, 2], [7, 3]] # connect lower face and upper face
LINES+= [[4, 1], [5, 0]] # front face and draw x

def publish_camera(cam_pub, bridge, image, borders_2d_cam2s=None, object_types=None, log=False):
    """
        Publish image in bgr8 format
        If borders_2d_cam2s is not None, publish also 2d boxes with color specified by object_types
        If object_types is None, set all color to cyan
        """
    if borders_2d_cam2s is not None:
        for i, box in enumerate(borders_2d_cam2s):
            top_left = int(box[0]), int(box[1])
            bottom_right = int(box[2]), int(box[3])
            if object_types is None:
                cv2.rectangle(image, top_left, bottom_right, (255, 255, 0), 2)
            else:
                cv2.rectangle(image, top_left, bottom_right, DETECTION_COLOR_MAP[object_types[i]], 2)
    cam_pub.publish(bridge.cv2_to_imgmsg(image, "bgr8"))

def publish_point_cloud(pcl_pub,point_cloud):
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = FRAME_ID
    pcl_pub.publish(pcl2.create_cloud_xyz32(header, point_cloud[:,:3]))

def publish_ego_car(ego_car_pub):
    # publish left and right 45 degree FOV lines and ego car model mesh
    marker_array = MarkerArray()
    marker = Marker()
    marker.header.frame_id = FRAME_ID
    marker.header.stamp = rospy.Time.now()

    marker.id = 0
    marker.action = Marker.ADD
    marker.lifetime = rospy.Duration()
    marker.type = Marker.LINE_STRIP
    # line
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.scale.x = 0.2 # line width

    marker.points = []

    # check the kitti axis model
    marker.points.append(Point(5,-5,0)) # left up
    marker.points.append(Point(0,0,0)) # center
    marker.points.append(Point(5, 5,0)) # right up
    marker_array.markers.append(marker)

    mesh_marker = Marker()
    mesh_marker.header.frame_id = FRAME_ID
    mesh_marker.header.stamp = rospy.Time.now()

    mesh_marker.id = -1
    mesh_marker.lifetime = rospy.Duration()
    mesh_marker.type = Marker.MESH_RESOURCE
    mesh_marker.mesh_resource = "package://kitti_tutorial/YYCar/BMW X5 4.dae"  # LOAD ERROR, DON'T KNOW WHY

    mesh_marker.pose.position.x = 0.0
    mesh_marker.pose.position.y = 0.0
    mesh_marker.pose.position.z = -1.73

    q = tf.transformations.quaternion_from_euler(np.pi / 2, 0, np.pi)
    mesh_marker.pose.orientation.x = q[0]
    mesh_marker.pose.orientation.y = q[1]
    mesh_marker.pose.orientation.z = q[2]
    mesh_marker.pose.orientation.w = q[3]

    mesh_marker.color.r = 1.0
    mesh_marker.color.g = 1.0
    mesh_marker.color.b = 1.0
    mesh_marker.color.a = 1.0

    mesh_marker.scale.x = 0.9
    mesh_marker.scale.y = 0.9
    mesh_marker.scale.z = 0.9
    marker_array.markers.append(mesh_marker)

    ego_car_pub.publish(marker_array)


def publish_imu(imu_pub, imu_data, log=False):
    """
    Publish IMU data
    http://docs.ros.org/melodic/api/sensor_msgs/html/msg/Imu.html
    """
    imu = Imu()
    imu.header.frame_id = FRAME_ID
    imu.header.stamp = rospy.Time.now()
    q = tf.transformations.quaternion_from_euler(float(imu_data.roll), float(imu_data.pitch), \
                                                     float(imu_data.yaw)) # prevent the data from being overwritten
    imu.orientation.x = q[0]
    imu.orientation.y = q[1]
    imu.orientation.z = q[2]
    imu.orientation.w = q[3]
    imu.linear_acceleration.x = imu_data.af
    imu.linear_acceleration.y = imu_data.al
    imu.linear_acceleration.z = imu_data.au
    imu.angular_velocity.x = imu_data.wf
    imu.angular_velocity.y = imu_data.wl
    imu.angular_velocity.z = imu_data.wu

    imu_pub.publish(imu)
    if log:
        rospy.loginfo("imu msg published")


def publish_gps(gps_pub, gps_data, log=False):
    """
    Publish GPS data
    """
    gps = NavSatFix()
    gps.header.frame_id = FRAME_ID
    gps.header.stamp = rospy.Time.now()
    gps.latitude = gps_data.lat
    gps.longitude = gps_data.lon
    gps.altitude = gps_data.alt

    gps_pub.publish(gps)
    if log:
        rospy.loginfo("gps msg published")


def publish_3dbox(box3d_pub, corners_3d_velos, track_ids, types=None, publish_id=True):
    """
    Publish 3d boxes in velodyne coordinate, with color specified by object_types
    If object_types is None, set all color to cyan
    corners_3d_velos : list of (8, 4) 3d corners
    """
    marker_array = MarkerArray()
    for i, corners_3d_velo in enumerate(corners_3d_velos):
        # 3d box
        marker = Marker()
        marker.header.frame_id = FRAME_ID
        marker.header.stamp = rospy.Time.now()
        marker.id = i
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(LIFETIME)
        marker.type = Marker.LINE_LIST

        b, g, r = DETECTION_COLOR_MAP[types[i]]
        if types is None:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
        else:
            marker.color.r = r/255.0
            marker.color.g = g/255.0
            marker.color.b = b/255.0
        marker.color.a = 1.0
        marker.scale.x = 0.1

        marker.points = []
        for l in LINES:
            p1 = corners_3d_velo[l[0]]
            marker.points.append(Point(p1[0], p1[1], p1[2]))
            p2 = corners_3d_velo[l[1]]
            marker.points.append(Point(p2[0], p2[1], p2[2]))
        marker_array.markers.append(marker)

        # add track id
        if publish_id:
            track_id = track_ids[i]
            text_marker = Marker()
            text_marker.header.frame_id = FRAME_ID
            text_marker.header.stamp = rospy.Time.now()

            text_marker.id = track_id + 1000
            text_marker.action = Marker.ADD
            text_marker.lifetime = rospy.Duration(LIFETIME)
            text_marker.type = Marker.TEXT_VIEW_FACING

            p4 = corners_3d_velo[4] # upper front left corner

            text_marker.pose.position.x = p4[0]
            text_marker.pose.position.y = p4[1]
            text_marker.pose.position.z = p4[2] + 0.5

            text_marker.text = 'track_id:' + str(track_id)

            text_marker.scale.x = 1
            text_marker.scale.y = 1
            text_marker.scale.z = 1

            if types is None:
                text_marker.color.r = 0.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
            else:
                b, g, r = DETECTION_COLOR_MAP[types[i]]
                text_marker.color.r = r/255.0
                text_marker.color.g = g/255.0
                text_marker.color.b = b/255.0
            text_marker.color.a = 1.0
        marker_array.markers.append(text_marker)

    box3d_pub.publish(marker_array)

def publish_loc(loc_pub, locations):
    marker_array = MarkerArray()

    marker = Marker()
    marker.header.frame_id = FRAME_ID
    marker.header.stamp = rospy.Time.now()

    marker.action = Marker.ADD
    marker.lifetime = rospy.Duration(LIFETIME)
    marker.type = Marker.LINE_STRIP

    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.scale.x = 0.2

    marker.points = []
    for p in locations:
        marker = Marker()
        marker.points.append(Point(p[0], p[1], 0))

    marker_array.markers.append(marker)
    imu_odom_pub.publish(marker_array)

def publish_imu_odom(imu_odom_pub, tracker, centers):
    marker_array = MarkerArray()

    for track_id in centers:

        marker = Marker()
        marker.header.frame_id = FRAME_ID
        marker.header.stamp = rospy.Time.now()

        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(LIFETIME)
        marker.type = Marker.LINE_STRIP
        marker.id = track_id

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.scale.x = 0.2

        marker.points = []
        for p in tracker[track_id].locations:
            marker.points.append(Point(p[0], p[1], 0))

        marker_array.markers.append(marker)
    imu_odom_pub.publish(marker_array)

def publish_dist(dist_pub, minPQDs):
    marker_array = MarkerArray()

    for i, (minP, minQ, minD) in enumerate(minPQDs):
        marker = Marker()
        marker.header.frame_id = FRAME_ID
        marker.header.stamp = rospy.Time.now()

        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(LIFETIME)
        marker.type = Marker.LINE_STRIP
        marker.id = i

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.5
        marker.scale.x = 0.1

        marker.points = []
        marker.points.append(Point(minP[0], minP[1], 0))
        marker.points.append(Point(minQ[0], minQ[1], 0))

        marker_array.markers.append(marker)

        text_marker = Marker()
        text_marker.header.frame_id = FRAME_ID
        text_marker.header.stamp = rospy.Time.now()

        text_marker.id = i + 1000
        text_marker.action = Marker.ADD
        text_marker.lifetime = rospy.Duration(LIFETIME)
        text_marker.type = Marker.TEXT_VIEW_FACING

        p = (minP + minQ) / 2.0
        text_marker.pose.position.x = p[0]
        text_marker.pose.position.y = p[1]
        text_marker.pose.position.z = 0.0

        text_marker.text = '%.2fm'%minD
        text_marker.scale.x = 1
        text_marker.scale.y = 1
        text_marker.scale.z = 1

        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 0.8
        marker_array.markers.append(text_marker)

    dist_pub.publish(marker_array)