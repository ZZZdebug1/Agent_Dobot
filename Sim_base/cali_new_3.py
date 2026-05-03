import math

from src.coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import sys
import cv2
import numpy as np
import struct
import depth_image_encoding as encoding

# 加载内参和畸变系数
data = np.load('calibration_results.npz')
mtx = data['mtx']
dist = data['dist']

client = RemoteAPIClient()
sim = client.require('sim')

# 启动仿真
sim.startSimulation()
print("Simulation start")

# 获得对象的句柄
tipHandle = sim.getObject('/Dobot/tip')  # 你需要替换为实际的末端执行器句柄
targetHandle = sim.getObject('/target')  # 你需要替换为实际的目标对象句柄
# targetObj = sim.getObject('/target')
visionSensorHandle = sim.getObject('/vision_1')
deepSensorHandle = sim.getObject('/vision_1/depth')

# 获得对象的句柄
gripperHandle = sim.getObject('/Dobot/suctionCup_link2')
jointHandles = []
for i in range(4):  # 假设Dobot有4个关节
    jointHandles.append(sim.getObject(f'/Dobot/motor{i+1}'))


def moveToConfig(handles, maxVel, maxAccel, maxJerk, targetConf, enable):
    # Convert targetConf from degrees to radians
    targetConf = [angle * math.pi / 180 for angle in targetConf]

    # Create a parameters dictionary for the movement
    params = {
        'joints': handles,
        'targetPos': targetConf,
        'maxVel': maxVel,
        'maxAccel': maxAccel,
        'maxJerk': maxJerk,
    }
    sim.moveToConfig(params)

    # Handle gripper activation based on the 'enable' parameter
    if enable:
        sim.writeCustomStringData(gripperHandle, 'activity', 'on')
    else:
        sim.writeCustomStringData(gripperHandle, 'activity', 'off')


# Define a function for the thread that controls the movement
modelBase = sim.getObject('/Dobot')
gripperHandle = sim.getObject('/Dobot/suctionCup_link2')

# Get the motor handles for the Dobot
motorHandles = []
for i in range(1, 5):
    motorHandles.append(sim.getObject(f'/motor{i}'))

# Get auxiliary motors if necessary
auxMotor1 = sim.getObject('/auxMotor1')
auxMotor2 = sim.getObject('/auxMotor2')

# Define velocity, acceleration, and jerk values in degrees per second
vel = 22
accel = 40
jerk = 80

# Convert velocity, acceleration, and jerk to radians per second
# maxVel = [vel * math.pi / 180] * 4
# maxAccel = [accel * math.pi / 180] * 4
# maxJerk = [jerk * math.pi / 180] * 4
maxVel = [vel, vel, vel, vel]
maxAccel = [accel, accel, accel, accel]
maxJerk = [jerk, jerk, jerk, jerk]



# 获取RGB和深度图像
def get_rgb_depth_images():
    # 获取RGB图像
    img, [resX, resY] = sim.getVisionSensorImg(visionSensorHandle)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
    # img = np.flipud(img)  # 上下翻转
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 转换为BGR格式


    # 矫正畸变
    img = cv2.undistort(img, mtx, dist)

    # 获取深度图像
    Deepdate = sim.getVisionSensorDepth(deepSensorHandle, 1)
    num_floats = Deepdate[1][0] * Deepdate[1][1]
    depth_data = struct.unpack(f'{num_floats}f', Deepdate[0])
    depth_array = np.array(depth_data)
    depth_image = depth_array.reshape((Deepdate[1][1], Deepdate[1][0]))
    depth_image_show = np.flipud(depth_image)

    # 深度图像着色
    deepcolor = encoding.FloatArrayToRgbImage(depth_image, scale_factor=256 * 3.5 * 1000)

    # 显示RGB图像和深度图像
    img_show = np.flipud(img)  # 上下翻转
    cv2.imshow('RGB Image', img_show)
    cv2.imshow('Depth Image', depth_image_show)
    # cv2.imshow('Depth Image', deepcolor)



    return img, depth_image, resX


# 检测红色物体并返回中心点坐标和深度值
def detect_red_object(img, depth_image):
    # 转换到HSV颜色空间以便过滤红色
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 红色的HSV范围（低阈值和高阈值）
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv_img, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv_img, lower_red, upper_red)

    # 合并两个红色的掩膜
    mask = mask1 + mask2

    # 进行形态学操作去噪点
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 查找红色物体的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_object_info = None
    if contours:
        # 找到最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)

        # 计算轮廓的中心点
        moments = cv2.moments(largest_contour)
        if moments['m00'] != 0:
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])

            # 在图像中标记中心点
            cv2.circle(img, (center_x, center_y), 4, (0, 255, 0), -1)
            # print(f"Red object center at: ({center_x}, {center_y})")

            # 获取深度图像中的深度值
            depth_value = depth_image[center_y, center_x]
            # print(f"Depth at ({center_x}, {center_y}): {depth_value} meters")

            # 可选：将检测到的红色物块轮廓画出来
            cv2.drawContours(img, [largest_contour], -1, (0, 255, 0), 1)

            # 返回中心点坐标和深度值
            red_object_info = (center_x, center_y, depth_value)

            # 将像素坐标转换为世界坐标
            world_coords = pixel_to_world(center_x, center_y, depth_value, mtx, resX)

    # 显示更新后的RGB图像
    # img_show = np.flipud(img)  # 上下翻转
    # cv2.imshow('Detected Red Object', img_show)

    return red_object_info, world_coords


def pixel_to_world(center_x, center_y, depth_value, mtx, resX):
    # 相机内参
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]

    # 修改像素坐标，水平翻转中心点的x坐标
    center_x = resX - center_x  # 进行水平翻转
    # center_y = resY - center_y  # 进行水平翻转

    # 将像素坐标转换为归一化图像平面坐标
    x = (center_x - cx) / fx
    y = (center_y - cy) / fy

    # 将归一化图像平面坐标和深度值转换为相机坐标系下的三维坐标
    z_cam = depth_value
    x_cam = x * z_cam
    y_cam = y * z_cam

    # 获取相机的全局变换矩阵
    camera_matrix = sim.getObjectMatrix(visionSensorHandle)  # -1 表示世界坐标系
    print(camera_matrix)
    # 如果返回的是一个包含12个元素的扁平化数组，先把它转换为3x4的矩阵
    camera_matrix = np.array(camera_matrix).reshape(3, 4)

    # 用一个[0, 0, 0, 1]的行向量补充成4x4矩阵
    camera_matrix = np.vstack([camera_matrix, np.array([0, 0, 0, 1])])

    # 将相机坐标系下的三维坐标转换到世界坐标系下的三维坐标
    cam_coords = np.array([x_cam, y_cam, z_cam, 1])
    world_coords = np.dot(camera_matrix, cam_coords)
    x_world, y_world, z_world = world_coords[:3]

    return x_world, y_world, z_world

# def world_to_dobot(world_coords, dobot_handle):
#     # 获取Dobot末端执行器相对于世界坐标系的变换矩阵
#     dobot_matrix = sim.getObjectMatrix(dobot_handle, -1)  # -1 表示世界坐标系
#
#     # 将扁平化的12元素数组转换为3x4矩阵
#     dobot_matrix = np.array(dobot_matrix).reshape(3, 4)
#     # 用一个[0, 0, 0, 1]的行向量补充成4x4矩阵
#     dobot_matrix = np.vstack([dobot_matrix, np.array([0, 0, 0, 1])])
#
#     # 将世界坐标转换为Dobot坐标系下的坐标
#     world_coords_homogeneous = np.array([world_coords[0], world_coords[1], world_coords[2], 1])
#     dobot_coords = np.dot(dobot_matrix.T, world_coords_homogeneous)
#
#     return dobot_coords[:3]

# 初始化状态
is_capturing_frame = False  # 用于管理当前是否处于静态帧模式
current_world_coords = None  # 保存当前检测到的世界坐标

while True:
    if not is_capturing_frame:
        img, depth_image, resX = get_rgb_depth_images()
        # img_show = np.flipud(img)  # 上下翻转
        # cv2.imshow('Video Stream', img_show)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # 捕获当前帧并进行红色物体检测
        img, depth_image, resX = get_rgb_depth_images()
        red_object_info, world_coords = detect_red_object(img, depth_image)
        if red_object_info:
            center_x, center_y, depth_value = red_object_info
            print(f"Detected Red Object: Pixel({center_x}, {center_y}), Depth: {depth_value}m")
            print(f"World Coordinates: {world_coords}")
            cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), -1)  # 标记中心点
            current_world_coords = world_coords  # 保存当前世界坐标
        img_show = np.flipud(img)  # 上下翻转
        cv2.imshow('Detected Frame', img_show)
        is_capturing_frame = True

    elif key == ord('c') and current_world_coords:
        print(f"Moving target to: {current_world_coords}")
        # 获取目标对象的全局位置
        # goalPos = [current_world_coords[0], current_world_coords[1], current_world_coords[2]]

        # 获取目标对象的全局方向（姿态）
        # goalOrientation = sim.getObjectOrientation(targetHandle, -1)  # -1 表示世界坐标系

        # dobot_coords = world_to_dobot(current_world_coords, tipHandle)

        goalOrientation = sim.getObjectPose(targetHandle, -1)
        goalTr = goalOrientation.copy()
        goalTr[0] = current_world_coords[0]
        goalTr[1] = current_world_coords[1]
        goalTr[2] = current_world_coords[2]

        # 设置移动参数
        params = {}
        params['ik'] = {'tip': tipHandle, 'target': targetHandle}
        params['targetPose'] = goalTr
        params['maxVel'] = maxVel
        params['maxAccel'] = maxAccel
        params['maxJerk'] = maxJerk

        # 执行移动
        sim.moveToPose(params)

    elif key == ord('q'):
        if is_capturing_frame:
            is_capturing_frame = False
            cv2.destroyWindow('Detected Frame')
        else:
            break



# 关闭所有窗口
cv2.destroyAllWindows()
