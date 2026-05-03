# utils_robot.py
# 启动并连接机械臂，导入各种工具包

print('导入机械臂连接模块')

import ctypes
import cv2
import numpy as np
import os
from multiprocessing import Process, Value, Queue
import time
from serial.tools import list_ports
from src.coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import sys
import struct
import depth_image_encoding as encoding




# 初始化仿真
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
targetObj = sim.getObject('/target')
tipobj = sim.getObject('/Dobot/tip')
visionSensorHandle = sim.getObject('/vision_1')
deepSensorHandle = sim.getObject('/vision_1/depth')
DobotHandle = sim.getObject('/Dobot')
gripperHandle = sim.getObject('/Dobot/suctionCup_link2')
# Cuboid_1Handle = sim.getObject('/Cuboid_1')

# 获取机械臂当前末端三维坐标作为初始位置
time.sleep(1)
initial_pos = sim.getObjectPosition(targetObj, DobotHandle)

# # 获取方块的边界框（bounding box）
# minXYZ, maxXYZ = sim.getShapeBB(Cuboid_1Handle)
#
# # 计算方块的高度
# height = maxXYZ[2] - minXYZ[2]
#
# print(f"抓取物的高度为: {height} 米")


def back_zero():
    print('机械臂归零')
    sim.setObjectPosition(targetObj, DobotHandle, initial_pos)
    time.sleep(2)


def move_to_position(target_pos):
    """将机械臂移动到指定位置，如果位置在工作范围内"""
    # 检查目标位置是否在工作范围内
    if is_within_workspace(target_pos[0], target_pos[1]):
        print(f"移动到指定位置: {target_pos}")
        # 移动到目标位置 (x, y, z, r) 并 等待完成
        sim.setObjectPosition(targetObj, DobotHandle, target_pos)
        time.sleep(2)  # 可选延迟，如果不需要可以删除
    else:
        print(f"目标位置 {target_pos} 超出了工作范围。")


def is_within_workspace(x, y):
    """检查给定的XY坐标是否在机械臂的工作范围内"""
    return 0.01731189 <= x * x + y * y <= 0.12252


def head_shake():
    """
    控制 Dobot 机械臂通过改变 y轴实现摇头。
    """

    # 设置摆动角度和速度
    shake_angle = 0.1  # 摆动角度
    # shake_speed = 80  # 摆动速度（80%）

    # 开始摆头动作
    print("开始左右摇头...")
    for _ in range(3):  # 摆动5次
        shake_position_up = [initial_pos[0] + shake_angle, initial_pos[1], initial_pos[2]]
        shake_position_down = [initial_pos[0] - shake_angle, initial_pos[1], initial_pos[2]]

        sim.setObjectPosition(targetObj, DobotHandle, shake_position_up)
        time.sleep(1)
        sim.setObjectPosition(targetObj, DobotHandle, shake_position_down)
        time.sleep(1)

    sim.setObjectPosition(targetObj, DobotHandle, initial_pos)
    print("摇头完成！")
    # 调用
    # head_shake(device)
    # device.close()

def head_nod():
    """
    控制 Dobot 机械臂通过改变 z轴实现点头。
    """

    # 设置摆动角度和速度
    nod_angle = 0.05  # 摆动角度

    # 开始摆头动作
    print("开始点头...")
    for _ in range(3):  # 摆动5次
        nod_position_up = [initial_pos[0], initial_pos[1], initial_pos[2] + nod_angle]
        nod_position_down = [initial_pos[0], initial_pos[1], initial_pos[2] - nod_angle * 2]

        sim.setObjectPosition(targetObj, DobotHandle, nod_position_up)
        time.sleep(1)
        sim.setObjectPosition(targetObj, DobotHandle, nod_position_down)
        time.sleep(1)

    sim.setObjectPosition(targetObj, DobotHandle, initial_pos)
    print("点头完成！")

def pump_on():
    """
    开启吸盘
    """
    sim.writeCustomStringData(gripperHandle, 'activity', 'on')
    print("吸泵开启")
    time.sleep(1)


def pump_off():
    '''
    关闭吸泵，吸泵放气，释放物体
    '''
    sim.writeCustomStringData(gripperHandle, 'activity', 'off')
    print("吸泵关闭")
    time.sleep(1)

# 获取RGB和深度图像
def get_rgb_depth_images():
    while True:
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

        # 处理按键事件
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # 捕获当前帧并保存
            cv2.imwrite("temp/vl_now.jpg", img)
            print("Image saved as temp/vl_now.jpg")
        elif key == ord('q'):
            cv2.destroyAllWindows()
            break  # 退出循环

    return img, depth_image, resX

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

    # # 获取相机的全局变换矩阵
    # camera_matrix = sim.getObjectMatrix(visionSensorHandle)  # -1 表示世界坐标系
    # print(camera_matrix)
    # # 如果返回的是一个包含12个元素的扁平化数组，先把它转换为3x4的矩阵
    # camera_matrix = np.array(camera_matrix).reshape(3, 4)
    #
    # # 用一个[0, 0, 0, 1]的行向量补充成4x4矩阵
    # camera_matrix = np.vstack([camera_matrix, np.array([0, 0, 0, 1])])
    #
    # # 将相机坐标系下的三维坐标转换到世界坐标系下的三维坐标
    # cam_coords = np.array([x_cam, y_cam, z_cam, 1])
    # world_coords = np.dot(camera_matrix, cam_coords)
    # x_world, y_world, z_world = world_coords[:3]
    #
    # return x_world, y_world, z_world

    # 获取相机相对于机械臂的变换矩阵
    relative_matrix = sim.getObjectMatrix(visionSensorHandle, DobotHandle)

    # 如果返回的是一个包含12个元素的扁平化数组，先把它转换为3x4的矩阵
    relative_matrix = np.array(relative_matrix).reshape(3, 4)

    # 用一个[0, 0, 0, 1]的行向量补充成4x4矩阵
    relative_matrix = np.vstack([relative_matrix, np.array([0, 0, 0, 1])])

    # 将相机坐标系下的三维坐标转换到机械臂坐标系下的三维坐标
    cam_coords = np.array([x_cam, y_cam, z_cam, 1])
    arm_coords = np.dot(relative_matrix, cam_coords)
    x_arm, y_arm, z_arm = arm_coords[:3]

    return x_arm, y_arm, z_arm
# head_nod()
# head_shake()
# move_to_position((0.20235115450372654, -0.16801518117871317, 0.16825887714883553))  # 传入你想要的目标坐标
# time.sleep(5)
# back_zero()
# get_rgb_depth_images()