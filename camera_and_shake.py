# utils_robot.py
# 同济子豪兄 2024-5-22
# 启动并连接机械臂，导入各种工具包

print('导入机械臂连接模块')

import ctypes
import cv2
import numpy as np
import os
from multiprocessing import Process, Value, Queue
import pyrealsense2.pyrealsense2 as rs
from serial.tools import list_ports
from pydobot import Dobot
import time
from serial.tools import list_ports




# Initialize Dobot
port = list_ports.comports()[0].device
device = Dobot(port=port, verbose=True)
device.suck(enable=False)
(x, y, z, r, j1, j2, j3, j4) = device.pose()
print(f'x:{x} y:{y} z:{z} j1:{j1} j2:{j2} j3:{j3} j4:{j4}')


def back_zero(device):
    print('机械臂归零')
    if device is None:
        print("错误：device 对象未初始化")
        return
    # print(f"移动到初始位置: {initial_pos}")
    # Set the initial position (x, y, z)
    initial_pos = [200, 0, 0]  # Example initial position, adjust as needed
    device.move_to(initial_pos[0], initial_pos[1], initial_pos[2], 0, wait=True)
    time.sleep(2)

    # 调用
    # back_zero(device)

def move_to_position(device, target_pos, workspace_radius=330):
    """将机械臂移动到指定位置，如果位置在工作范围内"""
    # 检查目标位置是否在工作范围内
    if is_within_workspace(target_pos[0], target_pos[1], workspace_radius):
        print(f"移动到指定位置: {target_pos}")
        # 移动到目标位置 (x, y, z, r) 并等待完成
        device.move_to(target_pos[0], target_pos[1], target_pos[2], 0, wait=True)
        time.sleep(2)  # 可选延迟，如果不需要可以删除
    else:
        print(f"目标位置 {target_pos} 超出了工作范围。")


def is_within_workspace(x, y, workspace_radius=330):
    """检查给定的XY坐标是否在机械臂的工作范围内"""
    return x * x + y * y <= workspace_radius * workspace_radius


def head_shake(device):
    """
    控制 Dobot 机械臂通过改变 y轴实现摇头。
    """

    # 设置摆动角度和速度
    shake_angle = 10  # 摆动角度
    # shake_speed = 80  # 摆动速度（80%）

    # 开始摆头动作
    print("开始左右摇头...")
    for _ in range(2):  # 摆动2次
        device.move_to(x, y+shake_angle, z, r, wait=False)
        device.move_to(x, y, z, r, wait=True)  # we wait until this movement is done before continuing
        print("摇头完成！")
    # 调用
    # head_shake(device)
    # device.close()

def head_nod(device):
    """
    控制 Dobot 机械臂通过改变 z轴实现点头。
    """

    # 设置摆动角度和速度
    nod_angle = 10  # 摆动角度
    # shake_speed = 80  # 摆动速度（80%）

    # 开始摆头动作
    print("开始上下点头...")
    for _ in range(2):  # 摆动2次
        device.move_to(x, y, z+ nod_angle, r, wait=False)
        device.move_to(x, y, z, r, wait=True)  # we wait until this movement is done before continuing
    print("点头完成！")

def pump_on():
    '''
    开启吸泵
    '''
    print('开启吸泵')
    device.suck(enable=True)  # 开启吸泵

def pump_off():
    '''
    关闭吸泵，吸泵放气，释放物体
    '''
    print('关闭吸泵')
    device.suck(enable=False)  # 关闭吸泵
    time.sleep(0.05)

def realsense_video():
    # Initialize Realsense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    align = rs.align(rs.stream.color)
    pipeline.start(config)

    alpha_val = 0.219

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            # 获取深度帧
            depth = frames.get_depth_frame()

            # 获取颜色帧
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())

            # 获取相机内参和畸变系数
            intr = color_frame.profile.as_video_stream_profile().intrinsics
            intr_matrix = np.array([
                [intr.fx, 0, intr.ppx],
                [0, intr.fy, intr.ppy],
                [0, 0, 1]
            ], dtype=np.float32)
            intr_coeffs = np.array(intr.coeffs, dtype=np.float32)

            # 对颜色图像进行畸变矫正
            undistorted_image = cv2.undistort(color_image, intr_matrix, intr_coeffs)
            # 获取深度图
            depth_img = np.asanyarray(depth.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=alpha_val), cv2.COLORMAP_JET
            )

            # 堆叠颜色帧和深度帧
            images = np.vstack((undistorted_image, depth_colormap))
            cv2.imshow("image", images)

            # 按键拍照/退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                cv2.imwrite("temp/vl_now.jpg", undistorted_image)
                print("Image saved as temp/vl_now.jpg")
                return depth, intr_matrix  # 返回深度帧和相机内参
            elif key == ord("q"):
                print("Exiting...")
                break
    finally:
        pipeline.stop()

# head_nod(device)
# move_to_position(device, [200, 0, 0])  # 传入你想要的目标坐标
# device.close()
# back_zero(device)