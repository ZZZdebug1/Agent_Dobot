import ctypes
import cv2
import numpy as np
import os
from multiprocessing import Process, Value, Queue
import pyrealsense2.pyrealsense2 as rs
from serial.tools import list_ports
from pydobot import Dobot

# Global variables to store mouse click coordinates
mouse_click_pos = None


def mouse_callback(event, x, y, flags, param):
    global mouse_click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_click_pos = (x, y)
        print(f"Mouse clicked at: {mouse_click_pos}")

def dobot_grasp(center_p_queue, action_queue, dobot_status):
    if os.path.exists("./save_parms/image_to_arm.npy"):
        image_to_arm = np.load("./save_parms/image_to_arm.npy")
    else:
        print("image_to_arm.npy not exist")
        return

    # Initialize Dobot
    port = list_ports.comports()[0].device
    device = Dobot(port=port, verbose=True)
    device.suck(enable=False)

    # Set the initial position (x, y, z)
    initial_pos = [200, 0, 0]  # Example initial position, adjust as needed

    # Move to the initial position at the start
    print(f"Moving to initial position: {initial_pos}")
    device.move_to(initial_pos[0], initial_pos[1], initial_pos[2], 0, wait=True)

    while True:
        # Wait for signal from action_queue to move the arm or reset to the initial position
        if not action_queue.empty():
            action = action_queue.get()

            # Move to the selected position
            if action == "move":
                if not center_p_queue.empty():
                    center_p = center_p_queue.get()
                    img_pos = np.ones(4)
                    img_pos[0:3] = center_p

                    # Convert camera coordinates to robot arm coordinates
                    arm_pos = np.dot(image_to_arm, np.array(img_pos))
                    print(f"Moving to: {arm_pos}")

                    if np.sqrt(arm_pos[0] * arm_pos[0] + arm_pos[1] * arm_pos[1]) > 330:
                        print(f"Cannot reach, distance: {np.sqrt(arm_pos[0] * arm_pos[0] + arm_pos[1] * arm_pos[1])}")
                        continue

                    # Move the robot arm to the XY position first, keeping Z at a higher level
                    device.speed(100, 100)
                    print(f"Moving to XY: ({arm_pos[0]}, {arm_pos[1]}) with Z: {arm_pos[2] + 60}")
                    device.move_to(arm_pos[0], arm_pos[1], arm_pos[2] + 60, 0, wait=True)  # XY movement with higher Z

                    # After XY movement is done, lower Z to the target position
                    print(f"Moving down to Z: {arm_pos[2]}")
                    device.move_to(arm_pos[0], arm_pos[1], arm_pos[2], 0, wait=True)  # Z movement after XY is done
                    device.suck(enable=False)  # 吸泵

            # Reset to initial position if "reset" is triggered
            elif action == "reset":
                print(f"Resetting to initial position: {initial_pos}")
                device.move_to(initial_pos[0], initial_pos[1], initial_pos[2], 0, wait=True)

def realsense_video(center_p_queue, action_queue, dobot_status):
    global mouse_click_pos

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

            # 设置鼠标回调函数
            cv2.namedWindow("image")
            cv2.setMouseCallback("image", mouse_callback)

            # 获取深度图
            depth_img = np.asanyarray(depth.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=alpha_val), cv2.COLORMAP_JET
            )

            # 堆叠颜色帧和深度帧
            images = np.vstack((undistorted_image, depth_colormap))
            cv2.imshow("image", images)

            # 检查是否有鼠标点击
            if mouse_click_pos is not None:
                x, y = mouse_click_pos

                # 获取点击点的深度值
                dist_to_center = depth.get_distance(x, y)

                # 将像素坐标转换为相机坐标
                x_cam, y_cam, z_cam = rs.rs2_deproject_pixel_to_point(
                    intr, [x, y], dist_to_center
                )

                print(f"Clicked coordinates in camera space: x: {x_cam:.3f}, y: {y_cam:.3f}, z: {z_cam:.3f}")

                # 将点击的相机坐标添加到队列中
                center_p_queue.put([x_cam, y_cam, z_cam])

                # 重置鼠标点击位置
                mouse_click_pos = None

            # 检查按键退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            # 检查是否按下's'触发移动
            if key == ord("s"):
                action_queue.put("move")

            # 检查是否按下'r'重置
            if key == ord("r"):
                action_queue.put("reset")

    finally:
        pipeline.stop()

if __name__ == "__main__":
    center_p_queue = Queue()  # Queue for inter-process communication
    action_queue = Queue()  # Queue to trigger actions
    dobot_status = Value(ctypes.c_int8, 0)  # 0: stop, 1: running

    process1 = Process(target=realsense_video, args=(center_p_queue, action_queue, dobot_status))
    process2 = Process(target=dobot_grasp, args=(center_p_queue, action_queue, dobot_status))

    process1.start()
    process2.start()

    process1.join()
    process2.join()

