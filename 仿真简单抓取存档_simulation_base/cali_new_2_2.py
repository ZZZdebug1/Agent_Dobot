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
targetObj = sim.getObject('/target')
tipobj = sim.getObject('/Dobot/tip')
visionSensorHandle = sim.getObject('/vision_1')
deepSensorHandle = sim.getObject('/vision_1/depth')
DobotHandle = sim.getObject('/Dobot')
gripperHandle=sim.getObject('/Dobot/suctionCup_link2')


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
            arm_coords = pixel_to_world(center_x, center_y, depth_value, mtx, resX)

    # 显示更新后的RGB图像
    # img_show = np.flipud(img)  # 上下翻转
    # cv2.imshow('Detected Red Object', img_show)

    return red_object_info, arm_coords


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

def set_gripper_state(enabled):
    if enabled:
        sim.setStringSignal('activity', 'on')
    else:
        sim.setStringSignal('activity', 'off')

# 初始化状态
is_capturing_frame = False  # 用于管理当前是否处于静态帧模式
current_arm_coords = None  # 保存当前检测到的世界坐标

while True:
    if not is_capturing_frame:
        img, depth_image, resX = get_rgb_depth_images()
        # img_show = np.flipud(img)  # 上下翻转
        # cv2.imshow('Video Stream', img_show)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # 捕获当前帧并进行红色物体检测
        img, depth_image, resX = get_rgb_depth_images()
        red_object_info, arm_coords = detect_red_object(img, depth_image)
        if red_object_info:
            center_x, center_y, depth_value = red_object_info
            print(f"Detected Red Object: Pixel({center_x}, {center_y}), Depth: {depth_value}m")
            print(f"World Coordinates: {arm_coords}")
            cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), -1)  # 标记中心点
            current_arm_coords = arm_coords  # 保存当前世界坐标
        img_show = np.flipud(img)  # 上下翻转
        cv2.imshow('Detected Frame', img_show)
        is_capturing_frame = True

    elif key == ord('c') and current_arm_coords:
        # 将target物体移动到检测到的红色物体位置
        print(f"Moving target to: {current_arm_coords}")
        # 移动target对象到红色物体的中心点位置
        # sim.setObjectPosition(targetObj, DobotHandle, [current_arm_coords[0], current_arm_coords[1], current_arm_coords[2]])
        # print(f"Target object moved to world coordinates: ({world_coords[0]}, {world_coords[1]}, {world_coords[2]})")

        # 获取目标对象当前的位置
        current_position = sim.getObjectPosition(targetObj, DobotHandle)

        # 第一步：先移动 x 和 y 轴到目标位置，保持 z 轴在初始高度
        target_position_xy = [current_arm_coords[0], current_arm_coords[1], current_position[2]]
        sim.setObjectPosition(targetObj, DobotHandle, target_position_xy)
        print(f"Target object moved to: {target_position_xy}")
        # 等待1.5秒
        time.sleep(1.5)

        # 第二步：再移动 z 轴到目标高度
        target_position = [current_arm_coords[0], current_arm_coords[1], current_arm_coords[2]]
        sim.setObjectPosition(targetObj, DobotHandle, target_position)
        print(f"Target object moved to: {target_position}")

        time.sleep(1.5)
        # set_gripper_state(True)
        # time.sleep(1.5)
        # print("Gripper (suction cup) activated.")
        # activity_status = sim.getStringSignal('activity')
        # print(f"Current gripper state: {activity_status}")


        target_position = [current_arm_coords[0], current_arm_coords[1], current_position[2]]
        sim.setObjectPosition(targetObj, DobotHandle, target_position)
        time.sleep(1.5)


        target_position = [current_position[0], current_position[1], current_position[2]]
        sim.setObjectPosition(targetObj, DobotHandle, target_position)
        time.sleep(1.5)

        target_position = [current_position[0], current_position[1], current_arm_coords[2]]
        sim.setObjectPosition(targetObj, DobotHandle, target_position)
        time.sleep(1.5)

        # set_gripper_state(False)
        # print("Gripper (suction cup) inactivated.")
        # activity_status = sim.getStringSignal('activity')
        # print(f"Current gripper state: {activity_status}")

        target_position = [current_position[0], current_position[1], current_position[2]]
        sim.setObjectPosition(targetObj, DobotHandle, target_position)



    elif key == ord('q'):
        if is_capturing_frame:
            is_capturing_frame = False
            cv2.destroyWindow('Detected Frame')
        else:
            break



# 关闭所有窗口
cv2.destroyAllWindows()
# 停止仿真
sim.stopSimulation()
