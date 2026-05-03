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
visionSensorHandle = sim.getObject('/vision_1')
deepSensorHandle = sim.getObject('/vision_1/depth')


# 获取RGB和深度图像
def get_rgb_depth_images():
    # 获取RGB图像
    img, [resX, resY] = sim.getVisionSensorImg(visionSensorHandle)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
    img = np.flipud(img)  # 上下翻转
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 转换为BGR格式

    # 获取深度图像
    Deepdate = sim.getVisionSensorDepth(deepSensorHandle, 1)
    num_floats = Deepdate[1][0] * Deepdate[1][1]
    depth_data = struct.unpack(f'{num_floats}f', Deepdate[0])
    depth_array = np.array(depth_data)
    depth_image = depth_array.reshape((Deepdate[1][1], Deepdate[1][0]))
    depth_image = np.flipud(depth_image)

    # 深度图像着色
    deepcolor = encoding.FloatArrayToRgbImage(depth_image, scale_factor=256 * 3.5 * 1000)

    # 显示RGB图像和深度图像
    cv2.imshow('RGB Image', img)
    cv2.imshow('Depth Image', depth_image)

    return img, depth_image


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
            print(f"Red object center at: ({center_x}, {center_y})")

            # 获取深度图像中的深度值
            depth_value = depth_image[center_y, center_x]
            print(f"Depth at ({center_x}, {center_y}): {depth_value} meters")

            # 可选：将检测到的红色物块轮廓画出来
            cv2.drawContours(img, [largest_contour], -1, (0, 255, 0), 1)

            # 返回中心点坐标和深度值
            red_object_info = (center_x, center_y, depth_value)

    # 显示更新后的RGB图像
    cv2.imshow('Detected Red Object', img)

    return red_object_info


# 主程序调用
while True:
    # 获取图像和深度信息
    img, depth_image = get_rgb_depth_images()

    # 检测红色物体并返回中心点和深度信息
    red_object_info = detect_red_object(img, depth_image)
    if red_object_info:
        center_x, center_y, depth_value = red_object_info
        print(f"Detected Red Object at ({center_x}, {center_y}) with depth: {depth_value} meters")

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 关闭所有窗口
cv2.destroyAllWindows()
