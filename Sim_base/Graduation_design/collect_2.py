from src.coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import sys
import cv2
import numpy as np
import struct
import depth_image_encoding as encoding

client = RemoteAPIClient()
sim = client.require('sim')

# 启动仿真
sim.startSimulation()
print("Simulation start")

visionSensorHandle = sim.getObject('/vision_1')

# 获取RGB和深度图像
def get_rgb_depth_images():

    img, [resX, resY] = sim.getVisionSensorImg(visionSensorHandle)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
    # img = np.flipud(img)  # 上下翻转
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 转换为BGR格式

    # 显示RGB图像和深度图像
    img_show = np.flipud(img)  # 上下翻转
    # cv2.imshow('RGB Image', img_show)

    return img_show

frame_number = 0  # 初始化frame_number
while True:
    img = get_rgb_depth_images()
    cv2.imshow("Image", img)
    # 处理按键事件
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if img is not None:
            cv2.imwrite(f'Camera_cali_2/frame_{frame_number}.png', img)
            print(f"Frame saved as Camera_cali_2/frame_{frame_number}.png")
            frame_number += 1  # 增加frame_number以便下次保存时编号递增
    elif key == ord('q'):
        cv2.destroyAllWindows()
        break  # 退出循环