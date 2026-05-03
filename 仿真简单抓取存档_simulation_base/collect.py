import sim
import time
import sys
import cv2
import numpy as np
import os

# 确保Camera_cali文件夹存在
if not os.path.exists('Camera_cali_2'):
    os.makedirs('Camera_cali_2')

# 关闭之前的连接
sim.simxFinish(-1)

# 获得客户端ID
clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
print("Connection success!!!")

if clientID != -1:
    print('Connected to remote API server')
else:
    print('Connection not successful')
    sys.exit('Could not connect')

# 启动仿真
sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
print("Simulation start")

# 使能同步模式
sim.simxSynchronous(clientID, False)

# 获得对象的句柄
ret, targetObj = sim.simxGetObjectHandle(clientID, 'target', sim.simx_opmode_blocking)
errorCode, visionSensorHandle = sim.simxGetObjectHandle(clientID, 'vision_1', sim.simx_opmode_oneshot_wait)
# 获得深度传感器的句柄
errorCode, depthSensorHandle = sim.simxGetObjectHandle(clientID, './depth', sim.simx_opmode_oneshot_wait)
if errorCode != sim.simx_return_ok:
    print("Error: Failed to get depth sensor handle")
    sys.exit("Could not get depth sensor handle")

# last_print_time = time.time()  # 初始化上一次打印时间

# 启动视觉传感器的图像流
errorCode, resolution, rawimage = sim.simxGetVisionSensorImage(clientID, visionSensorHandle, 0, sim.simx_opmode_streaming)

def readVisionSensor():
    global resolution
    # 获取RGB图像
    errorCode, resolution, rawimage = sim.simxGetVisionSensorImage(clientID, visionSensorHandle, 0,
                                                                   sim.simx_opmode_buffer)
    # 打印分辨率以检查问题
    # print(f"Resolution: {resolution}")

    if errorCode != sim.simx_return_ok:
        print("Error: Failed to get image data")
        return None
    sensorImage = np.array(rawimage).astype(np.uint8)  # 将原始图像转换为 uint8 类型
    sensorImage = sensorImage.astype(np.uint8)  # 将超出范围的值溢出

    if len(resolution) < 2:
        print("Error: Resolution data is incomplete!")
        return None

    sensorImage.resize([resolution[1], resolution[0], 3])  # 将图像调整为 (height, width, 3) 的格式
    sensorImage = cv2.cvtColor(sensorImage, cv2.COLOR_RGB2BGR)  # 将 RGB 转换为 BGR
    cv2.flip(sensorImage, 0, sensorImage)  # 图像上下翻转
    return sensorImage

frame_number = 0  # 初始化frame_number
while True:
    # 获取目标位置
    ret, arr = sim.simxGetObjectPosition(clientID, targetObj, -1, sim.simx_opmode_blocking)

    # 获取并显示图像
    image = readVisionSensor()

    cv2.imshow("Image", image)

    key = cv2.waitKey(1) & 0xFF  # 获取按键值

    # 如果按下 'q' 键，则退出循环
    if key == ord('q'):
        break
    # 如果按下 'c' 键，则保存图像
    elif key == ord('c'):
        if image is not None:
            cv2.imwrite(f'Camera_cali_2/frame_{frame_number}.png', image)
            print(f"Frame saved as Camera_cali_2/frame_{frame_number}.png")
            frame_number += 1  # 增加frame_number以便下次保存时编号递增

    # 打印目标位置
    # if ret == sim.simx_return_ok:
    #     print(f"Target position: {arr}")

# 释放资源和关闭窗口
cv2.destroyAllWindows()