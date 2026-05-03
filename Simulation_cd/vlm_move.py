# utils_vlm_move.py
# 输入指令，多模态大模型识别图像，吸泵吸取并移动物体

# print('神行太保：能看懂“图像”、听懂“人话”的机械臂')

from camera_and_shake import *
from asr import *
from vlm_2 import *
from PIL import Image
import time


def vlm_move(PROMPT='帮我把蓝色三角片放在白色方块上'):
    '''
    多模态大模型识别图像，吸泵吸取并移动物体
    input_way：speech语音输入，keyboard键盘输入
    '''

    print('多模态大模型识别图像，吸泵吸取并移动物体')

    # 机械臂归零
    # back_zero(device)
    # time.sleep(3)

    ## 第一步：完成手眼标定
    print('第一步：完成手眼标定')

    ## 第二步：发出指令
    print('第二步，给出的指令是：', PROMPT)

    ## 第三步：拍摄俯视图
    print('第三步：拍摄俯视图')
    img, depth_image, resX = get_rgb_depth_images()


    ## 第四步：将图片输入给多模态视觉大模型
    print('第四步：将图片输入给多模态视觉大模型')

    n = 1
    while n < 4:
        try:
            print('尝试第 {} 次访问多模态大模型'.format(n))
            result = pre_yi(PROMPT)
            print('多模态大模型调用成功！')
            print(result)
            break
        except Exception as e:
            print('多模态大模型返回数据结构错误，再尝试一次', e)
            n += 1

    ## 第五步：视觉大模型输出结果后处理和可视化
    print('第五步：视觉大模型输出结果后处理和可视化')
    START_X_CENTER, START_Y_CENTER, start_depth, END_X_CENTER, END_Y_CENTER, end_depth = post_processing_viz(result,
                                                                                                             depth_image,
                                                                                                             check=True)

    ## 第六步：坐标转换并移动机械臂
    print('第六步：坐标转换并移动机械臂')

    # 将像素坐标转换为世界坐标
    start_arm_pos = pixel_to_world(START_X_CENTER, START_Y_CENTER, start_depth, mtx, resX)
    end_arm_pos = pixel_to_world(END_X_CENTER, END_Y_CENTER, end_depth, mtx, resX)


    # 移动到起始位置
    print(f"Moving to start position: {start_arm_pos}")
    height = start_arm_pos[2] + 0.015
    print(f"目标物高度: {height}")

    # distance = np.sqrt(start_arm_pos[0] ** 2 + start_arm_pos[1] ** 2)
    # if distance >= 0.092252 or distance <= 0.01931189:
    #     print(f"Cannot reach, distance: {np.sqrt(start_arm_pos[0] ** 2 + start_arm_pos[1] ** 2)}")
    # else:

    # 第一步：先移动 x 和 y 轴到目标位置，保持 z 轴在初始高度
    target_position = [start_arm_pos[0], start_arm_pos[1], initial_pos[2]]
    sim.setObjectPosition(targetObj, DobotHandle, target_position)
    print(f"Target object moved to: {target_position}")
    # 等待1.5秒
    time.sleep(1.5)

    # 第二步：再移动 z 轴到目标高度
    target_position = [start_arm_pos[0], start_arm_pos[1], start_arm_pos[2]]
    sim.setObjectPosition(targetObj, DobotHandle, target_position)
    print(f"Target object moved to: {target_position}")

    time.sleep(1.5)

    # 开启吸盘
    sim.writeCustomStringData(gripperHandle, 'activity', 'on')
    print("Gripper activated")
    time.sleep(1)

    target_position = [start_arm_pos[0], start_arm_pos[1], initial_pos[2]]
    sim.setObjectPosition(targetObj, DobotHandle, target_position)
    # print(f"Target object moved to: {target_position}")
    time.sleep(1.5)

    # 移动到终点位置
    print(f"Moving to end position: {end_arm_pos}")

    # distance_2 = np.sqrt(end_arm_pos[0] ** 2 + end_arm_pos[1] ** 2)
    # if distance_2 >= 0.092252 or distance <= 0.01931189:
    #     print(f"Cannot reach, distance: {np.sqrt(end_arm_pos[0] ** 2 + end_arm_pos[1] ** 2)}")
    # else:
    target_position = [end_arm_pos[0], end_arm_pos[1], initial_pos[2]]
    sim.setObjectPosition(targetObj, DobotHandle, target_position)
    time.sleep(1.5)

    target_position = [end_arm_pos[0], end_arm_pos[1], end_arm_pos[2] + height]
    sim.setObjectPosition(targetObj, DobotHandle, target_position)
    time.sleep(1.5)

    # 关闭吸盘
    sim.writeCustomStringData(gripperHandle, 'activity', 'off')
    print("Gripper inactivated")
    time.sleep(1)

    target_position = [end_arm_pos[0], end_arm_pos[1], initial_pos[2]]
    sim.setObjectPosition(targetObj, DobotHandle, target_position)
    time.sleep(1.5)

    ## 第七步：机械臂归零
    print('第七步：机械臂归零')
    back_zero()
