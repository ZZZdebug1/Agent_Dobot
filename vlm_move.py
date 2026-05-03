# utils_vlm_move.py
# 同济子豪兄 2024-5-22
# 输入指令，多模态大模型识别图像，吸泵吸取并移动物体

# print('神行太保：能看懂“图像”、听懂“人话”的机械臂')

from camera_and_shake import *
from asr import *
from vlm import *

import time


def vlm_move(PROMPT='帮我把蓝色三角片放在白色方块上', device=device):
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
    depth_frame, intr = realsense_video()  # 调用函数并获取返回值

    ## 第四步：将图片输入给多模态视觉大模型
    print('第四步：将图片输入给多模态视觉大模型')


    n = 1
    while n < 5:
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
    x_cam_start, y_cam_start, z_cam_start, x_cam_end, y_cam_end, z_cam_end = post_processing_viz(result,
                                                                                                 depth_frame, intr,
                                                                                                 check=True)

    ## 第六步：坐标转换并移动机械臂
    print('第六步：坐标转换并移动机械臂')
    if os.path.exists("./save_parms/image_to_arm.npy"):
        image_to_arm = np.load("./save_parms/image_to_arm.npy")
    else:
        print("image_to_arm.npy not exist")
        return

    # 使用传入的相机坐标
    camera_start_pos = np.array([x_cam_start, y_cam_start, z_cam_start, 1])  # 添加齐次坐标
    camera_end_pos = np.array([x_cam_end, y_cam_end, z_cam_end, 1])  # 添加齐次坐标

    # 将相机坐标转换为机械臂坐标
    start_arm_pos = np.dot(image_to_arm, camera_start_pos.T)  # 注意转置
    end_arm_pos = np.dot(image_to_arm, camera_end_pos.T)  # 注意转置

    # 移动到起始位置
    print(f"Moving to start position: {start_arm_pos}")
    if np.sqrt(start_arm_pos[0] ** 2 + start_arm_pos[1] ** 2) > 330:
        print(f"Cannot reach, distance: {np.sqrt(start_arm_pos[0] ** 2 + start_arm_pos[1] ** 2)}")
    else:
        device.speed(100, 100)
        device.move_to(start_arm_pos[0], start_arm_pos[1], start_arm_pos[2] + 60, 0,
                       wait=True)  # XY movement with higher Z
        device.move_to(start_arm_pos[0], start_arm_pos[1], start_arm_pos[2], 0,
                       wait=True)  # Z movement after XY is done
        # device.suck(enable=True)  # Engage suction cup
        time.sleep(1)
        device.move_to(start_arm_pos[0], start_arm_pos[1], start_arm_pos[2] + 60, 0,
                       wait=True)  # 过渡

    # 移动到终点位置
    print(f"Moving to end position: {end_arm_pos}")
    if np.sqrt(end_arm_pos[0] ** 2 + end_arm_pos[1] ** 2) > 330:
        print(f"Cannot reach, distance: {np.sqrt(end_arm_pos[0] ** 2 + end_arm_pos[1] ** 2)}")
    else:
        device.speed(100, 100)
        device.move_to(end_arm_pos[0], end_arm_pos[1], end_arm_pos[2] + 60, 0, wait=True)  # XY movement with higher Z
        device.move_to(end_arm_pos[0], end_arm_pos[1], end_arm_pos[2], 0, wait=True)  # Z movement after XY is done
        # device.suck(enable=False)  # Release suction cup
        time.sleep(1)
        device.move_to(end_arm_pos[0], end_arm_pos[1], end_arm_pos[2] + 60, 0, wait=True)  # 过渡


    ## 第七步：机械臂归零
    print('第七步：机械臂归零')
    if device is not None:
        back_zero(device)
