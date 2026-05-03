# agent_go.py
# 机械臂+大模型+多模态+语音识别=具身智能体Agent

# 导入常用函数
from asr import *             # 录音+语音识别
from camera_and_shake import *    # 连接机械臂
from llm import *             # 大语言模型API
from camera_and_shake import *    # 机械臂运动
from vlm_move import *        # 多模态大模型识别图像，吸泵吸取并移动物体
from agent import *           # 智能体Agent编排
from tts import *             # 语音合成模块

print('播放欢迎词')
pump_off()
# back_zero()
play_wav('asset/welcome.wav')


def agent_play():
    '''
    主函数，语音控制机械臂智能体编排动作
    '''
    # 归零
    # initial_pos = [200, 0, 0]  # Example initial position, adjust as needed
    back_zero(device)
    
    # 输入指令
    # 先回到原点，再把LED灯改为墨绿色，然后把绿色方块放在篮球上
    start_record_ok = input('是否开启录音，输入数字录音指定时长，按k打字输入，按c输入默认指令:\n')
    if str.isnumeric(start_record_ok):
        DURATION = int(start_record_ok)
        record(DURATION=DURATION)   # 录音
        order = speech_recognition() # 语音识别
    elif start_record_ok == 'k':
        order = input('请输入指令:\n')
    elif start_record_ok == 'c':
        order = '先归零，再点头，然后把蓝色三角片放在白色方块上'
    else:
        print('无指令，退出')
        # exit()
        raise NameError('无指令，退出')
    
    # 智能体Agent编排动作
    agent_plan_output = agent_plan(order)
    
    print('智能体编排动作如下\n', agent_plan_output)
    # plan_ok = input('是否继续？按c继续，按q退出')
    plan_ok = 'c'
    if plan_ok == 'c':
        response = agent_plan_output['response'] # 获取机器人想对我说的话
        print('开始语音合成')
        tts(response)                     # 语音合成，导出wav音频文件
        play_wav('temp/tts.wav')          # 播放语音合成音频文件
        for each in agent_plan_output['function']: # 运行智能体规划编排的每个函数
            print('开始执行动作', each)
            exec(each)  # 使用exec来执行字符串中的代码
    elif plan_ok =='q':
        # exit()
        raise NameError('按q退出')

    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()

def main():
    while True:
        agent_play()
        plan_ok = input('是否继续？按c继续，按q退出:\n')
        if plan_ok == 'q':
            print('退出所有进程并释放资源')
            break  # 退出循环

# agent_play()
if __name__ == '__main__':
    main()

