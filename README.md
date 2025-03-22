# 简介
多模态大模型仿真+实物代码展示 

整体代码分为实物、仿真和软件界面三部分

仅为教学练习使用

实物采用设备：dobot桌面机械臂+吸泵+Realsense D435i相机

仿真软件为Coppeliasim EDU 4.8，采用ZMQremote API远程通信

仿真界面由PyQT5自行开发，做工粗糙仅练习使用

主要使用的大模型包括有：

Deepseek-R1-8b （本地化部署）

IDEA-Grounding-DINO-Base (本地化部署)

零一万物Yi-Vision-spark (远程API)

阿里Qwen-Max-2024-11-19 (远程API)


# 原理
实验的最终目标是向终端给出非固定的文本或语音指令后，机械臂可以按要求完成相应基础动作和目标抓取。

整体架构为：通过语音模型完成对语音指令的文本化操作；文本指令被送入提示词工程中，与预设的系统提示词组合传入大语言模型，在此期间大语言模型将基于系统提示词成为特定的“Agent智能体”。该智能体会理解、编排并返回文本指令中包含的动作，同时生成一句回复词，经处理后依次执行机械臂动作。其中动作若为目标抓取，则需再次调用大语言模型进行关系抽取，使用零样本目标检测模型找出抓取点始末位置，经坐标转换处理后传入机械臂执行相关抓取动作，过程中应注意始末点有效性；回复词文本合成为音频播放，丰富交互感和体验感。

![小论文-新 - visual selection](https://github.com/user-attachments/assets/f359bd83-0fe2-48b7-aded-279b89cb3d72)


# 相关视频及解读
B站：https://www.bilibili.com/video/BV1KKXzYoEik/?spm_id_from=333.1387.homepage.video_card.click&vd_source=754cd60572074e9c3818710bcb5bf283

CSDN:
# 实物

# 仿真

# 软件
界面如图示，使用下拉栏选择好模型，输入指令后再点击开始实验启动程序；

过程中日志会打印当前步骤，RGB和Depth下会输出RGB和深度图，截取当前帧和可视化后的抓取点图片会通过弹窗交由用户确认；

完成一次指令后可重新输入指令以及选择模型继续实验

![d7ee027a962da8c9139caf01851328fb](https://github.com/user-attachments/assets/fd5b5dd4-f71a-4f7d-b365-2806e9006841)
