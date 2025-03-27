# 🤖 基于大模型的智能体机械臂抓取实验设计

<img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python"> <img src="https://img.shields.io/badge/Simulator-Coppeliasim%204.8 EDU-orange" alt="Coppeliasim"> <img src="https://img.shields.io/badge/Framework-PyQt5-green" alt="PyQt5">

## 🌟 项目概览
面向大模型下的机械臂控制教学实验设计，包含**实物控制**、**仿真平台**和**交互界面**三大模块。
通过多种类大模型实现自然语言理解、目标检测和动作编排，支持不同部署模式的大模型灵活切换，创造出一个简单的具身智能体。

### 🛠️ 硬件配置
- **机械臂**: Dobot Magician 桌面级
- **末端执行器**: 真空吸泵模块
- **视觉系统**: Intel Realsense D435i RGB-D相机

### 🧠 核心模型
| 模型类型                | 部署方式       | 具体模型                          |
|-------------------------|----------------|-----------------------------------|
| 语言理解模型            | 本地部署       | Deepseek-R1-8b                    |
| 目标检测模型            | 本地部署       | IDEA-Grounding-DINO-Base          |
| 语言理解模型            | 云端API        | 零一万物Yi-Vision-spark           |
| 多模态模型              | 云端API        | 阿里Qwen-Max-2024-11-19           |

## 🔍 系统原理
![系统架构图](https://github.com/user-attachments/assets/f359bd83-0fe2-48b7-aded-279b89cb3d72)

1. **语音转文本**：语音模型处理音频转文本输入
2. **指令解析**：结合系统提示词构建Agent智能体
3. **动作编排**：大语言模型生成动作序列与自然语言响应
4. **视觉定位**：零样本目标检测+坐标转换（相机坐标系→机械臂基坐标系）
5. **执行验证**：路径规划与末端执行器控制
6. **语音反馈**：文本转语音交互

## 🎥 演示与解读

[![B站演示](https://img.shields.io/badge/Bilibili-演示视频-00A1D6)](https://www.bilibili.com/video/BV1mdZgYtEcV/?share_source=copy_web&vd_source=c2602f17e0f83fa76a014d113eea745b&t=0) 

https://www.bilibili.com/video/BV1mdZgYtEcV/?share_source=copy_web&vd_source=c2602f17e0f83fa76a014d113eea745b&t=0

[![CSDN技术解读](https://img.shields.io/badge/CSDN-技术解析-FF0000)](https://blog.csdn.net/qq_57553723/article/details/144115368?spm=1001.2014.3001.5502)


https://blog.csdn.net/qq_57553723/article/details/144115368?spm=1001.2014.3001.5502

## 🖥️ 软件界面

界面如图示，使用下拉栏选择好模型，输入指令后再点击“开始实验”启动程序；

过程中日志会打印当前步骤，RGB和Depth下会输出RGB和深度图，截取当前帧和可视化后的抓取点图片会通过弹窗交由用户确认；

完成一次指令后可重新输入指令以及选择模型，点击“继续执行”即可实现更换模型连续动作；（目前只做了与仿真通信）

![GUI界面](https://github.com/user-attachments/assets/fd5b5dd4-f71a-4f7d-b365-2806e9006841)

### 核心功能
- **多模型热切换**：本地/云端模型即时切换
- **三维可视化**：
  - RGB图像实时渲染
  - 深度图像显示
  - 抓取点位可视化标注
- **控制面板**：
  - 🚀 一键初始化设备连接
  - 🔄 模型切换无缝衔接
  - 🛑 紧急制动安全机制

> **操作规范**  
> - 操作结束后须执行紧急停止流程，避免直接关闭界面可能出现的线程卡死问题  
> - 云端模型需要预先配置API密钥  
> - 深度图解析需保持相机在线状态  

## 📜 开发说明
### 仿真环境
- **仿真平台**: CoppeliaSim EDU 4.8 EDU
- **通信接口**: ZMQ Remote API
- **机械臂模型**: Dobot Magician 桌面级模型


## ⚠️ 安全须知
1. 实物操作前必须完成碰撞检测
2. 坐标系转换需每周进行标定校准
3. 云端模型响应延迟可能达到500-800ms
4. 紧急停止将立即切断所有通信线程
