from serial.tools import list_ports
import pydobot

# 搜索Dobot端口
available_ports = list_ports.comports()
print(f'available ports: {[x.device for x in available_ports]}')
port = available_ports[0].device

# 创建Dobot实例
device = pydobot.Dobot(port=port, verbose=True)

# 获取当前姿态
(x, y, z, r, j1, j2, j3, j4) = device.pose()
print(f'x:{x} y:{y} z:{z} j1:{j1} j2:{j2} j3:{j3} j4:{j4}')

# 定义移动参数
move_distance = 20  # 移动距离
lift_height = 20     # 提升高度
suck_distance = 10   # 吸取下降距离

# 第一阶段：移动到目标点上方
device.move_to(x + move_distance, y, z, r, wait=True)

# 第二阶段：下降
device.move_to(x + move_distance, y, z - lift_height, r, wait=True)

# 第三阶段：吸取
# device.suck(True)  # 启用吸取功能

# 第四阶段：上升
device.move_to(x + move_distance, y, z - lift_height + suck_distance, r, wait=True)

# 第五阶段：移动到释放点上方
device.move_to(x, y, z - lift_height + suck_distance, r, wait=True)

# 第六阶段：下降到释放点
device.move_to(x, y, z, r, wait=True)

# 第七阶段：松开
# device.suck(False)  # 禁用吸取功能

# 关闭Dobot连接
device.close()