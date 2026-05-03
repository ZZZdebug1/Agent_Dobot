import sim
import sys

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

# 创建虚拟对象
position = [0.33062880204109807, 0.08167223584219344, 0.035003561973583674]
color = [0, 255, 0]  # 绿色
ret, dummy_handle = sim.simxCreateDummy(clientID, 0.02, color, sim.simx_opmode_blocking)
if ret == sim.simx_return_ok:
    print(f"Dummy created with handle: {dummy_handle}")
else:    print("Failed to create dummy")

# 设置虚拟对象的位置
ret = sim.simxSetObjectPosition(clientID, dummy_handle, -1, position, sim.simx_opmode_blocking)
if ret == sim.simx_return_ok:
    print("Dummy position set successfully")
else:
    print("Failed to set dummy position")

# 继续您的仿真操作...

# 停止仿真（如果需要）
# sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)

# 关闭连接（如果需要）
# sim.simxFinish(clientID)