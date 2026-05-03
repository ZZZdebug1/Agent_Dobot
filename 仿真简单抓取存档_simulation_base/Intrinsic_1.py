import cv2
import numpy as np
import glob

# 设置棋盘格的大小和角点数量
board_width = 7  # 棋盘格宽度
board_height = 10  # 棋盘格高度
square_size = 0.015  # 小方块的边长

# 准备对象点，如 (0,0,0), (1,0,0), (2,0,0) ..., (6,5,0)
objp = np.zeros((board_height * board_width, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2)
objp *= square_size

# 存储所有图片的对象点和图像点
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# 读取Camera_cali_2文件夹中的所有图片
images = glob.glob('Camera_cali_2/*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 寻找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (board_width, board_height), None)

    # 如果找到足够的角点，添加对象点和图像点
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 在图像上绘制并显示角点
        cv2.drawChessboardCorners(img, (board_width, board_height), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# 相机标定，计算内参和畸变系数
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 打印结果
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

# 保存结果到文件
np.savez('calibration_results.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)