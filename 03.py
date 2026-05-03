import cv2
import numpy as np
import pyrealsense2.pyrealsense2 as rs
import cv2.aruco as aruco

class RealSenseAruco:
    def __init__(self):
        # Initialize Realsense
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)
        self.pipeline.start(config)

        # Aruco parameters
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
        self.parameters = aruco.DetectorParameters()

        # Camera intrinsic parameters
        self.color_frame = None
        self.depth_frame = None
        self.intr = None
        self.intr_matrix = None
        self.intr_coeffs = None

    def get_camera_parameters(self):
        self.color_frame = self.color_frame.profile.as_video_stream_profile().intrinsics
        self.intr_matrix = np.array([
            [self.color_frame.fx, 0, self.color_frame.ppx],
            [0, self.color_frame.fy, self.color_frame.ppy],
            [0, 0, 1]
        ], dtype=np.float32)
        self.intr_coeffs = np.array(self.color_frame.coeffs, dtype=np.float32)

    def get_aruco_center(self, calib=True):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)

        # Get depth frame
        self.depth_frame = frames.get_depth_frame()

        # Display color frame
        self.color_frame = frames.get_color_frame()
        color_image = np.asanyarray(self.color_frame.get_data())

        # Undistort color image
        undistorted_image = cv2.undistort(color_image, self.intr_matrix, self.intr_coeffs)

        # Detect Aruco markers
        corners, ids, rejected_img_points = aruco.detectMarkers(undistorted_image, self.dictionary, parameters=self.parameters)

        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners, 0.05, self.intr_matrix, self.intr_coeffs)

        center = None
        if ids is not None:
            aruco.drawDetectedMarkers(undistorted_image, corners)
            cv2.drawFrameAxes(undistorted_image, self.intr_matrix, self.intr_coeffs, rvec, tvec, 0.05)

            for i, corner in zip(ids, corners):
                if calib:
                    x = (corner[0][0][0] + corner[0][3][0]) / 2
                    y = (corner[0][0][1] + corner[0][3][1]) / 2
                else:
                    x = (corner[0][0] + corner[0][2]) / 2
                    y = (corner[0][1] + corner[0][3]) / 2

                cv2.circle(undistorted_image, (int(x), int(y)), 3, (0, 0, 255), -1)

                dist_to_center = self.depth_frame.get_distance(int(x), int(y))
                x_cam, y_cam, z_cam = rs.rs2_deproject_pixel_to_point(self.color_frame.profile.as_video_stream_profile().intrinsics, [x, y], dist_to_center)

                cv2.putText(undistorted_image, "x: {:.3f}m".format(x_cam), (int(x) + 50, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(undistorted_image, "y: {:.3f}m".format(y_cam), (int(x) + 50, int(y) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(undistorted_image, "z: {:.3f}m".format(z_cam), (int(x) + 50, int(y) + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                center = [x_cam, y_cam, z_cam]

                break

        # Get depth image
        depth_img = np.asanyarray(self.depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.219), cv2.COLORMAP_JET)

        # Stack color frame and depth frame
        images = np.hstack((undistorted_image, depth_colormap))
        cv2.imshow("image", images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.pipeline.stop()
            cv2.destroyAllWindows()

    def run(self):
        try:
            self.get_camera_parameters()
            while True:
                self.get_aruco_center()
        except Exception as e:
            print(e)
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    rsa = RealSenseAruco()
    rsa.run()