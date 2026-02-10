import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
import os
from ultralytics import YOLO
from tracker.bot_sort_3d import BOTSORT_3D
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
import message_filters
from types import SimpleNamespace

class TrackerNode(Node):
    def __init__(self):
        super().__init__('tracker_node')

        # Declare and get parameters
        self.declare_parameter('model_path', 'yolo11n-seg.pt')
        self.declare_parameter('center_offset', 0.20)
        
        self.model_path = self.get_parameter('model_path').value
        self.center_offset = self.get_parameter('center_offset').value

        self.get_logger().info(f"Loading YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path)
        
        # Load 3D Tracker config from YAML
        config_path = os.path.join(os.path.dirname(__file__), 'tracker', 'botsort3d.yaml')
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        args = SimpleNamespace(**cfg)
        self.get_logger().info(f"Loaded 3D tracker config from {config_path}")
        
        self.tracker = BOTSORT_3D(args, frame_rate=30)
        self.get_logger().info("3D BOTSORT Tracker Initialized.")

        # Camera Intrinsics (Provided by User)
        self.fx = 394.989501953125
        self.fy = 395.01190185546875
        self.cx = 318.02618408203125
        self.cy = 199.12823486328125
        
        # ROS2 Setup
        self.bridge = CvBridge()
        
        # QoS Profile
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        
        self.rgb_sub = message_filters.Subscriber(
            self, Image, '/softbot/top_camera/rgb/image_raw', qos_profile=qos_profile)
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/softbot/top_camera/depth/image_raw', qos_profile=qos_profile)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.listener_callback)
        
        self.get_logger().info("Subscribed to RGB and Depth topics")

    def listener_callback(self, rgb_msg, depth_msg):
        try:
            # Convert ROS Image to OpenCV Image
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            
            # Predict
            results = self.model.predict(
                source=cv_image,
                show=False, 
                stream=False,
                classes=[0],
                verbose=False
            )
            
            # Prepare 3D Detections for Tracker
            detections_3d = self._prepare_detections(results[0], cv_depth)
            
            # Update Tracker
            tracked_objects = self.tracker.update(detections_3d, cv_image)
            
            # Visualize results
            self.visualize(cv_image, tracked_objects, cv_depth)
            
        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")
            import traceback
            traceback.print_exc()

    def _prepare_detections(self, result, depth_image):
        """
        Convert YOLO result + Depth to 3D Detection list.
        Each detection: {'bbox3d': [h, w, l, x, y, z, theta], 'score': float, 'cls': int}
        """
        detections = []
        
        if result.boxes:
            boxes = result.boxes.xywh.cpu().numpy() # x_c, y_c, w, h
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for i, box in enumerate(boxes):
                x_c, y_c, w, h = box
                score = scores[i]
                cls = int(classes[i])
                
                # Get Depth
                depth_m = self._calculate_depth(depth_image, (int(x_c), int(y_c)))
                
                if depth_m is not None and depth_m > 0:
                    # Project to 3D (Camera Coordinates)
                    # Z = depth
                    # X = (u - cx) * Z / fx
                    # Y = (v - cy) * Z / fy
                    z = depth_m
                    x = (x_c - self.cx) * z / self.fx
                    y = (y_c - self.cy) * z / self.fy
                    
                    # Estimate Dimensions (Approximation for Person)
                    # Height in pixels is h. H_real = h * Z / fy
                    # Width in pixels is w. W_real = w * Z / fx
                    h_real = h * z / self.fy
                    w_real = w * z / self.fx
                    l_real = 0.3 # Assumed thickness for person (approx 30cm)
                    
                    # Shift center behind the surface (depth measures front face)
                    z += l_real / 2
                    
                    theta = 0
                    
                    # KF state format: [x, y, z, theta, l, w, h]
                    bbox3d = np.array([x, y, z, theta, l_real, w_real, h_real])
                    
                    detections.append({
                        'bbox3d': bbox3d,
                        'score': score,
                        'cls': cls,
                        'bbox_2d': (x_c, y_c, w, h),  # For ReID crop extraction
                        'feat': None 
                    })
                    
        return detections

    def _get_color(self, track_id):
        """Generate a unique color for each track ID."""
        np.random.seed(track_id)
        color = np.random.randint(0, 255, size=3).tolist()
        return tuple(color)


    def visualize(self, frame, tracked_objects, depth_image):
        if frame is None:
            return

        for track in tracked_objects:
            # KF State: x, y, z, theta, l, w, h, id, score, cls, vx, vy, vz
            res = track.result
            x, y, z, theta, l, w, h = res[:7]
            track_id = int(res[7])
            vx, vy, vz = map(float, res[10:13])
            
            color = self._get_color(track_id)
            
            # Project back to 2D for drawing
            if z > 0:
                # Draw History Trail
                history = track.get_history()
                for i in range(1, len(history)):
                    pt1_3d = history[i-1]
                    pt2_3d = history[i]
                    
                    if pt1_3d[2] > 0 and pt2_3d[2] > 0:
                        u1 = int(pt1_3d[0] * self.fx / pt1_3d[2] + self.cx)
                        v1 = int(pt1_3d[1] * self.fy / pt1_3d[2] + self.cy)
                        u2 = int(pt2_3d[0] * self.fx / pt2_3d[2] + self.cx)
                        v2 = int(pt2_3d[1] * self.fy / pt2_3d[2] + self.cy)
                        
                        cv2.line(frame, (u1, v1), (u2, v2), color, 2)

                # Draw 3D Bounding Box
                self._draw_3d_box(frame, x, y, z, l, w, h, theta, track_id, color)

                # Draw Text and Circle at Front Face (Surface)
                z_front = z - l / 2
                if z_front > 0:
                    u = int(x * self.fx / z_front + self.cx)
                    v = int(y * self.fy / z_front + self.cy)
                    
                    # Draw Circle
                    cv2.circle(frame, (u, v), 5, color, -1)
                    
                    vel_text = f"V: {vx:.1f},{vy:.1f},{vz:.1f}"
                    text = f"ID:{track_id} Z:{z_front:.2f}m"
                    cv2.putText(frame, text, (u + 10, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(frame, vel_text, (u + 10, v + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("3D Tracker", frame)
        cv2.waitKey(1)

    def _draw_3d_box(self, frame, x, y, z, l, w, h, theta, track_id, color):
        # ... logic ...
        
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        
        x_corners = [w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2]
        y_corners = [h/2, -h/2, -h/2, h/2, h/2, -h/2, -h/2, h/2]
        z_corners = [l/2, l/2, l/2, l/2, -l/2, -l/2, -l/2, -l/2]
        
        corners_3d = np.vstack([x_corners, y_corners, z_corners]) # (3, 8)
        
        # Translate
        corners_3d[0, :] = corners_3d[0, :] + x
        corners_3d[1, :] = corners_3d[1, :] + y
        corners_3d[2, :] = corners_3d[2, :] + z
        
        # Project to 2D
        corners_2d = []
        for i in range(8):
            cx, cy, cz = corners_3d[:, i]
            if cz > 0:
                u = int(cx * self.fx / cz + self.cx)
                v = int(cy * self.fy / cz + self.cy)
                corners_2d.append((u, v))
            else:
                return # Clip
                
        if len(corners_2d) == 8:
            lines = [
                (0, 1), (1, 2), (2, 3), (3, 0), # Front face
                (4, 5), (5, 6), (6, 7), (7, 4), # Back face
                (0, 4), (1, 5), (2, 6), (3, 7)  # Connecting edges
            ]
            
            for p1, p2 in lines:
                cv2.line(frame, corners_2d[p1], corners_2d[p2], color, 2)
            
            # Draw ID at center
            # center_u = int(sum([p[0] for p in corners_2d]) / 8)
            # center_v = int(sum([p[1] for p in corners_2d]) / 8)
            # cv2.putText(frame, f"ID:{track_id}", (center_u, center_v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _calculate_depth(self, depth_image, center_point, kernel_size=2):
        """Calculate the median depth around a center point."""
        cx, cy = center_point
        h_d, w_d = depth_image.shape
        
        # Ensure ROI is within bounds
        x_min = max(0, cx - kernel_size)
        x_max = min(w_d, cx + kernel_size + 1)
        y_min = max(0, cy - kernel_size)
        y_max = min(h_d, cy + kernel_size + 1)
        
        roi = depth_image[y_min:y_max, x_min:x_max]
        
        # Filter out 0 (invalid depth) and take median
        valid_pixels = roi[roi > 0]
        if valid_pixels.size > 0:
            depth_mm = np.median(valid_pixels)
            return depth_mm / 1000.0
        
        return None

    def _prepare_detections(self, result, depth_image):
        # Redefined to match KF state order [x, y, z, theta, l, w, h]
        detections = []
        if result.boxes:
            boxes = result.boxes.xywh.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for i, box in enumerate(boxes):
                x_c, y_c, w, h = box
                score = scores[i]
                cls = int(classes[i])
                
                depth_m = self._calculate_depth(depth_image, (int(x_c), int(y_c)))
                
                if depth_m is not None and depth_m > 0:
                    z = depth_m
                    x = (x_c - self.cx) * z / self.fx
                    y = (y_c - self.cy) * z / self.fy
                    
                    h_real = h * z / self.fy
                    w_real = w * z / self.fx
                    l_real = 0.5 
                    theta = 0 
                    
                    # KF State Order: [x, y, z, theta, l, w, h]
                    bbox3d = np.array([x, y, z, theta, l_real, w_real, h_real])
                    
                    detections.append({
                        'bbox3d': bbox3d,
                        'score': score,
                        'cls': cls
                    })
        return detections

def main(args=None):
    rclpy.init(args=args)
    tracker_node = TrackerNode()
    try:
        rclpy.spin(tracker_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Runtime error: {e}")
    finally:
        if rclpy.ok():
            tracker_node.destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()