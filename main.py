import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from tracker import register_tracker
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
import message_filters

class TrackerNode(Node):
    def __init__(self):
        super().__init__('tracker_node')

        # Declare and get parameters
        self.declare_parameter('model_path', 'yolo11n-seg.pt')
        self.declare_parameter('tracker_config', 'botsort.yaml')
        self.declare_parameter('center_offset', 0.20)
        
        self.model_path = self.get_parameter('model_path').value
        self.tracker_config = self.get_parameter('tracker_config').value
        self.center_offset = self.get_parameter('center_offset').value

        self.get_logger().info(f"Loading YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path)

        # Register tracker
        register_tracker(self.model, persist=True)
        self.get_logger().info("YOLO model loaded and tracker registered.")
        
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
            
            # Predict and Track
            results = self.predict(cv_image)
            
            # Visualize results
            self.visualize(results, cv_depth)
            
        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")

    def predict(self, cv_image):
        """Run YOLO prediction and tracking on the image."""
        return self.model.predict(
            source=cv_image,
            show=False, 
            stream=False,
            classes=[0],
            tracker=self.tracker_config,
            verbose=False
        )

    def visualize(self, results, depth_image):
        """Visualize tracking results and calculate center points."""
        for r in results:
            annotated_frame = r.plot()
            
            if r.boxes:
                # r.boxes.xywh returns tensor: [[x_c, y_c, w, h], ...]
                boxes = r.boxes.xywh.cpu().numpy()
                masks = r.masks if r.masks is not None else None
                self._draw_center_points(annotated_frame, boxes, masks, depth_image)

            cv2.imshow("ROS2 Tracker", annotated_frame)
            
        # Wait for 1ms to display the window
        cv2.waitKey(1)

    def _draw_center_points(self, frame, boxes, masks=None, depth_image=None):
        """Calculate and draw center points on the frame."""
        for i, box in enumerate(boxes):
            x_c, y_c, w, h = box
            center_point = (int(x_c), int(y_c))

            if masks is not None and len(masks) > i:
                contour = masks.xy[i].astype(int)
                moments = cv2.moments(contour)
                if moments['m00'] != 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                    
                    if cv2.pointPolygonTest(contour, (cx, cy), False) >= 0:
                        center_point = (cx, cy)
            
            # Draw a circle at the center point
            cv2.circle(frame, center_point, 5, (0, 0, 255), -1)
            
            # Depth Calculation
            depth_text = ""
            if depth_image is not None:
                depth_m = self._calculate_depth(depth_image, center_point)
                if depth_m is not None:
                    depth_text = f" D: {depth_m:.2f}m"
            
            cv2.putText(frame, f"Center: {center_point}{depth_text}", (center_point[0] + 10, center_point[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

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
        # Cleanup
        if rclpy.ok():
            tracker_node.destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()