import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
from tracker import register_tracker
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

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
        
        self.subscription = self.create_subscription(
            Image,
            '/softbot/top_camera/rgb/image_raw',
            self.listener_callback,
            qos_profile)
        self.get_logger().info("Subscribed to /softbot/top_camera/rgb/image_raw")

    def listener_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV Image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Predict and Track
            results = self.predict(cv_image)
            
            # Visualize results
            self.visualize(results)
            
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

    def visualize(self, results):
        """Visualize tracking results and calculate center points."""
        for r in results:
            annotated_frame = r.plot()
            
            if r.boxes:
                # r.boxes.xywh returns tensor: [[x_c, y_c, w, h], ...]
                boxes = r.boxes.xywh.cpu().numpy()
                self._draw_center_points(annotated_frame, boxes)

            cv2.imshow("ROS2 Tracker", annotated_frame)
            
        # Wait for 1ms to display the window
        cv2.waitKey(1)

    def _draw_center_points(self, frame, boxes):
        """Calculate and draw center points on the frame."""
        for box in boxes:
            x_c, y_c, w, h = box
            # Offset center point a little up based on parameter
            center_point = (int(x_c), int(y_c - h * self.center_offset))
            
            # Draw a circle at the center point
            cv2.circle(frame, center_point, 5, (0, 0, 255), -1)
            # Optional: Put text text
            cv2.putText(frame, f"Center: {center_point}", (center_point[0] + 10, center_point[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

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