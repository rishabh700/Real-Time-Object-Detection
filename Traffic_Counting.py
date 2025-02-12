from Crossing_counter import RealTimeTrafficCounter
from coco_classes import class_names

model_path = "yolo11n.pt"
video_path = r"C:\Users\risha\OneDrive\Planning Work\Planning & Design\Data Collection Framework\Github Repos\Real-Time-Traffic-Vehicle-Counting\test_data\test_video_1.mp4"
out_put_video_path = r"C:\Users\risha\OneDrive\Planning Work\Planning & Design\Data Collection Framework\Github Repos\Real-Time-Traffic-Vehicle-Counting\test_data\Output\output_test_video_1.avi"
output_csv = r"C:\Users\risha\OneDrive\Planning Work\Planning & Design\Data Collection Framework\Github Repos\Real-Time-Traffic-Vehicle-Counting\test_data\Output\CSVs\output_test_video_1.csv"
line_coords = [(0, 1800), (4000, 1800)]  # Adjust line coordinates as needed
counter = RealTimeTrafficCounter(model_path, video_path, output_csv, out_put_video_path, line_coords)
counter.process_video()
