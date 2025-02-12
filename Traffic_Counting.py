from Crossing_counter import RealTimeTrafficCounter
from coco_classes import class_names

#Define input parameters
#confidence_threshold = 0.3 #Filters the detections with lower confidence values to remove false positives
#allowed_classes = {"car", "person", "bus", "train", "truck", "bicycle", "motorcycle"} # Define classes of interest
#max_age = 1 # Maximum number of frames to keep the object alive with no association
#min_hits = 3 # Minimum number of associated detections to create a track
#iou_threshold = 0.3 # Threshold for Intersection over Union (IoU) for matching detections to tracks
line_coords = [(0, 1800), (4000, 1800)]  # Adjust line coordinates as needed

#Output Files for Stages
# Detection Stage = 'Yes' or 'No'
# Tracking Stage = 'Yes' or 'No'
# Counting Stage = 'Yes' or 'No' 

#Define File Paths
model_path = "yolo11n.pt" # Auto-downloaded if not in the repository
video_path = r"C:\Users\risha\OneDrive\Planning Work\Planning & Design\Data Collection Framework\Github Repos\Real-Time-Traffic-Vehicle-Counting\test_data\test_video_1.mp4"
out_put_video_path = r"C:\Users\risha\OneDrive\Planning Work\Planning & Design\Data Collection Framework\Github Repos\Real-Time-Traffic-Vehicle-Counting\test_data\Output\output_test_video_1.avi"
output_csv = r"C:\Users\risha\OneDrive\Planning Work\Planning & Design\Data Collection Framework\Github Repos\Real-Time-Traffic-Vehicle-Counting\test_data\Output\CSVs\output_test_video_1.csv"
counter = RealTimeTrafficCounter(model_path, video_path, output_csv, out_put_video_path, line_coords)
counter.process_video()
