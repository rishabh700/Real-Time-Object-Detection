import cv2
import csv
import numpy as np
import torch
from ultralytics import YOLO
from sort import Sort
from coco_classes import class_names


class RealTimeTrafficCounter:
    def __init__(self, model_path, video_path, output_csv, output_video, line_coords):
        self.model_path = model_path
        self.video_path = video_path
        self.output_csv = output_csv
        self.output_video = output_video
        self.line_coords = line_coords
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load YOLO model
        self.model = YOLO(self.model_path).to(self.device)

        # Initialize SORT tracker
        self.tracker = Sort()

        # Define classes of interest
        self.allowed_classes = {"car", "person", "bus", "train", "truck", "bicycle", "motorcycle"}

        # Counting variables
        self.crossed_objects = {}
        self.counts = {class_name: {"up": 0, "down": 0} for class_name in self.allowed_classes}

        # Prepare CSV file
        with open(self.output_csv, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Frame_No", "ObjID", "x1", "y1", "x2", "y2", "Class_Name", "Confidence", "Direction", "Aggregate_Count"])

        self.writer = None

    def process_frame(self, frame, frame_number, fps):
        results = self.model(frame)
        detections = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()  # Extract confidence values
        classes = results[0].boxes.cls.cpu().numpy()

        dets = np.hstack((detections[:, :4], confidences.reshape(-1, 1)))  # Correctly format input for SORT
        tracks = self.tracker.update(dets)
        frame_data = []

        for track in tracks:
            x1, y1, x2, y2, obj_id = map(int, track[:5])

            # Find corresponding confidence value
            best_match_idx = -1
            best_iou = 0
            for i, det in enumerate(detections):
                iou = self._iou(track[:4], det[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = i

            if best_match_idx != -1:
                class_id = int(classes[best_match_idx])
                class_name = class_names[class_id] if class_id in range(len(class_names)) else "Unknown"
                confidence = round(float(confidences[best_match_idx]), 2)  # Extract correct confidence
            else:
                class_name = "Unknown"
                confidence = 0.0  # Default if not matched

            if class_name not in self.allowed_classes:
                continue  # Skip unwanted classes

            center_y = (y1 + y2) // 2
            line_y = self.line_coords[0][1]
            direction = None

            if obj_id not in self.crossed_objects:
                self.crossed_objects[obj_id] = center_y
            else:
                if self.crossed_objects[obj_id] < line_y <= center_y:
                    direction = "down"
                elif self.crossed_objects[obj_id] > line_y >= center_y:
                    direction = "up"

                if direction:
                    self.counts[class_name][direction] += 1
                    self.crossed_objects[obj_id] = center_y  # Update position

            # Save frame data to CSV
            frame_data.append([
                frame_number, obj_id, x1, y1, x2, y2, class_name, confidence,
                direction if direction else "N/A",
                sum(sum(count.values()) for count in self.counts.values())
            ])

            # Draw bounding box
            color = (0, 255, 0) if direction == "up" else (0, 0, 255) if direction == "down" else (255, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name} ID:{obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw counting line
        cv2.line(frame, self.line_coords[0], self.line_coords[1], (0, 255, 255), 3)

        # Display counts at the top
        info_text = [f"FPS: {fps:.2f}"]
        for key, counts in self.counts.items():
            info_text.append(f"{key}: Up-{counts['up']} | Down-{counts['down']}")

        for idx, line in enumerate(info_text):
            cv2.putText(frame, line, (10, 20 + (idx * 20)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Save data to CSV
        with open(self.output_csv, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(frame_data)

        # Write frame to output video
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            height, width = frame.shape[:2]
            self.writer = cv2.VideoWriter(self.output_video, fourcc, 30, (width, height), True)

        self.writer.write(frame)
        return frame

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_number = 0
        prev_time = cv2.getTickCount()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            curr_time = cv2.getTickCount()
            time_elapsed = (curr_time - prev_time) / cv2.getTickFrequency()
            fps = 1 / time_elapsed if time_elapsed > 0 else 0
            prev_time = curr_time

            processed_frame = self.process_frame(frame, frame_number, fps)
            #cv2.imshow("Real-Time Traffic Tracking", processed_frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

        cap.release()
        cv2.destroyAllWindows()

        # Release the video writer
        if self.writer is not None:
            self.writer.release()

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return interArea / float(boxAArea + boxBArea - interArea)