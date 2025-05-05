import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QComboBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


class VideoTrackerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("People tracking system (YOLOv8 + DeepSORT)")
        self.setGeometry(100, 100, 1000, 800)

        # Initialize the model and tracker
        self.model_names = {
            "YOLOv8 Nano (Fastest)": "yolov8n.pt",
            "YOLOv8 Small": "yolov8s.pt",
            "YOLOv8 Medium": "yolov8m.pt",
            "YOLOv8 Large (Most accurate)": "yolov8l.pt"
        }
        self.current_model_name = "yolov8n.pt"
        self.detection_model = YOLO(self.current_model_name)

        # Using the default feature extractor, the tracking parameters are adjusted
        self.tracker = DeepSort(
            max_age=30,
            n_init=5,
            max_cosine_distance=0.3,
            max_iou_distance=0.5,
            nn_budget=100
        )

        self.track_history = {}  # Storing trajectory history {id: [points]}
        self.unique_ids = set()  # Store all the IDs that appear
        self.predicted_pos = {}  # Storing the predicted location {id: (x,y)}
        self.occlusion_threshold = 50  # Pixel distance threshold (for ID recovery)

        # UI components
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")

        self.info_label = QLabel("No video loaded", self)
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 14px;")

        self.model_combo = QComboBox(self)
        self.model_combo.addItems(self.model_names.keys())
        self.model_combo.currentTextChanged.connect(self.change_model)

        self.import_btn = QPushButton("Load video", self)
        self.import_btn.clicked.connect(self.import_video)

        self.play_btn = QPushButton("Play/Pause", self)
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)

        self.export_btn = QPushButton("Export Video", self)
        self.export_btn.clicked.connect(self.export_video)
        self.export_btn.setEnabled(False)

        # Layout
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Choose model:"))
        control_layout.addWidget(self.model_combo)
        control_layout.addWidget(self.import_btn)
        control_layout.addWidget(self.play_btn)
        control_layout.addWidget(self.export_btn)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label, stretch=1)
        main_layout.addWidget(self.info_label)
        main_layout.addLayout(control_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Video processing variables
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)
        self.is_playing = False
        self.frame_count = 0

        # Video export variables
        self.video_writer = None
        self.exporting = False
        self.export_frame_count = 0
        self.export_file_path = ""
        self.export_fps = 30
        self.export_width = 0
        self.export_height = 0

    def change_model(self, model_name):
        """Switch detection model"""
        self.current_model_name = self.model_names[model_name]
        try:
            self.detection_model = YOLO(self.current_model_name)
            self.info_label.setText(f"Switched model: {model_name}")
            self.reset_tracking()
        except Exception as e:
            self.info_label.setText(f"Model loading failed: {str(e)}")

    def reset_tracking(self):
        """Reset tracking state"""
        self.unique_ids.clear()
        self.track_history.clear()
        self.predicted_pos.clear()

    def import_video(self):
        """Import video file"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose video", "", "Video (*.mp4 *.avi *.mov)")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            if self.cap.isOpened():
                self.play_btn.setEnabled(True)
                self.export_btn.setEnabled(True)
                self.info_label.setText(
                    f"Video loaded: {file_path.split('/')[-1]} | Use model: {self.model_combo.currentText()}")
                self.reset_tracking()

                # Get video info for export
                self.export_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                self.export_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.export_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def export_video(self):
        """Export processed video"""
        if not self.cap or not self.cap.isOpened():
            self.info_label.setText("No video loaded to export")
            return

        # Stop current playback
        if self.is_playing:
            self.toggle_play()

        # Choose save path
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processed Video",
            "",
            "MP4 files (*.mp4);;AVI files (*.avi)"
        )

        if not save_path:
            return  # User cancelled

        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_count = 0
        self.reset_tracking()

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for AVI
        self.video_writer = cv2.VideoWriter(
            save_path,
            fourcc,
            self.export_fps,
            (self.export_width, self.export_height)
        )

        if not self.video_writer.isOpened():
            self.info_label.setText("Failed to initialize video writer")
            return

        self.exporting = True
        self.export_file_path = save_path
        self.export_frame_count = 0
        self.info_label.setText(f"Exporting video to: {save_path.split('/')[-1]}...")

        # Use timer for export processing
        self.timer.start(30)  # ~30ms per frame

    def toggle_play(self):
        """Toggle play/pause"""
        if not self.is_playing:
            self.timer.start(30)  # ~30ms per frame
            self.is_playing = True
            self.info_label.setText(f"Playing... | Model: {self.model_combo.currentText()}")
        else:
            self.timer.stop()
            self.is_playing = False
            self.info_label.setText(
                f"Pause | Total track {len(self.unique_ids)} person | Current frame: {self.frame_count}")

    def process_frame(self):
        """Process each video frame"""
        ret, frame = self.cap.read()
        self.frame_count += 1
        if not ret:
            self.handle_end_of_video()
            return

        # Process frame and get frame with tracking info
        processed_frame = self.process_tracking(frame.copy())

        # If in export mode, write processed frame
        if self.exporting:
            self.video_writer.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
            self.export_frame_count += 1
            self.info_label.setText(
                f"Exporting... {self.export_frame_count} frames | "
                f"Total track {len(self.unique_ids)} person | "
                f"Model: {self.model_combo.currentText()}"
            )

            # Check if export is complete
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.frame_count >= total_frames:
                self.finish_exporting()
                return

        # Display frame
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def process_tracking(self, frame):
        """Process tracking logic and return processed frame"""
        # YOLOv8 person detection
        results = self.detection_model(frame, classes=[0], verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = box
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

        # DeepSORT tracking
        tracks = self.tracker.update_tracks(detections, frame=frame)

        # Get all IDs in current frame
        current_ids = set()
        for track in tracks:
            if track.is_confirmed():
                current_ids.add(track.track_id)

        # Motion prediction and ID recovery
        self.update_predictions(current_ids, frame)
        recovered_ids = self.check_occlusion_recovery(tracks, current_ids)
        current_ids.update(recovered_ids)

        # Update tracking display
        self.update_tracking_display(frame, tracks, current_ids)

        return frame

    def handle_end_of_video(self):
        """Handle end of video"""
        self.timer.stop()
        self.is_playing = False

        if self.exporting:
            self.finish_exporting()
        else:
            self.info_label.setText(
                f"Video end | Total track {len(self.unique_ids)} person | "
                f"Total frame: {self.frame_count}"
            )

    def finish_exporting(self):
        """Finish video export"""
        self.exporting = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_count = 0
        self.reset_tracking()

        self.info_label.setText(
            f"Export completed: {self.export_file_path.split('/')[-1]} | "
            f"Total track {len(self.unique_ids)} person | "
            f"Total frame: {self.export_frame_count}"
        )

    def update_predictions(self, current_ids, frame):
        """Update predicted positions and visualize"""
        # Motion prediction for lost targets
        for track_id in list(self.track_history.keys()):
            if track_id not in current_ids and len(self.track_history[track_id]) >= 2:
                # Simple linear prediction
                last_pos = self.track_history[track_id][-1]
                prev_pos = self.track_history[track_id][-2]
                dx = last_pos[0] - prev_pos[0]
                dy = last_pos[1] - prev_pos[1]
                self.predicted_pos[track_id] = (last_pos[0] + dx, last_pos[1] + dy)

                # Visualize predicted point (yellow)
                cv2.circle(frame, (int(self.predicted_pos[track_id][0]),
                                   int(self.predicted_pos[track_id][1])), 5, (0, 255, 255), -1)
            elif track_id in current_ids:
                self.predicted_pos.pop(track_id, None)

    def check_occlusion_recovery(self, tracks, current_ids):
        """Check and recover IDs after occlusion"""
        recovered_ids = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr()
            center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

            # If it's a newly appeared track
            if track_id not in self.track_history:
                # Check if close to predicted position of any lost target
                for old_id, pred_pos in self.predicted_pos.items():
                    if old_id not in current_ids:
                        distance = np.sqrt((center[0] - pred_pos[0]) ** 2 +
                                           (center[1] - pred_pos[1]) ** 2)
                        if distance < self.occlusion_threshold:
                            # Recover old ID
                            self.track_history[old_id].append(center)
                            recovered_ids.add(old_id)
                            # Associate new detection with old ID
                            track.track_id = old_id
                            break

        return recovered_ids

    def update_tracking_display(self, frame, tracks, current_ids):
        """Update tracking display and trajectories"""
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr()
            center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

            # Update trajectory history
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append(center)

            # Draw bounding box and ID
            color = (0, 255, 0)  # Green
            if track_id in self.predicted_pos:
                color = (0, 165, 255)  # Orange for recovered IDs

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw trajectory
            if len(self.track_history[track_id]) > 1:
                for i in range(1, len(self.track_history[track_id])):
                    cv2.line(frame, self.track_history[track_id][i - 1],
                             self.track_history[track_id][i], (0, 0, 255), 2)

        self.unique_ids.update(current_ids)

    def closeEvent(self, event):
        """Release resources when closing window"""
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        self.timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoTrackerApp()
    window.show()
    sys.exit(app.exec_())
