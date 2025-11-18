import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import argparse
import imutils
import time
from collections import deque

class MaskDetector:
    def __init__(self, model_path='models/mask_detector.h5', confidence_threshold=0.5):
        """
        Initialize the mask detector with improved features
        """
        try:
            self.model = load_model(model_path)
            self.classes = np.load('models/classes.npy')
            print(f"Model loaded successfully. Classes: {self.classes}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        self.confidence_threshold = confidence_threshold
        
        # Load face detector (try both methods)
        self.face_net = self.load_face_detector()
        
        # For smoothing predictions in video
        self.prediction_history = {}
        self.history_length = 5
        
        # Colors for different classes
        self.colors = {
            'with_mask': (0, 255, 0),      # Green
            'without_mask': (0, 0, 255),   # Red
            'improper_mask': (0, 165, 255) # Orange
        }
        
    def load_face_detector(self):
        """Load face detection model with fallback options"""
        try:
            # Try to load DNN face detector first (more accurate)
            face_net = cv2.dnn.readNetFromTensorflow(
                'models/opencv_face_detector_uint8.pb',
                'models/opencv_face_detector.pbtxt'
            )
            print("Using DNN face detector")
            return face_net
        except:
            print("DNN face detector not found, using Haar cascade")
            return None
    
    def detect_faces_dnn(self, frame):
        """Detect faces using DNN (more accurate)"""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        locations = []
        confidences = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                
                # Ensure bounding boxes fall within frame dimensions
                x, y = max(0, x), max(0, y)
                x1, y1 = min(w - 1, x1), min(h - 1, y1)
                
                # Skip if face is too small
                if (x1 - x) < 50 or (y1 - y) < 50:
                    continue
                
                # Extract face ROI with padding
                padding = 20
                y_start = max(0, y - padding)
                y_end = min(h, y1 + padding)
                x_start = max(0, x - padding)
                x_end = min(w, x1 + padding)
                
                face = frame[y_start:y_end, x_start:x_end]
                
                if face.size > 0:
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = face.astype(np.float32) / 255.0
                    
                    faces.append(face)
                    locations.append((x, y, x1, y1))
                    confidences.append(confidence)
        
        return faces, locations, confidences
    
    def detect_faces_haar(self, frame):
        """Fallback face detection using Haar cascades"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_haar = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        
        faces = []
        locations = []
        confidences = []
        
        for (x, y, w, h) in faces_haar:
            # Extract face with padding
            padding = 10
            h_frame, w_frame = frame.shape[:2]
            y_start = max(0, y - padding)
            y_end = min(h_frame, y + h + padding)
            x_start = max(0, x - padding)
            x_end = min(w_frame, x + w + padding)
            
            face = frame[y_start:y_end, x_start:x_end]
            
            if face.size > 0:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = face.astype(np.float32) / 255.0
                
                faces.append(face)
                locations.append((x, y, x + w, y + h))
                confidences.append(0.9)  # Default confidence for Haar
        
        return faces, locations, confidences
    
    def detect_faces(self, frame):
        """Detect faces using the best available method"""
        if self.face_net is not None:
            return self.detect_faces_dnn(frame)
        else:
            return self.detect_faces_haar(frame)
    
    def predict_mask(self, faces):
        """Predict mask status with confidence scores"""
        if len(faces) == 0:
            return []
        
        faces = np.array(faces, dtype=np.float32)
        predictions = self.model.predict(faces, batch_size=32, verbose=0)
        
        results = []
        for pred in predictions:
            class_idx = np.argmax(pred)
            confidence = pred[class_idx]
            label = self.classes[class_idx]
            results.append((label, confidence))
        
        return results
    
    def smooth_predictions(self, face_id, prediction):
        """Smooth predictions over time for video"""
        if face_id not in self.prediction_history:
            self.prediction_history[face_id] = deque(maxlen=self.history_length)
        
        self.prediction_history[face_id].append(prediction)
        
        # Get most common prediction
        labels = [p[0] for p in self.prediction_history[face_id]]
        confidences = [p[1] for p in self.prediction_history[face_id]]
        
        # Use voting for stability
        unique_labels, counts = np.unique(labels, return_counts=True)
        most_common_label = unique_labels[np.argmax(counts)]
        
        # Average confidence for the most common label
        label_confidences = [conf for label, conf in zip(labels, confidences) if label == most_common_label]
        avg_confidence = np.mean(label_confidences)
        
        return most_common_label, avg_confidence
    
    def draw_prediction(self, frame, box, label, confidence, face_confidence=None):
        """Draw bounding box and prediction with improved styling"""
        (x, y, x1, y1) = box
        color = self.colors.get(label, (255, 255, 255))
        
        # Draw bounding box with thickness based on confidence
        thickness = max(2, int(confidence * 4))
        cv2.rectangle(frame, (x, y), (x1, y1), color, thickness)
        
        # Prepare label text
        text = f"{label.replace('_', ' ').title()}: {confidence:.2f}"
        if face_confidence:
            text += f" (Face: {face_confidence:.2f})"
        
        # Calculate text background size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw text background
        cv2.rectangle(frame, (x, y - text_height - baseline - 5), 
                     (x + text_width, y), color, -1)
        
        # Draw text
        cv2.putText(frame, text, (x, y - baseline - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def detect_from_image(self, image_path, save_result=True):
        """Detect masks in a single image with enhanced output"""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image {image_path}")
            return None
        
        original_frame = frame.copy()
        
        # Detect faces and masks
        faces, locations, face_confidences = self.detect_faces(frame)
        predictions = self.predict_mask(faces)
        
        print(f"Detected {len(faces)} face(s)")
        
        # Draw results
        for i, (box, (label, confidence)) in enumerate(zip(locations, predictions)):
            face_conf = face_confidences[i] if i < len(face_confidences) else None
            frame = self.draw_prediction(frame, box, label, confidence, face_conf)
            
            print(f"Face {i+1}: {label} ({confidence:.2f})")
        
        if save_result:
            output_path = image_path.replace('.', '_result.')
            cv2.imwrite(output_path, frame)
            print(f"Result saved as {output_path}")
        
        return frame
    
    def detect_from_webcam(self, show_fps=True):
        """Real-time detection from webcam with FPS display"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # FPS calculation
        fps_counter = 0
        fps_start_time = time.time()
        fps_display = 0
        
        print("Starting webcam detection. Press 'q' to quit, 's' to save screenshot.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read from webcam")
                break
            
            frame = imutils.resize(frame, width=800)
            start_time = time.time()
            
            # Detect faces and masks
            faces, locations, face_confidences = self.detect_faces(frame)
            predictions = self.predict_mask(faces)
            
            # Apply smoothing for video
            smoothed_predictions = []
            for i, (box, (label, confidence)) in enumerate(zip(locations, predictions)):
                face_id = f"{box[0]}_{box[1]}"  # Simple face ID based on position
                smoothed_label, smoothed_confidence = self.smooth_predictions(face_id, (label, confidence))
                smoothed_predictions.append((smoothed_label, smoothed_confidence))
            
            # Draw results
            for i, (box, (label, confidence)) in enumerate(zip(locations, smoothed_predictions)):
                face_conf = face_confidences[i] if i < len(face_confidences) else None
                frame = self.draw_prediction(frame, box, label, confidence, face_conf)
            
            # Calculate and display FPS
            if show_fps:
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    fps_display = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()
                
                cv2.putText(frame, f"FPS: {fps_display}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display processing time
            process_time = time.time() - start_time
            cv2.putText(frame, f"Process Time: {process_time*1000:.1f}ms", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Display detection count
            cv2.putText(frame, f"Faces Detected: {len(faces)}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Face Mask Detection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = int(time.time())
                screenshot_path = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"Screenshot saved as {screenshot_path}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def detect_from_video(self, video_path, output_path=None):
        """Process video file with mask detection"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Properties: {width}x{height}, {fps}fps, {total_frames} frames")
        
        # Setup video writer if output path is specified
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect faces and masks
            faces, locations, face_confidences = self.detect_faces(frame)
            predictions = self.predict_mask(faces)
            
            # Draw results
            for i, (box, (label, confidence)) in enumerate(zip(locations, predictions)):
                face_conf = face_confidences[i] if i < len(face_confidences) else None
                frame = self.draw_prediction(frame, box, label, confidence, face_conf)
            
            # Add progress info
            progress = (frame_count / total_frames) * 100
            cv2.putText(frame, f"Progress: {progress:.1f}%", (10, height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if output_path:
                out.write(frame)
            
            # Show progress every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                estimated_total = elapsed * total_frames / frame_count
                remaining = estimated_total - elapsed
                print(f"Processed {frame_count}/{total_frames} frames "
                      f"({progress:.1f}%) - ETA: {remaining:.0f}s")
        
        cap.release()
        if output_path:
            out.release()
            print(f"Output video saved as {output_path}")
        
        total_time = time.time() - start_time
        print(f"Video processing completed in {total_time:.1f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Mask Detection')
    parser.add_argument('--mode', choices=['image', 'webcam', 'video'], default='webcam',
                       help='Detection mode')
    parser.add_argument('--input', type=str, help='Path to input image or video')
    parser.add_argument('--output', type=str, help='Path to output video (for video mode)')
    parser.add_argument('--model', type=str, default='models/mask_detector.h5',
                       help='Path to trained model')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Face detection confidence threshold')
    
    args = parser.parse_args()
    
    try:
        detector = MaskDetector(args.model, args.confidence)
        
        if args.mode == 'image':
            if not args.input:
                print("Please provide input image path with --input")
                exit(1)
            result = detector.detect_from_image(args.input)
            if result is not None:
                cv2.imshow("Result", result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        elif args.mode == 'video':
            if not args.input:
                print("Please provide input video path with --input")
                exit(1)
            detector.detect_from_video(args.input, args.output)
        
        else:  # webcam mode
            detector.detect_from_webcam()
    
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    except Exception as e:
        print(f"Error: {e}")