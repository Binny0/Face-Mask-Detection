from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import base64
import io
import os
import time
from PIL import Image
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

class WebMaskDetector:
    def __init__(self):
        """Initialize the web mask detector with error handling"""
        try:
            self.model = load_model('models/mask_detector.h5')
            self.classes = np.load('models/classes.npy')
            print(f"Model loaded successfully. Classes: {self.classes}")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Create dummy model for demonstration
            self.model = None
            self.classes = np.array(['with_mask', 'without_mask', 'improper_mask'])
        
        # Initialize face detection
        self.face_net = self.load_face_detector()
        
        # Colors for different predictions
        self.colors = {
            'with_mask': (0, 255, 0),      # Green
            'without_mask': (0, 0, 255),   # Red
            'improper_mask': (0, 165, 255) # Orange
        }
        
        # Video capture for webcam
        self.cap = None
        self.webcam_active = False
        self.detection_stats = {
            'total_detections': 0,
            'with_mask': 0,
            'without_mask': 0,
            'improper_mask': 0
        }
    
    def load_face_detector(self):
        """Load face detection model with fallback"""
        try:
            # Try DNN face detector first
            face_net = cv2.dnn.readNetFromTensorflow(
                'models/opencv_face_detector_uint8.pb',
                'models/opencv_face_detector.pbtxt'
            )
            print("Using DNN face detector")
            return face_net
        except:
            print("DNN face detector not found, using Haar cascade fallback")
            return None
    
    def detect_faces(self, frame, confidence_threshold=0.5):
        """Enhanced face detection with multiple methods"""
        if self.face_net is not None:
            return self.detect_faces_dnn(frame, confidence_threshold)
        else:
            return self.detect_faces_haar(frame)
    
    def detect_faces_dnn(self, frame, confidence_threshold):
        """DNN-based face detection"""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        locations = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                
                # Ensure bounding boxes are within frame
                x, y = max(0, x), max(0, y)
                x1, y1 = min(w - 1, x1), min(h - 1, y1)
                
                # Skip small faces
                if (x1 - x) < 30 or (y1 - y) < 30:
                    continue
                
                face = self.extract_face(frame, x, y, x1, y1)
                if face is not None:
                    faces.append(face)
                    locations.append((x, y, x1, y1))
        
        return faces, locations
    
    def detect_faces_haar(self, frame):
        """Haar cascade face detection fallback"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_haar = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        
        faces = []
        locations = []
        
        for (x, y, w, h) in faces_haar:
            face = self.extract_face(frame, x, y, x + w, y + h)
            if face is not None:
                faces.append(face)
                locations.append((x, y, x + w, y + h))
        
        return faces, locations
    
    def extract_face(self, frame, x, y, x1, y1):
        """Extract and preprocess face region"""
        # Add padding
        h, w = frame.shape[:2]
        padding = 10
        y_start = max(0, y - padding)
        y_end = min(h, y1 + padding)
        x_start = max(0, x - padding)
        x_end = min(w, x1 + padding)
        
        face = frame[y_start:y_end, x_start:x_end]
        
        if face.size == 0:
            return None
        
        # Preprocess face
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = face.astype(np.float32) / 255.0
        
        return face
    
    def predict_mask(self, faces):
        """Predict mask status with enhanced error handling"""
        if len(faces) == 0 or self.model is None:
            return []
        
        try:
            faces = np.array(faces, dtype=np.float32)
            predictions = self.model.predict(faces, batch_size=8, verbose=0)
            
            results = []
            for pred in predictions:
                class_idx = np.argmax(pred)
                confidence = float(pred[class_idx])
                label = self.classes[class_idx]
                results.append((label, confidence))
            
            # Update statistics
            for label, _ in results:
                self.detection_stats['total_detections'] += 1
                if label in self.detection_stats:
                    self.detection_stats[label] += 1
            
            return results
        except Exception as e:
            print(f"Prediction error: {e}")
            return []
    
    def draw_results(self, frame, locations, predictions):
        """Draw bounding boxes and labels with improved styling"""
        for (box, (label, confidence)) in zip(locations, predictions):
            (x, y, x1, y1) = box
            color = self.colors.get(label, (255, 255, 255))
            
            # Dynamic thickness based on confidence
            thickness = max(2, int(confidence * 4))
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x1, y1), color, thickness)
            
            # Prepare label
            display_label = label.replace('_', ' ').title()
            text = f"{display_label}: {confidence:.2f}"
            
            # Calculate text size
            font_scale = 0.6
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Draw text background
            text_y = y - 10
            if text_y < text_height + 10:
                text_y = y + 25
            
            cv2.rectangle(frame, (x, text_y - text_height - 5), 
                         (x + text_width, text_y + 5), color, -1)
            
            # Draw text
            cv2.putText(frame, text, (x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        return frame
    
    def process_image(self, image):
        """Process uploaded image"""
        # Convert PIL to OpenCV format
        image_np = np.array(image)
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        
        # Detect faces and predict masks
        faces, locations = self.detect_faces(image_np)
        predictions = self.predict_mask(faces)
        
        # Draw results
        result_image = self.draw_results(image_np, locations, predictions)
        
        return result_image, predictions
    
    def start_webcam(self):
        """Initialize webcam"""
        if not self.webcam_active:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.webcam_active = True
    
    def stop_webcam(self):
        """Stop webcam"""
        if self.webcam_active and self.cap:
            self.cap.release()
            self.webcam_active = False

# Initialize detector
detector = WebMaskDetector()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        image = Image.open(filepath)
        result_image, predictions = detector.process_image(image)
        
        # Convert result to base64
        _, buffer = cv2.imencode('.jpg', result_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_str = base64.b64encode(buffer).decode()
        
        # Prepare response data
        detection_results = []
        for label, confidence in predictions:
            detection_results.append({
                'label': label.replace('_', ' ').title(),
                'confidence': round(confidence, 3),
                'color': 'success' if label == 'with_mask' else 'danger' if label == 'without_mask' else 'warning'
            })
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'success': True,
            'image': img_str,
            'detections': len(predictions),
            'results': detection_results,
            'stats': detector.detection_stats
        })
    
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

def generate_frames():
    """Generate frames for video streaming"""
    detector.start_webcam()
    
    frame_count = 0
    fps_start = time.time()
    fps = 0
    
    while detector.webcam_active:
        try:
            if detector.cap is None:
                break
                
            ret, frame = detector.cap.read()
            if not ret:
                break
            
            # Calculate FPS
            frame_count += 1
            if time.time() - fps_start >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_start = time.time()
            
            # Process frame
            faces, locations = detector.detect_faces(frame)
            predictions = detector.predict_mask(faces)
            
            # Draw results
            frame = detector.draw_results(frame, locations, predictions)
            
            # Add FPS and detection info
            cv2.putText(frame, f"FPS: {fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        except Exception as e:
            print(f"Frame generation error: {e}")
            break
    
    detector.stop_webcam()

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    """Start webcam endpoint"""
    try:
        detector.start_webcam()
        return jsonify({'success': True, 'message': 'Webcam started'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    """Stop webcam endpoint"""
    try:
        detector.stop_webcam()
        return jsonify({'success': True, 'message': 'Webcam stopped'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stats')
def get_stats():
    """Get detection statistics"""
    return jsonify(detector.detection_stats)

@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    """Reset detection statistics"""
    detector.detection_stats = {
        'total_detections': 0,
        'with_mask': 0,
        'without_mask': 0,
        'improper_mask': 0
    }
    return jsonify({'success': True, 'message': 'Statistics reset'})

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Face Mask Detection Web Application...")
    print("Available endpoints:")
    print("  / - Main interface")
    print("  /upload - Image upload and processing")
    print("  /video_feed - Live webcam stream")
    print("  /stats - Detection statistics")
    
    app.run(debug=True, host='0.0.0.0', port=8080 , threaded=True)