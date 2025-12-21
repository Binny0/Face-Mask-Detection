import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

class MaskDetector:
    def __init__(self):
        """Initialize the mask detector"""
        try:
            # Try loading model from different possible paths
            model_paths = ['models/mask_detector.h5', 'mask_detector.h5', 'mask_detector.model']
            self.model = None
            
            for path in model_paths:
                if os.path.exists(path):
                    self.model = load_model(path)
                    print(f"✅ Model loaded from: {path}")
                    break
            
            if self.model is None:
                print("⚠️ Model file not found. Please upload mask_detector.h5")
                
            # Load classes
            classes_paths = ['models/classes.npy', 'classes.npy']
            self.classes = None
            
            for path in classes_paths:
                if os.path.exists(path):
                    self.classes = np.load(path)
                    print(f"✅ Classes loaded: {self.classes}")
                    break
            
            if self.classes is None:
                # Default classes if file not found
                self.classes = np.array(['with_mask', 'without_mask', 'improper_mask'])
                print(f"⚠️ Using default classes: {self.classes}")
                
        except Exception as e:
            print(f"❌ Error during initialization: {e}")
            self.model = None
            self.classes = np.array(['with_mask', 'without_mask', 'improper_mask'])
        
        # Initialize face detection
        self.face_net = self.load_face_detector()
        
        # Colors for different predictions (RGB format for PIL)
        self.colors = {
            'with_mask': (0, 255, 0),      # Green
            'without_mask': (255, 0, 0),   # Red
            'improper_mask': (255, 165, 0) # Orange
        }
    
    def load_face_detector(self):
        """Load face detection model with fallback"""
        try:
            # Try loading DNN face detector
            model_paths = [
                ('models/opencv_face_detector_uint8.pb', 'models/opencv_face_detector.pbtxt'),
                ('opencv_face_detector_uint8.pb', 'opencv_face_detector.pbtxt')
            ]
            
            for pb_path, pbtxt_path in model_paths:
                if os.path.exists(pb_path) and os.path.exists(pbtxt_path):
                    face_net = cv2.dnn.readNetFromTensorflow(pb_path, pbtxt_path)
                    print("✅ Using DNN face detector")
                    return face_net
            
            print("⚠️ DNN face detector not found, using Haar cascade")
            return None
            
        except Exception as e:
            print(f"⚠️ Face detector loading error: {e}, using Haar cascade")
            return None
    
    def detect_faces_dnn(self, frame, confidence_threshold=0.5):
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
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
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
        """Predict mask status"""
        if len(faces) == 0 or self.model is None:
            return []
        
        try:
            faces_array = np.array(faces, dtype=np.float32)
            predictions = self.model.predict(faces_array, batch_size=8, verbose=0)
            
            results = []
            for pred in predictions:
                class_idx = np.argmax(pred)
                confidence = float(pred[class_idx])
                label = self.classes[class_idx]
                results.append((label, confidence))
            
            return results
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return []
    
    def draw_results(self, frame, locations, predictions):
        """Draw bounding boxes and labels"""
        for (box, (label, confidence)) in zip(locations, predictions):
            (x, y, x1, y1) = box
            color = self.colors.get(label, (255, 255, 255))
            
            # Draw bounding box
            thickness = 3
            cv2.rectangle(frame, (x, y), (x1, y1), color, thickness)
            
            # Prepare label
            display_label = label.replace('_', ' ').title()
            text = f"{display_label}: {confidence:.2%}"
            
            # Text settings
            font_scale = 0.7
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Draw text background
            text_y = y - 10
            if text_y < text_height + 10:
                text_y = y + text_height + 15
            
            cv2.rectangle(frame, (x, text_y - text_height - 5), 
                         (x + text_width + 5, text_y + 5), color, -1)
            
            # Draw text
            cv2.putText(frame, text, (x + 2, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        return frame
    
    def process_image(self, image):
        """Main processing function for Gradio"""
        if self.model is None:
            return None, "❌ Model not loaded. Please ensure mask_detector.h5 is in the repository."
        
        if image is None:
            return None, "⚠️ Please upload an image."
        
        try:
            # Convert PIL to OpenCV format (BGR)
            image_np = np.array(image)
            if len(image_np.shape) == 2:  # Grayscale
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
            elif image_np.shape[2] == 4:  # RGBA
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
            else:  # RGB
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Detect faces
            if self.face_net is not None:
                faces, locations = self.detect_faces_dnn(image_np)
            else:
                faces, locations = self.detect_faces_haar(image_np)
            
            if len(faces) == 0:
                return image, "⚠️ No faces detected in the image. Please try another image with clear faces."
            
            # Predict masks
            predictions = self.predict_mask(faces)
            
            if len(predictions) == 0:
                return image, "❌ Prediction failed. Please try again."
            
            # Draw results
            result_image = self.draw_results(image_np, locations, predictions)
            
            # Convert back to RGB for display
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            # Create results text
            results_text = f"**🎯 Detected {len(predictions)} face(s):**\n\n"
            
            for i, (label, confidence) in enumerate(predictions, 1):
                emoji = "😷" if label == "with_mask" else "❌" if label == "without_mask" else "⚠️"
                display_label = label.replace('_', ' ').title()
                results_text += f"{emoji} **Face {i}:** {display_label} ({confidence:.1%} confidence)\n"
            
            # Add summary
            mask_count = sum(1 for l, _ in predictions if l == 'with_mask')
            no_mask_count = sum(1 for l, _ in predictions if l == 'without_mask')
            improper_count = sum(1 for l, _ in predictions if l == 'improper_mask')
            
            results_text += f"\n**📊 Summary:**\n"
            results_text += f"✅ With Mask: {mask_count}\n"
            results_text += f"❌ Without Mask: {no_mask_count}\n"
            results_text += f"⚠️ Improper Mask: {improper_count}\n"
            
            return result_image, results_text
            
        except Exception as e:
            return None, f"❌ Error processing image: {str(e)}"

# Initialize detector
print("🚀 Initializing Face Mask Detector...")
detector = MaskDetector()

# Create Gradio interface
with gr.Blocks(title="Face Mask Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 😷 Face Mask Detection System
        
        Upload an image to detect faces and check if they're wearing masks properly.
        
        **Supported formats:** JPG, PNG, JPEG
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("🔍 Detect Masks", variant="primary", size="lg")
            
            gr.Markdown(
                """
                ### 📝 Instructions:
                1. Upload an image with clear faces
                2. Click "Detect Masks"
                3. View the results with bounding boxes
                
                **Legend:**
                - 🟢 Green: Wearing mask correctly
                - 🔴 Red: Not wearing mask
                - 🟠 Orange: Mask worn improperly
                """
            )
        
        with gr.Column():
            output_image = gr.Image(label="Detection Results")
            output_text = gr.Markdown(label="Analysis")
    
    # Examples
    gr.Markdown("### 🖼️ Try these examples:")
    gr.Examples(
        examples=[],  # Add example image paths here if available
        inputs=input_image
    )
    
    # Event handler
    submit_btn.click(
        fn=detector.process_image,
        inputs=input_image,
        outputs=[output_image, output_text]
    )
    
    gr.Markdown(
        """
        ---
        **Note:** This model detects three classes:
        - ✅ With Mask
        - ❌ Without Mask  
        - ⚠️ Improper Mask
        
        Built with TensorFlow and OpenCV | Deployed on Hugging Face 🤗
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
