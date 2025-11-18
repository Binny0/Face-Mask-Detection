import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0 # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

# Set memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class MaskDetectionTrainer:
    def __init__(self, data_dir='data', img_size=(224, 224), model_type='mobilenet'):
        self.data_dir = data_dir
        self.img_size = img_size
        self.model_type = model_type
        self.model = None
        
    def load_dataset(self):
        """Load and preprocess the dataset with better error handling"""
        print("Loading dataset...")
        data = []
        labels = []
        
        categories = ['with_mask', 'without_mask', 'improper_mask']
        
        for category in categories:
            path = os.path.join(self.data_dir, category)
            if not os.path.exists(path):
                print(f"Warning: Directory {path} does not exist!")
                continue
                
            category_count = 0
            for img_name in os.listdir(path):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                    
                img_path = os.path.join(path, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                        
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    
                    # Normalize pixel values
                    img = img.astype(np.float32) / 255.0
                    
                    data.append(img)
                    labels.append(category)
                    category_count += 1
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
            
            print(f"Loaded {category_count} images from {category}")
        
        # Convert to numpy arrays
        data = np.array(data, dtype=np.float32)
        labels = np.array(labels)
        
        # Encode labels
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        
        print(f"Dataset loaded: {len(data)} images")
        print(f"Classes: {le.classes_}")
        print(f"Class distribution: {np.bincount(labels_encoded)}")
        
        return data, labels_encoded, le
    
    def create_model(self, num_classes=3):
        """Create improved model with better architecture"""
        print(f"Creating {self.model_type} model...")
        
        # Choose base model
        if self.model_type == 'mobilenet':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        elif self.model_type == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        else:
            raise ValueError("Unsupported model type. Use 'mobilenet' or 'efficientnet'")
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add improved custom head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model, base_model
    
    def train_model(self):
        """Train the model with improved techniques"""
        # Load dataset
        X, y, label_encoder = self.load_dataset()
        
        if len(X) == 0:
            print("No data found! Please add images to data folders.")
            return None, None
        
        # Split dataset
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")
        
        # Create model
        self.model, base_model = self.create_model(num_classes=len(np.unique(y)))
        
        # Calculate class weights for imbalanced dataset
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        print(f"Class weights: {class_weight_dict}")
        
        # Enhanced data augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.1,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator()  # No augmentation for validation
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_mask_detector.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Phase 1: Train with frozen base
        print("\n=== Phase 1: Training with frozen base ===")
        history1 = self.model.fit(
            train_datagen.flow(X_train, y_train, batch_size=32),
            validation_data=val_datagen.flow(X_val, y_val, batch_size=32),
            epochs=15,
            steps_per_epoch=len(X_train) // 32,
            validation_steps=len(X_val) // 32,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Phase 2: Fine-tuning with unfrozen layers
        print("\n=== Phase 2: Fine-tuning ===")
        base_model.trainable = True
        
        # Freeze early layers, unfreeze later layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Fine-tuning callbacks
        finetune_callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=3,
                min_lr=1e-8,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_mask_detector_finetuned.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        history2 = self.model.fit(
            train_datagen.flow(X_train, y_train, batch_size=16),
            validation_data=val_datagen.flow(X_val, y_val, batch_size=16),
            epochs=10,
            steps_per_epoch=len(X_train) // 16,
            validation_steps=len(X_val) // 16,
            callbacks=finetune_callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Save final model
        self.model.save('models/mask_detector.h5')
        
        # Save label encoder classes
        np.save('models/classes.npy', label_encoder.classes_)
        
        print("Model training completed and saved!")
        
        # Combine and plot training history
        combined_history = self.combine_histories(history1, history2)
        self.plot_training_history(combined_history)
        
        # Evaluate model
        self.evaluate_model(X_val, y_val, label_encoder)
        
        return self.model, label_encoder
    
    def combine_histories(self, hist1, hist2):
        """Combine training histories from two phases"""
        combined = {}
        for key in hist1.history.keys():
            combined[key] = hist1.history[key] + hist2.history[key]
        return type('History', (), {'history': combined})()
    
    def evaluate_model(self, X_val, y_val, label_encoder):
        """Evaluate model performance"""
        from sklearn.metrics import classification_report, confusion_matrix
        
        print("\n=== Model Evaluation ===")
        
        # Predictions
        y_pred = self.model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred_classes, 
                                  target_names=label_encoder.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred_classes)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(label_encoder.classes_))
        plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
        plt.yticks(tick_marks, label_encoder.classes_)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, history):
        """Plot enhanced training history"""
        plt.figure(figsize=(15, 5))
        
        # Accuracy plot
        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.axvline(x=14, color='red', linestyle='--', alpha=0.7, label='Fine-tuning starts')
        plt.title('Model Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss plot
        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.axvline(x=14, color='red', linestyle='--', alpha=0.7, label='Fine-tuning starts')
        plt.title('Model Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate plot (if available)
        plt.subplot(1, 3, 3)
        if 'lr' in history.history:
            plt.plot(history.history['lr'], linewidth=2, color='orange')
            plt.title('Learning Rate', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Learning Rate\nNot Available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title('Learning Rate', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Face Mask Detection Model')
    parser.add_argument('--model', choices=['mobilenet', 'efficientnet'], 
                       default='mobilenet', help='Base model architecture')
    parser.add_argument('--data_dir', default='data', help='Path to data directory')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    
    args = parser.parse_args()
    
    trainer = MaskDetectionTrainer(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        model_type=args.model
    )
    
    model, label_encoder = trainer.train_model()