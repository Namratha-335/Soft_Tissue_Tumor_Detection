# # ------------------------------------------
# # IMPORTS
# # ------------------------------------------
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# from sklearn.metrics import confusion_matrix, classification_report
# import seaborn as sns

# # ------------------------------------------
# # DATASET PATHS
# # ------------------------------------------
# train_dir = "Brain_Tumor_Dataset/Training"
# test_dir  = "Brain_Tumor_Dataset/Testing"

# IMG_SIZE = (224, 224)
# BATCH_SIZE = 32

# # ------------------------------------------
# # DATA PREPROCESSING
# # ------------------------------------------
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=30,
#     zoom_range=0.3,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     horizontal_flip=True,
#     brightness_range=[0.7, 1.3],
#     validation_split=0.2
# )

# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='training'
# )

# val_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='validation'
# )

# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     shuffle=False
# )

# # ------------------------------------------
# # BUILD MODEL (TRANSFER LEARNING)
# # ------------------------------------------
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
# base_model.trainable = False  # freeze pretrained layers

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(0.5)(x)
# preds = Dense(4, activation='softmax')(x)  # 4 classes

# model = Model(inputs=base_model.input, outputs=preds)

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()

# # ------------------------------------------
# # TRAIN MODEL
# # ------------------------------------------
# es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# history = model.fit(
#     train_generator,
#     epochs=20,
#     validation_data=val_generator,
#     callbacks=[es]
# )

# # Save model
# model.save("tumor_transfer_model.h5")

# # ------------------------------------------
# # PLOT TRAINING RESULTS
# # ------------------------------------------
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.legend()
# plt.title('Accuracy Curve')
# plt.show()

# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.legend()
# plt.title('Loss Curve')
# plt.show()

# # ------------------------------------------
# # EVALUATE ON TEST DATA
# # ------------------------------------------
# test_loss, test_acc = model.evaluate(test_generator)
# print("Test Accuracy:", test_acc)

# # ------------------------------------------
# # CONFUSION MATRIX + CLASSIFICATION REPORT
# # ------------------------------------------
# Y_pred = model.predict(test_generator)
# y_pred = np.argmax(Y_pred, axis=1)

# cm = confusion_matrix(test_generator.classes, y_pred)
# plt.figure(figsize=(6,6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
#             xticklabels=test_generator.class_indices.keys(),
#             yticklabels=test_generator.class_indices.keys())
# plt.title("Confusion Matrix")
# plt.show()

# print(classification_report(test_generator.classes, y_pred, 
#                             target_names=test_generator.class_indices.keys()))
# # ------------------------------------------
# # GRAD-CAM HEATMAP (FIXED VERSION)
# # ------------------------------------------
# from tensorflow.keras.preprocessing import image
# import os

# # ==== FIX 1: GET AN ACTUAL IMAGE FILE ====
# # Option A: Get the first image from the test directory
# test_class = "C:/Users/SMILE/OneDrive/Desktop/Soft_Tissue_Tumor_Detection/Brain_Tumor_Dataset/Testing/meningioma_tumor"  # Change to: glioma_tumor, meningioma_tumor, no_tumor, or pituitary_tumor
# test_class_dir = os.path.join(test_dir, test_class)
# image_files = [f for f in os.listdir(test_class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
# if image_files:
#     img_path = os.path.join(test_class_dir, image_files[0])
# else:
#     raise FileNotFoundError(f"No images found in {test_class_dir}")

# print(f"Using image: {img_path}")

# # Load and preprocess image
# img = image.load_img(img_path, target_size=(224, 224))
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0) / 255.0

# # ==== FIX 2: FIND THE CORRECT LAST CONV LAYER ====
# # For MobileNetV2, the last conv layer is typically 'Conv_1' or 'out_relu'
# # Let's use a safer approach
# last_conv_layer_name = None

# # Try common MobileNetV2 layer names first
# for layer_name in ['out_relu', 'Conv_1', 'Conv_1_bn']:
#     try:
#         layer = base_model.get_layer(layer_name)
#         last_conv_layer_name = layer_name
#         print(f"Found layer: {last_conv_layer_name}")
#         break
#     except:
#         continue

# # If not found, search through layers
# if last_conv_layer_name is None:
#     for layer in reversed(base_model.layers):
#         try:
#             # Check if layer has output_shape and it's 4D
#             if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
#                 last_conv_layer_name = layer.name
#                 break
#             # Alternative: check layer type
#             elif isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
#                 last_conv_layer_name = layer.name
#                 break
#         except:
#             continue

# if last_conv_layer_name is None:
#     raise ValueError("Could not find a convolutional layer in the model")

# print(f"Using last conv layer: {last_conv_layer_name}")

# # ==== FIX 3: CREATE GRAD-CAM MODEL ====
# grad_model = tf.keras.models.Model(
#     inputs=model.input,
#     outputs=[base_model.get_layer(last_conv_layer_name).output, model.output]
# )

# # ==== FIX 4: COMPUTE GRADIENTS CORRECTLY ====
# with tf.GradientTape() as tape:
#     conv_outputs, predictions = grad_model(img_array)
#     pred_index = tf.argmax(predictions[0])
#     class_channel = predictions[:, pred_index]

# # Calculate gradients of the predicted class with respect to the feature maps
# grads = tape.gradient(class_channel, conv_outputs)

# # Global average pooling of gradients
# pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

# # Weight the channels by the pooled gradients
# conv_outputs = conv_outputs[0].numpy()
# pooled_grads = pooled_grads.numpy()

# # Compute weighted combination
# for i in range(pooled_grads.shape[-1]):
#     conv_outputs[:, :, i] *= pooled_grads[i]

# # Create heatmap by averaging across channels
# heatmap = np.mean(conv_outputs, axis=-1)

# # Normalize heatmap between 0 and 1
# heatmap = np.maximum(heatmap, 0)
# if np.max(heatmap) != 0:
#     heatmap /= np.max(heatmap)

# # ==== FIX 5: OVERLAY VISUALIZATION ====
# # Read original image
# orig_img = cv2.imread(img_path)
# if orig_img is None:
#     raise ValueError(f"Could not read image at {img_path}")

# # Resize heatmap to match image size
# heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))

# # Convert heatmap to RGB
# heatmap_colored = np.uint8(255 * heatmap_resized)
# heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)

# # Superimpose heatmap on original image
# superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)

# # Get class names
# class_names = list(test_generator.class_indices.keys())
# predicted_class = class_names[pred_index.numpy()]
# confidence = float(predictions[0][pred_index]) * 100

# # ==== DISPLAY RESULTS ====
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# axes[0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
# axes[0].set_title("Original Image")
# axes[0].axis("off")

# axes[1].imshow(heatmap_resized, cmap='jet')
# axes[1].set_title("Grad-CAM Heatmap")
# axes[1].axis("off")

# axes[2].imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
# axes[2].set_title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%")
# axes[2].axis("off")

# plt.tight_layout()
# plt.savefig('gradcam_result.png', dpi=150, bbox_inches='tight')
# plt.show()

# print(f"\nPredicted class: {predicted_class}")
# print(f"Confidence: {confidence:.2f}%")
# print(f"All class probabilities:")
# for i, class_name in enumerate(class_names):
#     print(f"  {class_name}: {float(predictions[0][i]) * 100:.2f}%")








# ------------------------------------------
# IMPORTS
# ------------------------------------------
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

# ------------------------------------------
# DATASET PATHS
# ------------------------------------------
train_dir = "Brain_Tumor_Dataset/Training"
test_dir  = "Brain_Tumor_Dataset/Testing"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

print("=" * 70)
print("BRAIN TUMOR DETECTION - TRAINING")
print("=" * 70)

# ------------------------------------------
# DATA PREPROCESSING
# ------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    zoom_range=0.4,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.6, 1.4],
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"\nTraining samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Test samples: {test_generator.samples}")
print(f"Classes: {list(train_generator.class_indices.keys())}")

# ------------------------------------------
# COMPUTE CLASS WEIGHTS
# ------------------------------------------
print("\n" + "=" * 70)
print("Computing class weights for balanced training...")
print("=" * 70)

for class_name, class_idx in train_generator.class_indices.items():
    count = np.sum(train_generator.classes == class_idx)
    print(f"{class_name}: {count} samples")

class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

print("\nClass weights applied:")
for class_name, class_idx in train_generator.class_indices.items():
    print(f"{class_name}: {class_weights[class_idx]:.3f}")
print("=" * 70)

# ------------------------------------------
# BUILD MODEL (TRANSFER LEARNING)
# ------------------------------------------
print("\nBuilding model...")

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Unfreeze last 50 layers for better learning
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

print(f"Total layers: {len(base_model.layers)}")
print(f"Trainable layers: 50 + custom head")

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.6)(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
preds = Dense(4, activation='softmax', kernel_regularizer=l2(0.01))(x)

model = Model(inputs=base_model.input, outputs=preds)

model.compile(
    optimizer=Adam(learning_rate=0.00005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model compiled")
model.summary()

# ------------------------------------------
# CALLBACKS
# ------------------------------------------
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    ),
    ModelCheckpoint(
        'tumor_transfer_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        verbose=1,
        min_lr=1e-7
    )
]

# ------------------------------------------
# TRAIN MODEL
# ------------------------------------------
print("\n" + "=" * 70)
print("TRAINING STARTED")
print("=" * 70 + "\n")

history = model.fit(
    train_generator,
    epochs=40,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# Save model
model.save("tumor_transfer_model.h5")
print("\n✓ Model saved: tumor_transfer_model.h5")

# ------------------------------------------
# PLOT TRAINING RESULTS
# ------------------------------------------
print("\nGenerating training plots...")

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.legend()
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.legend()
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()

print("✓ Saved: training_history.png")

# ------------------------------------------
# EVALUATE ON TEST DATA
# ------------------------------------------
print("\n" + "=" * 70)
print("EVALUATING ON TEST DATA")
print("=" * 70)

test_loss, test_acc = model.evaluate(test_generator)
print(f"\n** Test Accuracy: {test_acc*100:.2f}% **")

# ------------------------------------------
# CONFUSION MATRIX + CLASSIFICATION REPORT
# ------------------------------------------
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

cm = confusion_matrix(test_generator.classes, y_pred)
plt.figure(figsize=(8,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=test_generator.class_indices.keys(),
            yticklabels=test_generator.class_indices.keys(),
            cbar_kws={'label': 'Count'})
plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

print("✓ Saved: confusion_matrix.png")

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
report = classification_report(test_generator.classes, y_pred, 
                            target_names=test_generator.class_indices.keys(),
                            digits=3)
print(report)

# Save report
with open('classification_report.txt', 'w') as f:
    f.write("Brain Tumor Detection - Classification Report\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
    f.write(f"Test Loss: {test_loss:.4f}\n\n")
    f.write(report)

print("✓ Saved: classification_report.txt")

# ------------------------------------------
# TEST EACH CLASS
# ------------------------------------------
print("\n" + "=" * 70)
print("TESTING PREDICTIONS ON EACH CLASS")
print("=" * 70)

from tensorflow.keras.preprocessing import image
import os

all_correct = 0
all_total = 0

for class_name in sorted(train_generator.class_indices.keys()):
    class_path = os.path.join(test_dir, class_name)
    if os.path.exists(class_path):
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if image_files:
            correct = 0
            total = min(10, len(image_files))  # Test up to 10 images per class
            
            for img_file in image_files[:total]:
                img_path = os.path.join(class_path, img_file)
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                
                pred = model.predict(img_array, verbose=0)[0]
                predicted_class = list(train_generator.class_indices.keys())[np.argmax(pred)]
                confidence = np.max(pred) * 100
                
                if predicted_class == class_name:
                    correct += 1
            
            all_correct += correct
            all_total += total
            
            accuracy = (correct / total) * 100
            bar = "█" * int(accuracy / 5) + "░" * (20 - int(accuracy / 5))
            
            status = "✅" if accuracy >= 70 else "⚠️" if accuracy >= 50 else "❌"
            print(f"{status} {class_name:20s} | {bar} | {correct:2d}/{total:2d} ({accuracy:5.1f}%)")

overall_sample_acc = (all_correct / all_total) * 100 if all_total > 0 else 0
print(f"\nOverall Sample Accuracy: {overall_sample_acc:.1f}% ({all_correct}/{all_total})")

# ------------------------------------------
# GRAD-CAM HEATMAP (FIXED VERSION)
# ------------------------------------------
print("\n" + "=" * 70)
print("GENERATING GRAD-CAM VISUALIZATION")
print("=" * 70)

# Get first available class
test_class = None
for class_name in sorted(test_generator.class_indices.keys()):
    class_path = os.path.join(test_dir, class_name)
    if os.path.exists(class_path):
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if image_files:
            test_class = class_name
            break

if test_class:
    test_class_dir = os.path.join(test_dir, test_class)
    image_files = [f for f in os.listdir(test_class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if image_files:
        img_path = os.path.join(test_class_dir, image_files[0])
    else:
        raise FileNotFoundError(f"No images found in {test_class_dir}")

    print(f"Using image: {img_path}")

    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Find the correct last conv layer
    last_conv_layer_name = None

    for layer_name in ['out_relu', 'Conv_1', 'Conv_1_bn']:
        try:
            layer = base_model.get_layer(layer_name)
            last_conv_layer_name = layer_name
            print(f"Found layer: {last_conv_layer_name}")
            break
        except:
            continue

    if last_conv_layer_name is None:
        for layer in reversed(base_model.layers):
            try:
                if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                    last_conv_layer_name = layer.name
                    break
                elif isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    last_conv_layer_name = layer.name
                    break
            except:
                continue

    if last_conv_layer_name is None:
        print("⚠️ Could not find convolutional layer for Grad-CAM")
    else:
        print(f"Using last conv layer: {last_conv_layer_name}")

        # Create Grad-CAM model
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[base_model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0].numpy()
        pooled_grads = pooled_grads.numpy()

        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        # Read original image
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            raise ValueError(f"Could not read image at {img_path}")

        heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
        heatmap_colored = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)

        class_names = list(test_generator.class_indices.keys())
        predicted_class = class_names[pred_index.numpy()]
        confidence = float(predictions[0][pred_index]) * 100

        # Display results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(heatmap_resized, cmap='jet')
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        axes[2].imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig('gradcam_result.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("✓ Saved: gradcam_result.png")

        print(f"\nPredicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"\nAll class probabilities:")
        for i, class_name in enumerate(class_names):
            print(f"  {class_name}: {float(predictions[0][i]) * 100:.2f}%")

# ------------------------------------------
# FINAL SUMMARY
# ------------------------------------------
print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"✓ Test Accuracy: {test_acc*100:.2f}%")
print(f"✓ Model saved: tumor_transfer_model.h5")
print(f"✓ Files generated:")
print(f"   - training_history.png")
print(f"   - confusion_matrix.png")
print(f"   - classification_report.txt")
print(f"   - gradcam_result.png")
print("=" * 70)