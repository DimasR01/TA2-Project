import os
import numpy as np
import open3d as o3d
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Define key parameters
PARAMS = {
    'n_points': 500,  # Number of points in each point cloud sample
    'train_data_dir': 'C:/Users/dimas/Documents/Kuliah/Semester 8/TA2/training',  # Path to the train directory
    'test_data_dir': 'C:/Users/dimas/Documents/Kuliah/Semester 8/TA2/testing',  # Path to the test directory
    'epochs': 50, # 10
    'batch_size': 16, # 16
    'input_shape': (500, 2)  # Input shape based on n_points and 2 coordinates (x, y)
}

LABELS = {'kursiputar': 0, 'manusia': 1, 'sova': 2}

def load_point_cloud_data_from_pcd(data_dir, n_points, labels_dict):
    data = []  # Initialize an empty list to store point cloud data
    labels = []  # Initialize an empty list to store labels
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                if filename.endswith(".pcd"):
                    filepath = os.path.join(class_dir, filename)
                    pcd = o3d.io.read_point_cloud(filepath)
                    points = np.asarray(pcd.points)[:, :2]  # Use only x and y coordinates
                    if points.shape[0] >= n_points:
                        points = points[:n_points]
                    else:
                        # Pad with zeros if less than n_points
                        padding = np.zeros((n_points - points.shape[0], 2))
                        points = np.vstack((points, padding))
                    data.append(points)  # No need for a channel axis with Conv1D
                    label = labels_dict[class_name]
                    labels.append(label)
    return np.array(data), np.array(labels)

# Load training data
train_data, train_labels = load_point_cloud_data_from_pcd(PARAMS['train_data_dir'], PARAMS['n_points'], LABELS)
train_labels = to_categorical(train_labels, num_classes=len(LABELS))

# Load test data
test_data, test_labels = load_point_cloud_data_from_pcd(PARAMS['test_data_dir'], PARAMS['n_points'], LABELS)
test_labels_categorical = to_categorical(test_labels, num_classes=len(LABELS))

# Model Preparation
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation="relu", input_shape=PARAMS['input_shape']))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv1D(filters=128, kernel_size=2, activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv1D(filters=256, kernel_size=2, activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.7))
model.add(BatchNormalization())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(len(LABELS), activation="softmax"))  # Multi-class classification

model.summary()

model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

# Model Training
history = model.fit(train_data, train_labels, epochs=PARAMS['epochs'], batch_size=PARAMS['batch_size'], verbose=1)

# Model Evaluation
test_loss, test_accuracy = model.evaluate(test_data, test_labels_categorical, verbose=1)
print(f'Test Accuracy: {test_accuracy:.4f}')

# Predicting on test data
predictions = model.predict(test_data)
predicted_labels = np.argmax(predictions, axis=1)

# Displaying results
for i, (true_label, predicted_label) in enumerate(zip(test_labels, predicted_labels)):
    print(f'Test Sample {i+1}: True Label: {true_label}, Predicted Label: {predicted_label}')

# Plot Training Loss and Accuracy
plt.figure(figsize=(12, 6))
epochs = range(1, PARAMS['epochs'] + 1)

plt.plot(epochs, history.history['loss'], 'r', label='Training Loss')
plt.plot(epochs, history.history['accuracy'], 'b', label='Training Accuracy')

plt.title('Training Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend(loc='upper right')
plt.show()

# Confusion Matrix
cm = confusion_matrix(test_labels, predicted_labels)
print("Confusion Matrix:")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=LABELS.keys(), yticklabels=LABELS.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
report = classification_report(test_labels, predicted_labels, target_names=LABELS.keys())
print("Classification Report:")
print(report)
