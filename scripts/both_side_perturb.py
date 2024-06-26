import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(x_train, y_train), (_, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = np.expand_dims(x_train, axis=-1)
y_train = to_categorical(y_train, num_classes=10)

# Define and train a simple neural network model
model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Compute gradients of the model's output with respect to the input image
def compute_gradients(model, input_image):
    with tf.GradientTape() as tape: # This records operations for automatic differentiation using TensorFlow's gradient tape.
        tape.watch(input_image)  # This tells TensorFlow to watch the input image tensor for changes,
                                  # allowing gradients to be computed with respect to it.
        predictions = model(input_image)
        loss = tf.reduce_mean(predictions)
    gradients = tape.gradient(loss, input_image) # This line computes the gradients of the loss with respect to the input image
    # tensor using automatic differentiation.
    return gradients

# Find points near the decision boundary
def find_boundary_points(model, x_data):
    boundary_points = []
    for i in range(len(x_data)):
        input_image = tf.convert_to_tensor(x_data[i:i+1])
        gradients = compute_gradients(model, input_image)
        gradient_magnitude = tf.norm(gradients)
        #print(gradient_magnitude)
        if gradient_magnitude > 2e-07:  # Threshold for large gradient magnitude
            boundary_points.append(input_image.numpy())# This condition checks if the magnitude of gradients is more
            # than a threshold, indicating that the point is near the decision boundary.
    return boundary_points

# Perturb a point by epsilon
def perturb_point(point, epsilon):
    return np.clip(point + epsilon * np.random.randn(*point.shape), 0, 1)

# Perturb a point in opposite direction by epsilon
def perturb_point_neg(point, epsilon):
    return np.clip(point - epsilon * np.random.randn(*point.shape), 0, 1)


# Check if perturbed point generates an adversarial example
def is_adversarial(model, original_point, perturbed_point):
    original_prediction = np.argmax(model.predict(original_point))
    perturbed_prediction = np.argmax(model.predict(perturbed_point))
    return original_prediction != perturbed_prediction

# Initialize parameters
epsilon = 0.2  # Perturbation amount

# Find points near the decision boundary
boundary_points = find_boundary_points(model, x_train)
print("Total Points :",len(x_train))
print("Boundary Points :",len(boundary_points))
#Generate adversarial examples
generated_images = []

#dictionary storing points and directions +1 or -1
pts = [] #points
dir = [] #corresponding direction
delta = 3 #parameter selecting how many new points we need to generate in our chosen direction
for point in boundary_points:
    perturbed_point = perturb_point(point, epsilon)
    perturbed_point_neg = perturb_point_neg(point, epsilon)
    if is_adversarial(model, point, perturbed_point):
        print("got_pos")
        pts.append(point)
        dir.append(1)
        generated_images.append(perturbed_point.squeeze())
        if len(generated_images) >= 10:
            break
    if is_adversarial(model, point, perturbed_point_neg):
        print("got_neg")
        pts.append(point)
        dir.append(-1)
        generated_images.append(perturbed_point.squeeze())
        if len(generated_images) >= 10:
            break

#generate more points in the selected direction
for i in range(len(pts)):
    while delta>0:
      if dir[i]==1:
        pts[i] = perturb_point(pts[i], epsilon)
      else:
        pts[i] = perturb_point_neg(pts[i],epsilon)
      if is_adversarial(model, point, pts[i]):
        generated_images.append(pts[i].squeeze())
      delta-=1
      
# # Plot generated adversarial examples
plt.figure(figsize=(10, 2))
for i in range(len(generated_images)):
    plt.subplot(1, len(generated_images), i+1)
    plt.imshow(generated_images[i], cmap='gray')

    plt.axis('off')
plt.tight_layout()
plt.show()