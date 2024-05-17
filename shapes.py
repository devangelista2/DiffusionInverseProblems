import cv2
import numpy as np

##### PARAMETERS
N_train = 10_000
N_test = 1_000

nx, ny = 256, 256

# Circles
n_circles = 1
radius_m, radius_M = 20, 50

# Rectangles
n_rectangles = 1
l_min, l_max = 10, 30

# Initialization
train_images = np.zeros((N_train, nx, ny))
test_images = np.zeros((N_test, nx, ny))

#### TRAIN SET
for i in range(N_train):
    # CIRCLES
    for _ in range(n_circles):
        C = np.random.randint(radius_M, nx - radius_M - 1, size=(2, )) # Center
        R = np.random.randint(radius_m, radius_M, size=(1,)) # Radius
        intensity = np.random.uniform(0, 1, size=(1,))

        # Draw circle
        cv2.circle(train_images[i, :, :], (int(C[0]), int(C[1])), int(R), intensity, -1)

        # Add to the datum
        train_images[i, :, :] = train_images[i, :, :]

    # RECTANGLES
    for _ in range(n_rectangles):
        vertex1 = np.random.randint(0, nx - l_max, (2, ))
        vertex2 = vertex1 + np.random.randint(l_min, l_max, (2, ))
        intensity = np.random.uniform(0, 1, size=(1,))

        # Draw rectangle
        cv2.rectangle(train_images[i, :, :], (int(vertex1[0]), int(vertex1[1])), 
                      (int(vertex2[0]), int(vertex2[1])), intensity, -1)
    # Scale for uint conversion
    train_images[i, :, :] = train_images[i, :, :] / train_images[i, :, :].max() * 255
        

#### TEST SET
for i in range(N_test):
    # CIRCLES
    for _ in range(n_circles):
        C = np.random.randint(radius_M, nx - radius_M - 1, size=(2, )) # Center
        R = np.random.randint(radius_m, radius_M, size=(1,)) # Radius
        intensity = np.random.uniform(0, 1, size=(1,))

        # Draw circle
        cv2.circle(test_images[i, :, :], (int(C[0]), int(C[1])), int(R), intensity, -1)

    # RECTANGLES
    for _ in range(n_rectangles):
        vertex1 = np.random.randint(0, nx - l_max, (2, ))
        vertex2 = vertex1 + np.random.randint(l_min, l_max, (2, ))
        intensity = np.random.uniform(0, 1, size=(1,))

        # Draw rectangle
        cv2.rectangle(test_images[i, :, :], (int(vertex1[0]), int(vertex1[1])), 
                      (int(vertex2[0]), int(vertex2[1])), intensity, -1)
    # Scale for uint conversion
    test_images[i, :, :] = test_images[i, :, :] / test_images[i, :, :].max() * 255
        
np.save('./train.npy', train_images.astype(np.uint8))
np.save('./test.npy', test_images.astype(np.uint8))