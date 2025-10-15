import numpy as np

# Train npz data
train_x=np.load("train.npz")["x"]# 25 data points
train_y=np.load("train.npz")["y"]

# Test npz data 
test_x=np.load("test.npz")["x"]#100 data points
test_y=np.load("test.npz")["y"]

# Train 100 npz data
train_x_100=np.load("train_100.npz")["x"]#100 data points
train_y_100=np.load("train_100.npz")["y"]#100 data points


print("INITIALIZED DATA COMPLETE")

print("Train x shape:",train_x.shape)
print("Train y shape:",train_y.shape)
print("Test x shape:",test_x.shape)
print("Test y shape:",test_y.shape)
print("Train 100 x shape:",train_x_100.shape)
print("Train 100 y shape:",train_y_100.shape)