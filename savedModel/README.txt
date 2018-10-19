
TRAINING HYPER-PARAMETER

batch_size = 128
eval_size = 1
image_width = train_images[0].shape[0]
image_height = train_images[0].shape[1]
target_size = max(train_labels) + 1
num_channels = 3
generations = 100
eval_every = 5
conv1_features = 32
conv2_features = 64
max_pool_size = 2
fully_connected_size1 = 400
fully_connected_size2 = 100
filter_size = 3
conv_stride = 1
dropout = 0.5