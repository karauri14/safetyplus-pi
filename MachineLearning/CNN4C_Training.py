import tensorflow as tf
import numpy as np
import time
import datetime
import cv2
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import tag_constants
from tensorflow.contrib.layers import flatten

ops.reset_default_graph()

TRAIN_DATA_DIR = "JPTSign/train_data"
TEST_DATA_DIR = "JPTSign/test_data"
VALID_DATA_DIR ="JPTSign/valid_data"
EXPORT_DIR = "./savedModel/"
DATASET_IMAGE_SIZE = 32
training_date = str(datetime.datetime(*(time.localtime())[:6]))


def load_dataset(data_dir):
    images = list()
    labels = list()

    classes = [d for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d)) ]

    for c in classes:
        label= os.path.join(data_dir,c)
        file_name = [os.path.join(label,f)
                    for f in os.listdir(label)
                    if os.path.isfile(os.path.join(label,f)) and not f.startswith('.')]

        for file in sorted(file_name):
            images.append(cv2.imread(file))
            labels.append(classes.index(c))
        #print("Class #{0}:{1}".format(classes.index(c), c))

    return images, np.array(labels)

# Load dataset
print("Training dataset is loading.....")
train_images, train_labels = load_dataset(TRAIN_DATA_DIR)
print("Testing dataset is loading.....")
test_images, test_labels = load_dataset(TEST_DATA_DIR)

# Resize to fixed size
train_images = [cv2.resize(image, (DATASET_IMAGE_SIZE,DATASET_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
            for image in train_images]
test_images = [cv2.resize(image, (DATASET_IMAGE_SIZE,DATASET_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
            for image in test_images]

train_images = np.array(train_images)
test_images = np.array(test_images)

train_images, train_labels = shuffle(train_images, train_labels)
test_images, test_labels = shuffle(test_images, test_labels)

print("Train_images #", len(train_images))
print("Test_images #", len(test_images))

# Nomalize
train_images = (train_images-train_images.mean())/(train_images.max()-train_images.min())
test_images = (test_images-test_images.mean())/(test_images.max()-test_images.min())

sess = tf.Session()

batch_size = 128
test_size = 200
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
fully_connected_size1 = 120
fully_connected_size2 = 60
filter_size = 3
conv_stride = 1
dropout = 0.7

keep_prob = tf.placeholder(tf.float32, (None), name='keep_prob')

x_input_shape = (None, image_width, image_height, num_channels)
x_input = tf.placeholder(tf.float32, shape=x_input_shape)
y_target = tf.placeholder(tf.int32, shape=(None))

test_input_shape = (None, image_width, image_height, num_channels)
test_input = tf.placeholder(tf.float32, shape=test_input_shape)
test_target = tf.placeholder(tf.int32, shape=(None))

eval_input_shape = (eval_size, image_width, image_height, num_channels)
eval_input = tf.placeholder(tf.float32, shape=eval_input_shape, name='x_input')
eval_target = tf.placeholder(tf.int32, shape=(eval_size), name='y_target')

conv1_weight = tf.Variable(tf.truncated_normal([filter_size, filter_size, num_channels, conv1_features], stddev=0.1, dtype=tf.float32))
conv1_bias = tf.Variable(tf.zeros([conv1_features],dtype=tf.float32))
conv2_weight = tf.Variable(tf.truncated_normal([filter_size, filter_size, conv1_features, conv2_features], stddev=0.1, dtype=tf.float32))
conv2_bias = tf.Variable(tf.zeros([conv2_features],dtype=tf.float32))

resulting_width = image_width // (max_pool_size * max_pool_size)
resulting_height = image_height // (max_pool_size * max_pool_size)
full1_input_size = resulting_width * resulting_height*conv2_features
full1_weight = tf.Variable(tf.truncated_normal([full1_input_size, fully_connected_size1], stddev=0.1, dtype=tf.float32))
full1_bias = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev=0.1, dtype=tf.float32))
full2_weight = tf.Variable(tf.truncated_normal([fully_connected_size1, fully_connected_size2], stddev=0.1, dtype=tf.float32))
full2_bias = tf.Variable(tf.truncated_normal([fully_connected_size2], stddev=0.1, dtype=tf.float32))
full3_weight = tf.Variable(tf.truncated_normal([fully_connected_size2, target_size], stddev=0.1, dtype=tf.float32))
full3_bias = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))

def my_conv_net(input_data, keep_prob):
   # First Conv-ReLU-MaxPool Layer
    conv1 = tf.nn.conv2d(input_data, conv1_weight, strides=[1, conv_stride, conv_stride, 1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
    max_pool1 = tf.nn.max_pool(relu1, ksize=[1, max_pool_size, max_pool_size, 1], strides=[1, max_pool_size, max_pool_size, 1], padding='SAME')
    max_pool1 = tf.nn.dropout(max_pool1, keep_prob)
    # Second Conv-ReLU-MaxPool Layer
    conv2 = tf.nn.conv2d(max_pool1, conv2_weight, strides=[1, conv_stride, conv_stride, 1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
    max_pool2 = tf.nn.max_pool(relu2, ksize=[1, max_pool_size, max_pool_size, 1], strides=[1, max_pool_size, max_pool_size, 1], padding='SAME')
    max_pool2 = tf.nn.dropout(max_pool2, keep_prob)
    # Transform Output into a 1xN layer for next fully connected layer
    #final_conv_shape = max_pool2.get_shape().as_list()
    #print(final_conv_shape)
    #final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
    #final_shape = 8 * 8 * 64
    #flat_output = tf.reshape(max_pool2, [None, final_shape])
    flat_output = flatten(max_pool2)
    # First Fully Connected Layer
    fully_connected1 = tf.nn.relu(tf.add(tf.matmul(flat_output, full1_weight), full1_bias))
    fully_connected1 = tf.nn.dropout(fully_connected1, keep_prob)
    # Second Fully Connected Layer
    fully_connected2 = tf.nn.relu(tf.add(tf.matmul(fully_connected1, full2_weight), full2_bias))
    fully_connected2 = tf.nn.dropout(fully_connected2, keep_prob)
    # Third Fully Connected Layer
    final_model_output = tf.add(tf.matmul(fully_connected2, full3_weight), full3_bias)
    return final_model_output

# Create accuracy function
def get_accuracy(logits, targets):
    batch_predictions = np.argmax(logits, axis=1)
    num_correct = np.sum(np.equal(batch_predictions, targets))
    return 100. * num_correct/batch_predictions.shape[0]
# Add name to tensor
def addNameToTensor(Tensor, Name):
    return tf.identity(Tensor, name=Name)

model_output = my_conv_net(x_input, keep_prob)
test_model_output = my_conv_net(test_input, keep_prob)
eval_model_output = my_conv_net(eval_input, keep_prob)
eval_model_output = addNameToTensor(eval_model_output, "model_output")

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=y_target))
test_loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=test_model_output, labels=test_target))
eval_loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=eval_model_output, labels=eval_target))

prediction = tf.nn.softmax(model_output)
test_prediction = tf.nn.softmax(test_model_output)
eval_prediction = tf.nn.softmax(eval_model_output, name='prediction')

my_optimizer = tf.train.AdamOptimizer()
train_step = my_optimizer.minimize(loss)
# Initialize Variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess.run(init)

num_train = len(train_images)
num_test = len(test_images)

train_loss = []
test_loss = []
train_acc = []
test_acc = []
for i in range(generations):
    for offset in range(0, num_train, batch_size):
        end = offset + batch_size
        rand_x = train_images[offset:end]
        rand_y = train_labels[offset:end]
        train_dict = {x_input: rand_x, y_target: rand_y, keep_prob:dropout}
        sess.run(train_step, feed_dict=train_dict)
        temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict=train_dict)
        temp_train_acc = get_accuracy(temp_train_preds, rand_y)
    if (i+1) % eval_every == 0:
        #eval_index = np.random.choice(len(test_images), size=test_size)
        eval_index = [d for d in range(num_test)]
        eval_x = test_images[eval_index]
        eval_y = test_labels[eval_index]
        test_dict = {test_input: eval_x, test_target: eval_y, keep_prob:1.0}
        temp_test_loss, test_preds = sess.run([test_loss1, test_prediction], feed_dict=test_dict)
        temp_test_acc = get_accuracy(test_preds, eval_y)
        # Record and print results
        train_loss.append(temp_train_loss)
        test_loss.append(temp_test_loss)
        train_acc.append(temp_train_acc)
        test_acc.append(temp_test_acc)
        acc_and_loss = [(i+1), temp_train_loss, temp_test_loss, temp_train_acc, temp_test_acc]
        acc_and_loss = [np.round(x,2) for x in acc_and_loss]
        print('Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f}% ({:.2f}%)'.format(*acc_and_loss))

inputs = {
    "x_input": x_input,
    "y_target": y_target,
    "keep_prob": keep_prob
}

outputs = {
    "model_output": eval_model_output,
    "prediction": eval_prediction
}

tf.saved_model.simple_save(sess, EXPORT_DIR+training_date, inputs, outputs)
print("Model saved")

# Matlotlib code to plot the loss and accuracies
eval_indices = range(0, generations, eval_every)
# Plot loss over time
plt.plot(eval_indices, train_loss, 'k-', label='Train Loss')
plt.plot(eval_indices, test_loss, 'r--', label='Test Loss')
plt.title('Softmax Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Softmax Loss')
plt.legend(loc='upper right')
plt.show()

# Plot train and test accuracy
plt.plot(eval_indices, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(eval_indices, test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# # Plot some samples
# # Plot the 6 of the last batch results:
# Nrows = 6
# Ncols = 7
# Total = Nrows * Ncols
# actuals = rand_y[0:Total]
# predictions = np.argmax(temp_train_preds, axis=1)[0:Total]
# images = np.squeeze(rand_x[0:Total])
#
# for i in range(Total):
#     plt.subplot(Nrows, Ncols, i+1)
#     plt.imshow(np.reshape(images[i], [32, 32, 3]), cmap='Greys_r')
#     plt.title('Actual: ' + str(actuals[i]) + ' Pred: ' + str(predictions[i]),
#               fontsize=10)
#     frame = plt.gca()
#     frame.axes.get_xaxis().set_visible(False)
#     frame.axes.get_yaxis().set_visible(False)
# plt.show()

# saver.restore(sess, './JPTSignClass.ckpt')
# print("Model restored")
# validate_size = len(valid_images)
# valid_dict = {test_input: valid_images[:validate_size], test_target: valid_labels[:validate_size], keep_prob:1.0}
# valid_model, valid_preds = sess.run([test_model_output, test_prediction], feed_dict=valid_dict)
# temp_valid_acc = get_accuracy(valid_preds, valid_labels[:validate_size])
# print(temp_valid_acc)
# check = valid_model[0] < 9
# if check.all():
#     print('not a sign')
# print(valid_model[0])

sess.close()
