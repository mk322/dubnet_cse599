from dubnet import *

def fc_net():
    l = [   make_connected_layer(3072, 72),
            make_activation_layer(RELU),

            make_connected_layer(72, 512),
            make_activation_layer(RELU),

            make_connected_layer(512, 1104),
            make_activation_layer(RELU),

            make_connected_layer(1104, 256),
            make_activation_layer(RELU),

            make_connected_layer(256, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)


def conv_net():
    l = [   make_convolutional_layer(3, 8, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(3, 2),
            make_convolutional_layer(8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(3, 2),
            make_convolutional_layer(16, 32, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(3, 2),
            make_convolutional_layer(32, 64, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(3, 2),
            make_connected_layer(256, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 1000
rate = .01
momentum = .9
decay = .005

m = fc_net()
conv = True

if conv: 
    m = conv_net()

print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
#
# ConvNet: training accuracy = 53.45% ; test accuracy = 52.84%
# Fully connected network: training accurracy = 42.25%; test accuracy= 41.65%
#
# The fully connected network tries to find a relationship or features by using all pixels and channels of an image.
# In fact, for the image classification task, only the neighboring pixels give the most useful information about a 
# particular pixel and convolutions are best suited for this. The ConvNet uses the same number of operations more efficiently
# by using convolutions. Thus, the accuracy of ConvNet is higher than the accuracy of the fully connected network.