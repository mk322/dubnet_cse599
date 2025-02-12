from dubnet import *
def conv_net():
    l = [   make_convolutional_layer(3, 8, 3, 1),
            make_batchnorm2d_layer(8),
            make_activation_layer(RELU),
            make_maxpool_layer(3, 2),
            make_convolutional_layer(8, 16, 3, 1),
            make_batchnorm2d_layer(16),
            make_activation_layer(RELU),
            make_maxpool_layer(3, 2),
            make_convolutional_layer(16, 32, 3, 1),
            make_batchnorm2d_layer(32),
            make_activation_layer(RELU),
            make_maxpool_layer(3, 2),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]

    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 500
rate = .01
momentum = .9
decay = .005

#m = conn_net()
m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# 7.6 Question: What do you notice about training the convnet with/without batch normalization? How does it affect convergence? How does it affect what magnitude of learning rate you can use? Write down any observations from your experiments:
# Without batch normaliztion:
# training accuracy: 0.49685999751091003   test accuracy: 0.49570000171661377
# With bach normalization:
# training accuracy:    test accuracy:
#
#


