import network1
import mnist_loader1
training_data, test_data, _ = mnist_loader1.load_data_wrapper();

net=network1.Network([784,30,10]);
net.SGD(training_data, 30, 10, 0.01, 0.1, test_data=test_data);