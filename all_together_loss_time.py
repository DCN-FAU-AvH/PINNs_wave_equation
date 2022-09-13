import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from itertools import product
import deepxde as dde
import math
import pathlib
import os
import time
#import tf.keras.callbacks.EarlyStopping
import tensorflow as tf
from toolz import partition
OUTPUT_DIRECTORY = pathlib.Path.cwd() / "results" / "wave_equation"
if not OUTPUT_DIRECTORY.exists():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


def pde(x, y):  # wave equation
    dy_tt = dde.grad.hessian(y, x, i=1, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)

    return dy_tt - 4*dy_xx


def initial_pos(x):  # initial position

    return np.sin(np.pi * x[:, 0:1])


def initial_velo(x):  # initial velocity

    return 0.0


def boundary_left(x, on_boundary):  # boundary x=0
    is_on_boundary_left = on_boundary and np.isclose(x[0], 0)

    return is_on_boundary_left

def boundary_right(x, on_boundary):  # boundary x=1
    is_on_boundary_right = on_boundary and np.isclose(x[0], 1)

    return is_on_boundary_right

def boundary_bottom(x, on_boundary):  # boundary t=0
    is_on_boundary_bottom = (
        on_boundary
        and np.isclose(x[1], 0)
        and not np.isclose(x[0], 0)
        and not np.isclose(x[0], 1)
    )

    return is_on_boundary_bottom


def boundary_upper(x, on_boundary):  # boundary t=2
    is_on_boundary_upper = (
        on_boundary
        and np.isclose(x[1], 2)
        and not np.isclose(x[0], 0)
        and not np.isclose(x[0], 1)
    )

    return is_on_boundary_upper

def func(x):
    return np.sin(np.pi * x[:, 0:1]) * np.cos(2*np.pi*x[:, 1:])

geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 2])


bc1 = dde.DirichletBC(geom, lambda x: 0, boundary_left)  #correct

bc2 = dde.DirichletBC(geom, lambda x: 0, boundary_right) #correct

bc3 = dde.DirichletBC(geom, initial_pos, boundary_bottom)  #correct

bc4 = dde.NeumannBC(geom, initial_velo, boundary_bottom)  #correct




data = dde.data.TimePDE(
    geom,
    pde,
    [bc1, bc2, bc3, bc4],
    num_domain=200,
    num_boundary=100, solution=func, num_test=250
)
print("The training set is {}".format(data.train_x_all.T))
net = dde.maps.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")  #the input layer has size 2, there are 4 hidden layers of size 50 and one output layer of size 1

#Activation function is tanh; the weights are initially chosen to be uniformly distributed according to Glorat distribution

#callback=dde.callbacks.EarlyStopping(min_delta=0.001, patience=1000, monitor='loss_test')

model = dde.Model(data, net)

model.compile("adam", lr=0.001)

class TimeHistory(dde.callbacks.Callback):
   
    def on_train_begin(self):
        self.times = []
   
    def on_epoch_begin(self):
        self.epoch_time_start = time.process_time()
    
    def on_epoch_end(self):
        self.times.append(time.process_time() - self.epoch_time_start)
  


time_callback = TimeHistory()
history=model.train(callbacks=[time_callback], epochs=10000)
def fun(x):
    x=np.array(x)
    return np.sum(x)
chunks_1 = list(partition(1000, time_callback.times))
time_epochs=list(map(fun, chunks_1))
model.compile("L-BFGS-B")

losshistory, train_state = model.train()

dde.saveplot(
    losshistory, train_state, issave=True, isplot=True, output_dir=OUTPUT_DIRECTORY
)
print("Loss history gives {}".format(losshistory.loss_train))


print("Loss history gives {}".format(losshistory.metrics_test))


a=list(map(np.sum, losshistory.loss_test))[0:10]
print("Total loss is {}".format(a))


print("This are the times {}".format(time_epochs))
print("This are the test losses {}".format(losshistory.loss_test[0:10]))

time_taken=[]
b=0
for i in range(len(time_epochs)):
    b=b+time_epochs[i]
    time_taken.append(b)


#plt.plot(time_taken, a, label="[50] x 4")
#plt.xlabel("Computational time (s)")
#plt.ylabel("Test error")
#plt.title("Test error vs. Computational time")

#net_1 = dde.maps.FNN([2] + [100] * 2 + [1], "tanh", "Glorot uniform")  #the input layer has size 2, there are 4 hidden layers of size 50 and one output layer of size 1
#net_2 = dde.maps.FNN([2] + [100] * 4 + [1], "tanh", "Glorot uniform")  #the input layer has size 2, there are 4 hidden layers of size 50 and one output layer of size 1
#net_3 = dde.maps.FNN([2] + [20] * 10 + [1], "tanh", "Glorot uniform")  #the input layer has size 2, there are 4 hidden layers of size 50 and one output layer of size 1
#4444444--------------------------------------------------------
net_4 = dde.maps.FNN([2] + [5] * 2 + [1], "tanh", "Glorot uniform")
net_5 = dde.maps.FNN([2] + [5] * 6 + [1], "tanh", "Glorot uniform")
net_6 = dde.maps.FNN([2] + [5] * 10 + [1], "tanh", "Glorot uniform")
net_7 = dde.maps.FNN([2] + [8] * 10 + [1], "tanh", "Glorot uniform")
net_8 = dde.maps.FNN([2] + [10] * 10 + [1], "tanh", "Glorot uniform")
net_9 = dde.maps.FNN([2] + [10] * 15 + [1], "tanh", "Glorot uniform")
#net_10 = dde.maps.FNN([2] + [20] * 10 + [1], "tanh", "Glorot uniform")


model_1 = dde.Model(data, net_4)

model_1.compile("adam", lr=0.001)

class TimeHistory_1(dde.callbacks.Callback):
   
    def on_train_begin(self):
        self.times = []
   
    def on_epoch_begin(self):
        self.epoch_time_start = time.process_time()
    
    def on_epoch_end(self):
        self.times.append(time.process_time() - self.epoch_time_start)

time_callback_1 = TimeHistory_1()
history=model_1.train(callbacks=[time_callback_1], epochs=10000)
def fun(x):
    x=np.array(x)
    return np.sum(x)
chunks_11 = list(partition(1000, time_callback_1.times))
time_epochs_1=list(map(fun, chunks_11))
model_1.compile("L-BFGS-B")

losshistory_1, train_state_1 = model_1.train()

dde.saveplot(
    losshistory_1, train_state_1, issave=True, isplot=True, output_dir=OUTPUT_DIRECTORY
)
#print("Loss history gives {}".format(losshistory.loss_train))


#print("Loss history gives {}".format(losshistory.metrics_test))


a1=list(map(np.sum, losshistory_1.loss_test))[0:10]
#print("Total loss is {}".format(a))


#print("This are the times {}".format(time_epochs))
#print("This are the test losses {}".format(losshistory.loss_test[0:10]))

time_taken1=[]
b=0
for i in range(len(time_epochs_1)):
    b=b+time_epochs_1[i]
    time_taken1.append(b)
    
#plt.plot(time_taken1, a1, label="[100] x 2")
##555555555555------------------------------------------------------------------

model_2 = dde.Model(data, net_5)

model_2.compile("adam", lr=0.001)

class TimeHistory_2(dde.callbacks.Callback):
   
    def on_train_begin(self):
        self.times = []
   
    def on_epoch_begin(self):
        self.epoch_time_start = time.process_time()
    
    def on_epoch_end(self):
        self.times.append(time.process_time() - self.epoch_time_start)

time_callback_2 = TimeHistory_2()
history=model_2.train(callbacks=[time_callback_2], epochs=10000)
def fun(x):
    x=np.array(x)
    return np.sum(x)
chunks_12 = list(partition(1000, time_callback_2.times))
time_epochs_2=list(map(fun, chunks_12))
model_2.compile("L-BFGS-B")

losshistory_2, train_state_2 = model_2.train()

dde.saveplot(
    losshistory_2, train_state_2, issave=True, isplot=True, output_dir=OUTPUT_DIRECTORY
)
#print("Loss history gives {}".format(losshistory.loss_train))


#print("Loss history gives {}".format(losshistory.metrics_test))


a2=list(map(np.sum, losshistory_2.loss_test))[0:10]
#print("Total loss is {}".format(a))


#print("This are the times {}".format(time_epochs))
#print("This are the test losses {}".format(losshistory.loss_test[0:10]))

time_taken2=[]
b=0
for i in range(len(time_epochs_2)):
    b=b+time_epochs_2[i]
    time_taken2.append(b)
    
#plt.plot(time_taken2, a2, label="[100] x 4")

#666666666666666666--------------------------------------------------------------
model_3 = dde.Model(data, net_6)

model_3.compile("adam", lr=0.001)

class TimeHistory_3(dde.callbacks.Callback):
   
    def on_train_begin(self):
        self.times = []
   
    def on_epoch_begin(self):
        self.epoch_time_start = time.process_time()
    
    def on_epoch_end(self):
        self.times.append(time.process_time() - self.epoch_time_start)

time_callback_3 = TimeHistory_3()
history=model_3.train(callbacks=[time_callback_3], epochs=10000)
def fun(x):
    x=np.array(x)
    return np.sum(x)
chunks_13 = list(partition(1000, time_callback_3.times))
time_epochs_3=list(map(fun, chunks_13))
model_3.compile("L-BFGS-B")

losshistory_3, train_state_3 = model_3.train()

dde.saveplot(
    losshistory_3, train_state_3, issave=True, isplot=True, output_dir=OUTPUT_DIRECTORY
)
#print("Loss history gives {}".format(losshistory.loss_train))


#print("Loss history gives {}".format(losshistory.metrics_test))


a3=list(map(np.sum, losshistory_3.loss_test))[0:10]
#print("Total loss is {}".format(a))


#print("This are the times {}".format(time_epochs))
#print("This are the test losses {}".format(losshistory.loss_test[0:10]))

time_taken3=[]
b=0
for i in range(len(time_epochs_3)):
    b=b+time_epochs_3[i]
    time_taken3.append(b)
    
    
    
    
    
    
    
#777777777777777777-----------------------------------------

model_4 = dde.Model(data, net_7)

model_4.compile("adam", lr=0.001)

class TimeHistory_4(dde.callbacks.Callback):
   
    def on_train_begin(self):
        self.times = []
   
    def on_epoch_begin(self):
        self.epoch_time_start = time.process_time()
    
    def on_epoch_end(self):
        self.times.append(time.process_time() - self.epoch_time_start)

time_callback_4 = TimeHistory_4()
history=model_4.train(callbacks=[time_callback_4], epochs=10000)
def fun(x):
    x=np.array(x)
    return np.sum(x)
chunks_14 = list(partition(1000, time_callback_4.times))
time_epochs_4=list(map(fun, chunks_14))
model_4.compile("L-BFGS-B")

losshistory_4, train_state_4 = model_4.train()

dde.saveplot(
    losshistory_4, train_state_4, issave=True, isplot=True, output_dir=OUTPUT_DIRECTORY
)
#print("Loss history gives {}".format(losshistory.loss_train))


#print("Loss history gives {}".format(losshistory.metrics_test))


a4=list(map(np.sum, losshistory_4.loss_test))[0:10]
#print("Total loss is {}".format(a))


#print("This are the times {}".format(time_epochs))
#print("This are the test losses {}".format(losshistory.loss_test[0:10]))

time_taken4=[]
b=0
for i in range(len(time_epochs_4)):
    b=b+time_epochs_4[i]
    time_taken4.append(b)
    

#888888888888888888888888------------------------------------------------------



model_5 = dde.Model(data, net_8)

model_5.compile("adam", lr=0.001)

class TimeHistory_5(dde.callbacks.Callback):
   
    def on_train_begin(self):
        self.times = []
   
    def on_epoch_begin(self):
        self.epoch_time_start = time.process_time()
    
    def on_epoch_end(self):
        self.times.append(time.process_time() - self.epoch_time_start)

time_callback_5 = TimeHistory_5()
history=model_5.train(callbacks=[time_callback_5], epochs=10000)
def fun(x):
    x=np.array(x)
    return np.sum(x)
chunks_15 = list(partition(1000, time_callback_5.times))
time_epochs_5=list(map(fun, chunks_15))
model_5.compile("L-BFGS-B")

losshistory_5, train_state_5 = model_5.train()

dde.saveplot(
    losshistory_5, train_state_5, issave=True, isplot=True, output_dir=OUTPUT_DIRECTORY
)
#print("Loss history gives {}".format(losshistory.loss_train))


#print("Loss history gives {}".format(losshistory.metrics_test))


a5=list(map(np.sum, losshistory_5.loss_test))[0:10]
#print("Total loss is {}".format(a))


#print("This are the times {}".format(time_epochs))
#print("This are the test losses {}".format(losshistory.loss_test[0:10]))

time_taken5=[]
b=0
for i in range(len(time_epochs_5)):
    b=b+time_epochs_5[i]
    time_taken5.append(b)
    


#9999999999999999999999------------------------------------------------------------------

model_6 = dde.Model(data, net_9)

model_6.compile("adam", lr=0.001)

class TimeHistory_6(dde.callbacks.Callback):
   
    def on_train_begin(self):
        self.times = []
   
    def on_epoch_begin(self):
        self.epoch_time_start = time.process_time()
    
    def on_epoch_end(self):
        self.times.append(time.process_time() - self.epoch_time_start)

time_callback_6 = TimeHistory_6()
history=model_6.train(callbacks=[time_callback_6], epochs=10000)
def fun(x):
    x=np.array(x)
    return np.sum(x)
chunks_16 = list(partition(1000, time_callback_6.times))
time_epochs_6=list(map(fun, chunks_16))
model_6.compile("L-BFGS-B")

losshistory_6, train_state_6 = model_6.train()

dde.saveplot(
    losshistory_6, train_state_6, issave=True, isplot=True, output_dir=OUTPUT_DIRECTORY
)
#print("Loss history gives {}".format(losshistory.loss_train))


#print("Loss history gives {}".format(losshistory.metrics_test))


a6=list(map(np.sum, losshistory_6.loss_test))[0:10]
#print("Total loss is {}".format(a))


#print("This are the times {}".format(time_epochs))
#print("This are the test losses {}".format(losshistory.loss_test[0:10]))

time_taken6=[]
b=0
for i in range(len(time_epochs_6)):
    b=b+time_epochs_6[i]
    time_taken6.append(b)











    
plt.plot(time_taken3, a3, label="50 nodes")
plt.plot(time_taken4, a4, label="80 nodes")
plt.plot(time_taken5, a5, label="100 nodes")
plt.plot(time_taken6, a6, label="150 nodes")
plt.plot(time_taken, a, label="200 nodes")
plt.plot(time_taken2, a2, label="30 nodes")
plt.plot(time_taken1, a1, label="10 nodes")
plt.xlabel("Computational time (s)")
plt.ylabel("Test error")
plt.title("Test error vs. Computational time")

plt.legend()




















