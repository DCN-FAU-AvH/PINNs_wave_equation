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
    #def __init__(self):
        #self.times = []
    def on_train_begin(self):
        self.times = []
    #    self.time0=[]  #everything started
   #     self.timef=[]   #everything ended
    def on_epoch_begin(self):
        self.epoch_time_start = time.process_time()
     #   self.time0.append(self.epoch_time_start)
    def on_epoch_end(self):
        self.times.append(time.process_time() - self.epoch_time_start)
      #  self.epoch_time_end = time.process_time()
       # self.timef.append(self.epoch_time_end)


time_callback = TimeHistory()
history=model.train(callbacks=[time_callback], epochs=10000)
def fun(x):
    x=np.array(x)
    return np.sum(x)
#print(len(time_callback.times))
chunks_1 = list(partition(1000, time_callback.times))
#print(time_callback.times)
time_epochs=list(map(fun, chunks_1))
print("Time epochs {}".format(time_epochs))
model.compile("L-BFGS-B")
print("Time epochs {}".format(time_epochs))



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


plt.plot(time_taken, a)
plt.xlabel("Computational time (s)")
plt.ylabel("Test error")
plt.title("Test error vs. Computational time")

print("Comp. time: {}".format(time_taken))
print("Error: {}".format(a))