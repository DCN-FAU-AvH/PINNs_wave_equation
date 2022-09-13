import numpy as np
from itertools import product
import deepxde as dde
import math
import pathlib
import os
import tensorflow as tf
import matplotlib.pyplot as plt
OUTPUT_DIRECTORY = pathlib.Path.cwd() / "results" / "linear_wave"
if not OUTPUT_DIRECTORY.exists():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)



#first case a(x)=1

#def a(x,y):
 #   if x<0.5:
  #      return 0.5-x
  #  else:
   #     return x-0.5
        


def pde(x, y):  # wave equation
    dy_tt = dde.grad.hessian(y, x, i=1, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    
    return dy_tt - 16*abs(0.5-x[:,0:1])**2*dy_xx


def initial_pos(x):  # initial position

    return np.sin((np.pi * x[:, 0:1])/2)


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


geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 2])


#bc1 = dde.DirichletBC(geom, lambda x: 0, boundary_left)  #correct
bc1 = dde.DirichletBC(geom, lambda x: 0, boundary_left)  #correct
#bc2 = dde.DirichletBC(geom, lambda x: 0, boundary_right) #correct
bc2 = dde.NeumannBC(geom, lambda x: 0, boundary_right) #correct
bc3 = dde.DirichletBC(geom, initial_pos, boundary_bottom)  #correct

bc4 = dde.NeumannBC(geom, initial_velo, boundary_bottom)  #correct




data = dde.data.PDE(
    geom,
    pde,
    [bc1, bc2, bc3, bc4],
    num_domain=400,
    num_boundary=800,
)
print("The training set is {}".format(data.train_x_all.T))
net = dde.maps.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")  #the input layer has size 2, there are 4 hidden layers of size 50 and one output layer of size 1

#Activation function is tanh; the weights are initially chosen to be uniformly distributed according to Glorat distribution

model = dde.Model(data, net)

model.compile("adam", lr=0.001)

model.train(epochs=7000)

model.compile("L-BFGS-B")

losshistory, train_state = model.train()

dde.saveplot(
    losshistory, train_state, issave=True, isplot=True, output_dir=OUTPUT_DIRECTORY
)



#Post-processing: error analysis and figures

xx=np.linspace(0,1,1000)
tt=np.linspace(0,2,2000)
#X_repeated = np.repeat(xx, tt.shape[0])
#t_tiled = np.tile(tt, xx.shape[0])
#X=np.vstack((X_repeated, t_tiled)).T
list_all_data=list(product(*[list(xx), list(tt)], repeat=1))
#print("List all data is {}".format(list_all_data))
#X=[np.asarray(list_all_data[k]) for k in range(len(list_all_data))]
#X=np.asarray(X)
print("List {}".format(list_all_data))
training_set=[(data.train_x_all.T[0][k], data.train_x_all.T[1][k]) for k in range(len(data.train_x_all.T[0]))]

gen_error_set=set(list_all_data)-set(training_set)
l1=list(gen_error_set)
print("li is {}".format(l1))
validating_set=[np.asarray(l1[k]) for k in range(len(l1))]
print("Dania {}".format(validating_set))
validating_set=np.asarray(validating_set)
print("Dania 2 {}".format(validating_set))
#print("Validating set {}".format(validating_set))
predicted_solution = np.ravel(model.predict(validating_set)) 
print("Pred solution {}".format(predicted_solution))

X = np.linspace(0, 1, 1000)
t = np.linspace(0, 2, 2000)


X_repeated = np.repeat(X, t.shape[0])
t_tiled = np.tile(t, X.shape[0])
XX = np.vstack((X_repeated, t_tiled)).T

state_predict = model.predict(XX).T
state_predict_M = state_predict.reshape((1000, 2000)).T

Xx, Tt = np.meshgrid(np.linspace(0, 1, 1000), np.linspace(0, 2, 2000))

fig = plt.figure()  # plot of predicted state
ax = plt.axes(projection="3d")
surf = ax.plot_surface(
    Xx, Tt, state_predict_M, cmap="hsv_r", linewidth=0, antialiased=False
)

ax.set_title(r"PINN state: $a(x)=16|x-\frac{1}{2}|^{2}$")
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
fig.colorbar(surf, shrink=0.6, aspect=10)

plt.show()
























