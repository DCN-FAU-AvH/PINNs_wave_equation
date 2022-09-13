import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from itertools import product
import deepxde as dde
import math
import pathlib
import os

OUTPUT_DIRECTORY = pathlib.Path.cwd() / "results" / "linear_wave"
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
    is_on_boundary_left =on_boundary and np.isclose(x[0], 0)
    return is_on_boundary_left

#def boundary_right(x, on_boundary):  # boundary x=1
 #   is_on_boundary_right = on_boundary and np.isclose(x[0], 1)

   # return is_on_boundary_right

def boundary_bottom(x, on_boundary):  # boundary t=0
    is_on_boundary_bottom = (
        on_boundary
        and np.isclose(x[1], 0)
        and not np.isclose(x[0], 0)
        and not np.isclose(x[0], 1)
    )

    return is_on_boundary_bottom


def boundary_upper(x, on_boundary):  # boundary t=4
    is_on_boundary_upper = (
        on_boundary
        and np.isclose(x[1], 4) and not np.isclose(x[0], 0) and not np.isclose(x[0], 1)
       
    )

    return is_on_boundary_upper


geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 4])


bc1 = dde.DirichletBC(geom, lambda x: 0, boundary_left)  #correct

#bc2 = dde.DirichletBC(geom, lambda x: 0, boundary_right) #correct

bc2 = dde.DirichletBC(geom, initial_pos, boundary_bottom)  #correct

bc3 = dde.NeumannBC(geom, initial_velo, boundary_bottom)  #correct



bc4 = dde.DirichletBC(geom, lambda x: 0, boundary_upper)  #correct

bc5 = dde.NeumannBC(geom, initial_velo, boundary_upper)  #correct


data = dde.data.PDE(
    geom,
    pde,
    [bc1, bc2, bc3, bc4, bc5],
    num_domain=1000,
    num_boundary=700,
)
print("The training set is {}".format(data.train_x_all.T))
net = dde.maps.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")  #the input layer has size 2, there are 4 hidden layers of size 50 and one output layer of size 1

#Activation function is tanh; the weights are initially chosen to be uniformly distributed according to Glorat distribution

model = dde.Model(data, net)

model.compile("adam", lr=0.001)

model.train(epochs=10000)

model.compile("L-BFGS-B")

losshistory, train_state = model.train()

dde.saveplot(
    losshistory, train_state, issave=True, isplot=True, output_dir=OUTPUT_DIRECTORY
)


X = np.linspace(0, 1, 600)
t = np.linspace(0, 4, 2400)

X_repeated = np.repeat(X, t.shape[0])
t_tiled = np.tile(t, X.shape[0])
XX = np.vstack((X_repeated, t_tiled)).T

state_predict = model.predict(XX).T
state_predict_M = state_predict.reshape((600, 2400)).T

Xx, Tt = np.meshgrid(np.linspace(0, 1, 600), np.linspace(0, 4, 2400))

fig = plt.figure()  # plot of predicted state
ax = plt.axes(projection="3d")
surf = ax.plot_surface(
    Xx, Tt, state_predict_M, cmap="hsv_r", linewidth=0, antialiased=False
)

ax.set_title("PINN state ")
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
fig.colorbar(surf, shrink=0.6, aspect=10)

plt.show()



tt = np.linspace(0, 4, 40000)
xx1 = np.ones_like(tt)
X1 = np.vstack((xx1, tt)).T


control_predict_1 = np.ravel(model.predict(X1))  # predicted control1
control_predict_1[0]=0



fig=plt.figure()
plt.plot(tt, control_predict_1, label="Control function u(t)", color="r")
plt.xlabel("t")
plt.ylabel("u(t)")
plt.legend()





