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
bc1 = dde.NeumannBC(geom, lambda x: 0, boundary_left)  #correct
#bc2 = dde.DirichletBC(geom, lambda x: 0, boundary_right) #correct
bc2 = dde.NeumannBC(geom, lambda x: 0, boundary_right) #correct
bc3 = dde.DirichletBC(geom, initial_pos, boundary_bottom)  #correct

bc4 = dde.NeumannBC(geom, initial_velo, boundary_bottom)  #correct




data = dde.data.PDE(
    geom,
    pde,
    [bc1, bc2, bc3, bc4],
    num_domain=200,
    num_boundary=100,
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

xx=np.linspace(0,1,100)
tt=np.linspace(0,2,200)
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
def exact_function(x):
    return math.sin(math.pi*x[0])*math.cos(2*math.pi*x[1])

#Generalization error:
exact_solution=np.asarray(list(map(exact_function, validating_set)))
print("Exact solution {}".format(exact_solution))
gen_error = (1/len(predicted_solution))*np.linalg.norm(predicted_solution - exact_solution)
print("Generalization error is {}".format(gen_error))
relative_error = np.linalg.norm(predicted_solution - exact_solution) / np.linalg.norm(exact_solution)
print("Relative error is {}".format(relative_error))
#Training error:
exact_solution_training=np.asarray(list(map(exact_function, data.train_x_all)))
predicted_solution_training=np.ravel(model.predict(data.train_x_all)) 
training_error=(1/len(predicted_solution_training))*np.linalg.norm(predicted_solution_training- exact_solution_training)
X = np.linspace(0, 1, 100)
t = np.linspace(0, 2, 200)

X_repeated = np.repeat(X, t.shape[0])
t_tiled = np.tile(t, X.shape[0])
XX = np.vstack((X_repeated, t_tiled)).T

state_predict = model.predict(XX).T
state_predict_M = state_predict.reshape((100, 200)).T

Xx, Tt = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 2, 200))

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


def explicit_state(x,t):
    return np.sin(math.pi*x)*np.cos(2*math.pi*t)


state_exact = explicit_state(Xx, Tt)  # computation of exact state

fig = plt.figure()  # plot of exact state
ax = plt.axes(projection="3d")
surf2 = ax.plot_surface(
    Xx, Tt, state_exact, cmap="hsv_r", linewidth=0, antialiased=False
)

ax.set_title("Exact state ")
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")

fig.colorbar(surf2, shrink=0.6, aspect=10)

plt.show()



fig = plt.figure()  # plot of the difference between exact and PINN state
ax = plt.axes(projection="3d")
surf2 = ax.plot_surface(
    Xx, Tt, state_exact-state_predict_M, cmap="hsv_r", 
    linewidth=0, antialiased=False
)

ax.set_title("Difference of the exact and PINN state")
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")

fig.colorbar(surf2, shrink=0.6, aspect=10)

plt.show()


#Plot the training set and validation set
fig = plt.figure()

plt.plot(data.train_x_all.T[0], data.train_x_all.T[1],"o")
#plt.plot(validating_set.T[0], validating_set.T[1], "r", label="Validation set")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Training set")
#plt.legend()


print("Training set {}".format(data.train_x_all))
#print("Validation set {}".format(validating_set))














