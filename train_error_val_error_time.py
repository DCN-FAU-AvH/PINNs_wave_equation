import time

t0=time.process_time()
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

def func(x):
    return np.sin(np.pi * x[:, 0:1]) * np.cos(2*np.pi*x[:, 1:])
geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 2])


bc1 = dde.DirichletBC(geom, lambda x: 0, boundary_left)  #correct

bc2 = dde.DirichletBC(geom, lambda x: 0, boundary_right) #correct

bc3 = dde.DirichletBC(geom, initial_pos, boundary_bottom)  #correct

bc4 = dde.NeumannBC(geom, initial_velo, boundary_bottom)  #correct



data=[]
models=[]
net = dde.maps.FNN([2] + [50] * 4 + [1],"tanh", "Glorot uniform") 
xx=np.linspace(0,1,600)
tt=np.linspace(0,2,600)
list_all_data=list(product(*[list(xx), list(tt)], repeat=1))
gen_err=[]
train_err=[]
training_set_all=[]
compiling_time=[]
def exact_function(x):
    return math.sin(math.pi*x[0])*math.cos(2*math.pi*x[1])



for k in range(6):
    observe_x_1_val=np.vstack((np.linspace(0, 1, num=199-10*k), np.full((199-10*k), 2))).T
    observe_x_1 = np.vstack((np.linspace(0, 1, num=1+10*k), np.full((1+10*k), 2))).T
    observe_x_2_val=np.vstack((np.linspace(0, 1, num=199-10*k), np.full((199-10*k), 0))).T
    observe_x_2 = np.vstack((np.linspace(0, 1, num=1+10*k), np.full((1+10*k), 0))).T
    observe_x_3_val=np.vstack((np.full((199-10*k), 0),np.linspace(0, 2, num=199-10*k))).T
    observe_x_3 = np.vstack((np.full((1+10*k), 0),np.linspace(0, 2, num=1+10*k))).T
    observe_x_4_val=np.vstack((np.full((199-10*k), 1),np.linspace(0, 2, num=199-10*k))).T
    observe_x_4 = np.vstack((np.full((1+10*k), 1),np.linspace(0, 2, num=1+10*k))).T
    observe_x_5_val=np.vstack((np.full((199-10*k), 0.5),np.linspace(0, 2, num=199-10*k))).T
    observe_x_5=np.vstack((np.full((1+10*k), 0.5),np.linspace(0, 2, num=1+10*k))).T
    observe_x_6_val=np.vstack((np.linspace(0, 1, num=199-10*k), np.full((199-10*k), 1))).T
    observe_x_6=np.vstack((np.linspace(0, 1, num=1+10*k), np.full((1+10*k), 1))).T

    observe_x=np.array(list(observe_x_1)+list(observe_x_2)+list(observe_x_3)+list(observe_x_4)+list(observe_x_6)+list(observe_x_5))
   # print(observe_x)
    observe_final_val=np.array(list(observe_x_1_val)+list(observe_x_2_val)+list(observe_x_3_val)+list(observe_x_4_val)+list(observe_x_5_val)+list(observe_x_6_val))
    #print(len(observe_final_val)
   # observe_final_val1=[(observe_final_val.T[0][i], observe_final_val.T[1][i]) for i in range(len(observe_final_val.T[0]))

    observe_final_val1=[(observe_final_val.T[0][i], observe_final_val.T[1][i]) for i in range(len(observe_final_val.T[0]))]
   # print(observe_final_val1)   
    training_set=[(observe_x.T[0][i], observe_x.T[1][i]) for i in range(len(observe_x.T[0]))]  
   # print(training_set)                  
    gen_err_set=set(observe_final_val1)-set(training_set)
   # print(gen_err_set)                    
    l1=list(gen_err_set)
    validating_set=[np.asarray(l1[i]) for i in range(len(l1))]
    #print(validating_set)   
    observe_y=dde.icbc.PointSetBC(observe_x, func(observe_x), component=0)
    data = dde.data.PDE(geom, pde, [bc1, bc2, bc3, bc4, observe_y], num_domain=0, num_boundary=0,anchors=observe_x,num_test=1000, solution=func)    
    model = dde.Model(data, net)
    #models.append(model)
    model.compile("adam", lr=0.001, loss_weights=[1,1,1,1,1,1])
    model.train(epochs=6000)
    model.compile("L-BFGS-B", loss_weights=[1,1,1,1,1,1])
    losshistory, train_state=model.train()
    
    predicted_solution = np.ravel(model.predict(validating_set)) 
    exact_solution=np.asarray(list(map(exact_function, validating_set)))
 #   gen_error = (1/len(predicted_solution))*np.linalg.norm(predicted_solution - exact_solution)
    gen_error = (1/len(predicted_solution))*np.sum(predicted_solution**2 + exact_solution**2-2*predicted_solution*exact_solution)
    gen_err.append(gen_error)
    training_error=np.sum(np.array(losshistory.loss_train[-1]))
    train_err.append(training_error)
    compiling_time.append(time.process_time()-t0)
    

no_samples=[6+k*60 for k in range(6)]

fig = plt.figure() 
plt.plot(no_samples, gen_err, color="red",marker="o", linestyle="-")
#plot.plot(x, y, color="red", marker="o",  linestyle="--")
plt.xlabel("No. training samples")
plt.ylabel("Validation error")
plt.title("Validation error vs. no of training samples")
plt.show()
plt.savefig("Generr.png")


fig = plt.figure()  # plot of predicted state
plt.plot(no_samples, train_err, marker="o", linestyle="-")
plt.xlabel("No. training samples")
plt.ylabel("Training error")
plt.title("Training error vs. no of training samples")
plt.show()
plt.savefig("Trainerr.png")

           
                        
fig = plt.figure()  # plot of predicted state
plt.plot(no_samples, compiling_time, color="green",marker="o", linestyle="-")
plt.xlabel("No. training samples")
plt.ylabel("Computational time")
plt.title("Computational time vs. no of training samples")
plt.show()
plt.savefig("Time.png")                  
                        
                        