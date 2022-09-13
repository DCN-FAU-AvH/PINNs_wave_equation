# Approximating the 1D wave equation using Physics Informed Neural Networks (PINNs)

An implementation of Physics-Informed Neural Networks (PINNs) to solve various forward and inverse problems for the 1 dimensional wave equation.

<p align="center">
<img src="https://github.com/DCN-FAU-AvH/PINNs_wave_equation/blob/main/Icon.png" width="70%" height="70%" >
</p>

## Detailed explanation
[all_together_loss_time.py](https://github.com/DCN-FAU-AvH/PINNs_wave_equation/blob/main/all_together_loss_time.py) shows the test error-computational time dependency for different structures of neural networks.

[changing_nodes_test_loss.py](https://github.com/DCN-FAU-AvH/PINNs_wave_equation/blob/main/changing_nodes_test_loss.py) shows the test error-computational time dependency for neural network structures with different numbers of nodes. 

[control.py](https://github.com/DCN-FAU-AvH/PINNs_wave_equation/blob/main/control.py) solves the null controllability problem of the 1d wave equation. 

[first_case_no_damage.py](https://github.com/DCN-FAU-AvH/PINNs_wave_equation/blob/main/first_case_no_damage.py) solves the degenerating 1d wave equation when $a(x)=4$

[second_case_damage.py](https://github.com/DCN-FAU-AvH/PINNs_wave_equation/blob/main/second_case_damage.py) solves the degenerating 1d wave equation when $a(x)=8|x-0.5|$

[third_case_double_damage.py](https://github.com/DCN-FAU-AvH/PINNs_wave_equation/blob/main/third_case_double_damage.py) solves the degenerating 1d wave equation when $a(x)=16|x-0.5|^2$

[inverse_problem.py](https://github.com/DCN-FAU-AvH/PINNs_wave_equation/blob/main/inverse_problem.py) solves the inverse problem of the 1d wave equation

[test_loss_time.py](https://github.com/DCN-FAU-AvH/PINNs_wave_equation/blob/main/test_loss_time.py) shows the test error-computational time dependency for a specific structure of neural network

[train_error_val_error_time.py](https://github.com/DCN-FAU-AvH/PINNs_wave_equation/blob/main/train_error_val_error_time.py) displays the train error/validation error/computational time-size of training set dependencies 
 
[Wave_equation.py](https://github.com/DCN-FAU-AvH/PINNs_wave_equation/blob/main/Wave_equation.py) solves the 1d wave equation

[Wave_equation_otherBC](https://github.com/DCN-FAU-AvH/PINNs_wave_equation/blob/main/Wave_equation_otherBC.py) solves the 1d wave equation with Neumann boundary conditions 

[degenerate_wave.m](https://github.com/DCN-FAU-AvH/PINNs_wave_equation/blob/main/degenerate_wave.m) solves a wave equation $u_{tt}(t,x) + a(x) u_{xx}(t,x) = 0$ on $x \in (0,1)$ in which the stiffness is $a(x) = 4(2|x-\frac{1}{2}|)^\alpha$ by finite differences. (used for comparison)
