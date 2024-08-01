import numpy as np
import matplotlib.pyplot as plt

def plot_direction_field(f, xmin, xmax, ymin, ymax, xstep, ystep, unitary=False, plot_streamlines=False):
    """
    Plots the direction field associated with a first-order differential equation dy/dx = f(x, y).
    
    Parameters:
        f (function): The function f(x, y) representing the differential equation.
        xmin (float): Minimum value of x for the plot.
        xmax (float): Maximum value of x for the plot.
        ymin (float): Minimum value of y for the plot.
        ymax (float): Maximum value of y for the plot.
        xstep (float): Step size for x-axis.
        ystep (float): Step size for y-axis.
        unitary (bool): If True, plots the unitary vector field. Defaults to False.
        plot_streamlines (bool): If True, plots the streamlines (solution curves). Defaults to False.
    
    Returns:
        None: Displays the plot.
    """
    x = np.linspace(xmin, xmax, int((xmax - xmin) / xstep))
    y = np.linspace(ymin, ymax, int((ymax - ymin) / ystep))
    X, Y = np.meshgrid(x, y)
    U = np.ones_like(X)
    V = f(X, Y)
    
    if unitary:
        N = np.sqrt(U**2 + V**2)
        U = U / N
        V = V / N
    
    plt.figure(figsize=(10, 7))
    plt.quiver(X, Y, U, V, color='r')
    
    if plot_streamlines:
        plt.streamplot(X, Y, U, V, color='b')
    
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Campo de Direcciones')
    plt.grid(True)
    plt.show()

# Ejemplo 3: Campo de direcciones para dy/dx = y^2 - y
def f1(x, y):
    return y**2 - y

plot_direction_field(f1, xmin=-5, xmax=5, ymin=-5, ymax=5, xstep=0.5, ystep=0.5, unitary=False, plot_streamlines=True)

def f2(x, y):
    return -x*y

plot_direction_field(f2, xmin=-5, xmax=5, ymin=-5, ymax=5, xstep=0.5, ystep=0.5, unitary=False, plot_streamlines=True)
