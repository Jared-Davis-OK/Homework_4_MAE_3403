import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Main Program

def main():

    # Parameters (CHANGE HERE IF DESIRED)

    x1 = 1.0      # Circle center x-coordinate
    y1 = 0.0      # Circle center y-coordinate
    radius = 4.0  # Circle radius (default gives 16)

    a = 0.5       # Parabola width
    b = 1.0       # Parabola vertical offset



    # System of Nonlinear Equations

    def system(vars):
        x, y = vars

        eq1 = (y - y1)**2 + (x - x1)**2 - radius**2   # Circle equation
        eq2 = y - (a * x**2 + b)                      # Parabola equation

        return [eq1, eq2]


    # Initial Guesses (two intersection points)

    guess1 = [-3, 3]
    guess2 = [3, 3]

    sol1 = fsolve(system, guess1)
    sol2 = fsolve(system, guess2)

    x_sol1, y_sol1 = sol1
    x_sol2, y_sol2 = sol2


    # Print Results

    print("Intersection Points:")
    print(f"Point 1: ({x_sol1:.6f}, {y_sol1:.6f})")
    print(f"Point 2: ({x_sol2:.6f}, {y_sol2:.6f})")



    # Plotting

    x_vals = np.linspace(-10, 10, 600)

    # Parabola
    y_parabola = a * x_vals**2 + b

    # Circle (parametric form to avoid sqrt warnings)
    theta = np.linspace(0, 2*np.pi, 600)
    x_circle = x1 + radius * np.cos(theta)
    y_circle = y1 + radius * np.sin(theta)

    plt.figure()
    plt.plot(x_vals, y_parabola, label="Parabola")
    plt.plot(x_circle, y_circle, label="Circle")

    plt.plot(x_sol1, y_sol1, 'ro')
    plt.plot(x_sol2, y_sol2, 'ro')

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Intersection of Circle and Parabola")
    plt.legend()
    plt.grid(True)

    plt.show()


# Run Program

if __name__ == "__main__":
    main()
