import numpy as np
import matplotlib
import math
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
from scipy.interpolate import interp2d

plot_boundary = 7
num_plot_points = 30
d = 2
lr = 1e-6
radius = 1e-4
with_flips = True

x_init = np.array([-6.2, 8.3])

x_star = np.array([-4.0, 3.0])

def get_trajectory(x0, beta_inv=1, eta=1, iters=100):
    trajectory = [x0]
    for _ in range(iters):
        noise = np.sqrt(2 * beta_inv * eta) * np.random.normal(x0.shape)
        x0 = x0 - eta * h(x0, x_star) + noise
        trajectory.append(x0)
    return trajectory


def check_property(x, y, x_star):
    def update_point(x):
        x_new = x - lr * h(x, x_star)
        return x_new

    # if with_flips:    
    #     subcase1_1 = np.linalg.norm(update_point(x) - update_point(y)) < np.linalg.norm(x - y)
    #     subcase1_2 = np.linalg.norm(update_point(-x) - update_point(-y)) < np.linalg.norm(x - y)
    #     case1 = subcase1_1 and subcase1_2

    #     subcase2_1 = np.linalg.norm(update_point(-x) - update_point(y)) < np.linalg.norm(x - y)
    #     subcase2_2 = np.linalg.norm(update_point(x) - update_point(-y)) < np.linalg.norm(x - y)
    #     case2 = subcase2_1 and subcase2_2
    #     res = True if (case1 or case2) else False          
    # else:
    res = True if np.linalg.norm(update_point(x) - update_point(y)) < 1.0 * np.linalg.norm(x - y) else False
    
    return res

def set_proper_limits(cos_output):
    return min(max(-1, cos_output), 1)

def g(theta):
    cos_output = ((np.pi - theta) * np.cos(theta) + np.sin(theta)) / np.pi
    return np.arccos(set_proper_limits(cos_output))

def g_proxy(theta):
    return theta * (1 - theta / (2 * np.pi))

def h(x, x_star):
    return x / 2**d - h_bar(x, x_star)

def calculate_theta_bar(i, x, y):
    if i == 0:
        if d == 1:
            return 0
        else:
            angle = np.arccos(max(-1, min((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y)), 1), -1))
            return angle

    else:
        return g(calculate_theta_bar(i - 1, x, y))

def calculate_vector_hat(x):
    return x / np.linalg.norm(x)

def h_bar(x, x_star):
    def first_term():
        prodd = 1
        for i in range(d):
            prodd *= (np.pi - calculate_theta_bar(i, x, x_star)) / np.pi
        return (prodd * x_star) / 2**d

    def second_term():
        summ = np.zeros(d)
        for i in range(d):
            sum_factor = np.sin(calculate_theta_bar(i, x, x_star)) / np.pi
            prodd = 1
            for j in range(i + 1, d):
                prodd *= (np.pi - calculate_theta_bar(j, x, x_star)) / np.pi
            summ += sum_factor * prodd * np.linalg.norm(x_star) * calculate_vector_hat(x)
        return summ / 2**d
    
    return first_term() + second_term()


def plot_line(x_star):
    a = x_star[1] / x_star[0]
    x1 = np.linspace(-x_star, x_star, 3)
    y1 = a * x1
    plt.plot(x1, y1, color='green')

def calculate_radius(point):
    return math.sqrt(point[0] ** 2 + point[1] ** 2)

def run_dynamical_system(x_init, t=10):
    x_curr = x_init
    trajectories = []
    for i in range(t):
        x_curr = g(x_curr)
        trajectories.append(x_curr)
    return trajectories



if __name__ == '__main__':
    angles = np.linspace(0, np.pi, 100)
    g_angles = [g(x) for x in angles]
    g_proxy_angles = [g_proxy(x) for x in angles]
    
    # plt.plot(angles, g_angles)
    # plt.plot(angles, g_proxy_angles)
    # plt.legend(['g', 'g_proxy'])
    # plt.show()

    # trajectory = run_dynamical_system(-0.5)
    # plt.plot(trajectory)
    # plt.show()


    # plot vector field
    y = [g(x) for x in angles]
    fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
    x1 = np.linspace(-plot_boundary, plot_boundary, num_plot_points)
    x2 = np.linspace(-plot_boundary, plot_boundary, num_plot_points)

    
    Z = []
    axis_points = []
    for x1_item in x1:
        for x2_item in x2:
            axis_points.append(np.array([x1_item, x2_item]))
            Z.append(h(axis_points[-1], x_star))

    x = np.array([x[0] for x in axis_points])
    y = np.array([x[1] for x in axis_points])
    
    xi = np.linspace(x.min(), x.max(), x.size)
    yi = np.linspace(y.min(), y.max(), y.size)
    
    z1 = np.array([-z[0] for z in Z])
    z2 = np.array([-z[1] for z in Z])

    uCi = interp2d(x, y, z1)(xi, yi)
    vCi = interp2d(x, y, z2)(xi, yi)
    ax.streamplot(xi, yi, uCi, vCi, linewidth=1, cmap=plt.cm.inferno, density=2, arrowstyle='->', arrowsize=1.5)
    # plt.quiver([x[0] for x in axis_points], [x[1] for x in axis_points],
    #     [-z[0] for z in Z], [-z[1] for z in Z])
    plt.scatter(x_star[0], x_star[1], color='green')
    plt.text(x_star[0] + 0.08, x_star[1] + 0.08, "$x_{star}$", fontsize=18, color='green')

    #
    circle = plt.Circle((0, 0), radius=1.6, color='r', fill=False)
    ax.add_patch(circle)
    plt.text(1.3, 1.3, "Non-smooth region", fontsize=12, color='red')

    #
    circle = plt.Circle(x_star, radius=2, color='green', fill=False)
    ax.add_patch(circle)
    plt.text(x_star[0] + 1.5, x_star[1] + 1.5, "Strongly convex region", fontsize=12, color='green')

    ax.set_aspect('equal')

    # ax.tick_params(bottom=False, top=False, left=False, right=False)
    plt.show()


    # # run dynamical system
    # trajectory = get_trajectory(x_init, iters=10000, beta_inv=1/1e6, eta=1e-2)

    # # green: trajectory
    # plt.scatter([x[0] for x in trajectory], [x[1] for x in trajectory], color='green')
    # # purple: where it started
    # plt.scatter(trajectory[0][0], trajectory[0][1], color='purple')
    # # red: where it ended
    # plt.scatter(trajectory[-1][0], trajectory[-1][1], color='red')


    # plt.show()
