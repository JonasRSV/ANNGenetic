from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import ann


def function_to_maximize(X, Y):
    return -1 * (X**2 / 40 + Y**2 / 40)


generation_colors = ["#330000", "#880000", "#FF0000", "#333300", "#888800", "#FFFF00",
                     "#333333", "#888888", "#330033", "#880088", "#FF00FF", "#008800",
                     "#00FF00", "#000088", "#0000FF"]

generations = len(generation_colors)

# Random Input for giggles

network = ann.ANN(1)

# Some Layers so that the network can learn
network.add_layer(ann.Layer(6, act=ann.sigmoid))

# Output, Should eventually become 0, 0 since that maximizes the function
network.add_layer(ann.Layer(2, act=ann.sigmoid))

# 10 Family members
ga = ann.Genetic(10)
ga.create_family(network)

ri = np.array([1])

fig = plt.figure()
ax = fig.gca(projection='3d')

for gc in generation_colors:
    xs, ys, zs = [], [], []
    evl = []
    for member in ga:
        x, y = member.prop(ri)
        xs.append(x)
        ys.append(y)
        
        z = function_to_maximize(x, y)
        
        print(x, y, z)
        zs.append(z)

        evl.append(z)
    
    ga.evolve(evl)
    plt.scatter(xs, ys, zs=zs, s=30, c=gc)



# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = function_to_maximize(X, Y)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_zlim(-1.01, 1)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
