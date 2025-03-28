import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
xs = np.linspace(0,100,101)
ys = np.linspace(0,100,101)
X,Y = np.meshgrid(xs,ys)
print(X.shape,Y.shape)
k = 1/100
angle = -25
kx = np.sin(angle/180*np.pi)*k
ky = np.cos(angle/180*np.pi)*k
Z = kx*X+ky*Y
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Z, cmap="jet",
                       linewidth=0, antialiased=False)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
#ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()