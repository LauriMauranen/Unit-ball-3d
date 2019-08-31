# Unit-ball-3d

If for some peculiar reason you need stl-file for 3d-sphere of some shape you are in the right place! 

Function `ball` returns vertices and faces and if filename is set it saves stl-file to working directory. Parameters y_param and x_param move points to the sides or to the center. With value 1 points are concentrated in sides and with 2 very in the middle.

```python
from mpl_toolkits.mplot3d import Axes3D
from unitball3d import ball
import matplotlib.pyplot as plt

vertices, faces = ball(p=2, n=10, filename='ball.stl', y_param=1, x_param=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = vertices[:,0]
y = vertices[:,1]
z = vertices[:,2]

ax.plot_trisurf(x, y, triangles=faces, Z=z)

plt.show()
```
![alt text][pic]

[pic] = https://github.com/LauriMauranen/Unit-ball-3d/blob/master/ball.stl
