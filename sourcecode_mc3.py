import numpy as np
import time
import ipywidgets as wid
import traitlets
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import integrate
from ipywidgets import interact, fixed
from scipy.optimize import root
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def mc_example_3_1_func(x,y):
    return (x*x*y)

def mc3_example_2_1(option):
	nxy = 51
	x = np.linspace(0.,1.,nxy)
	y = np.linspace(3.,4.,nxy)
	X,Y = np.meshgrid(x, y)
	Z = mc_example_3_1_func(X, Y)
	
	verts1 = []
	verts2 = []
	verts3 = []
	if option == 'x':
		verts1.append([x[np.int(nxy/2)],y[0],0.])
		for i in range(nxy):
			verts1.append([x[np.int(nxy/2)],y[i],mc_example_3_1_func(x[np.int(nxy/2)], y[i])])
		verts1.append([x[np.int(nxy/2)],y[-1],0.])
		verts2.append([x[np.int(nxy/4)],y[0],0.])
		for i in range(nxy):
			verts2.append([x[np.int(nxy/4)],y[i],mc_example_3_1_func(x[np.int(nxy/4)], y[i])])
		verts2.append([x[np.int(nxy/4)],y[-1],0.])
		verts3.append([x[np.int(nxy/1.2)],y[0],0.])
		for i in range(nxy):
			verts3.append([x[np.int(nxy/1.2)],y[i],mc_example_3_1_func(x[np.int(nxy/1.2)], y[i])])
		verts3.append([x[np.int(nxy/1.2)],y[-1],0.])
	elif option == 'y':
		verts1 = []
		verts1.append([x[0],y[np.int(nxy/2)],0.])
		for i in range(nxy):
			verts1.append([x[i],y[np.int(nxy/2)],mc_example_3_1_func(x[i],y[np.int(nxy/2)])])
		verts1.append([x[-1],y[np.int(nxy/2)],0.])
		verts2.append([x[0],y[np.int(nxy/4)],0.])
		for i in range(nxy):
			verts2.append([x[i],y[np.int(nxy/4)],mc_example_3_1_func(x[i],y[np.int(nxy/4)])])
		verts2.append([x[-1],y[np.int(nxy/4)],0.])
		verts3.append([x[0],y[np.int(nxy/1.2)],0.])
		for i in range(nxy):
			verts3.append([x[i],y[np.int(nxy/1.2)],mc_example_3_1_func(x[i],y[np.int(nxy/1.2)])])
		verts3.append([x[-1],y[np.int(nxy/1.2)],0.])
	
	fig = plt.figure(figsize=(16, 8))
	if option == 'default':
		ax1 = fig.add_subplot(121, projection='3d')
		ax1.set_xlabel('x')
		ax1.set_ylabel('y')
		ax1.set_zlabel('f(x,y)')
		ax1.set_zlim([0,4])
		ax1.set_title('Surface')
		ax1.plot_surface(X, Y, 0*Z, rstride=1, cstride=1, color='0.75', linewidth=0, antialiased=True)
		ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.plasma, linewidth=0, antialiased=False)
		ax1.view_init(30, 315)
		ax2 = fig.add_subplot(122)
		ax2.set_xlabel('x')
		ax2.set_ylabel('y')
		ax2.set_title('Integration Region')
		ax2.plot([x[0],x[-1],x[-1],x[0],x[0]],[y[0],y[0],y[-1],y[-1],y[0]], 'k--o')
	else:
		ax1 = fig.add_subplot(111, projection='3d')
		ax1.set_xlabel('x')
		ax1.set_ylabel('y')
		ax1.set_zlabel('f(x,y)')
		ax1.set_zlim([0,4])
		ax1.plot_wireframe(X, Y, 0*Z, rstride=nxy, cstride=nxy, color='k', linewidth=1.0, antialiased=True)
		ax1.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='black', linewidth=0.4)
		face1 = Poly3DCollection([verts1], linewidth=1, alpha=0.5)
		face2 = Poly3DCollection([verts2], linewidth=1, alpha=0.5)
		face3 = Poly3DCollection([verts3], linewidth=1, alpha=0.5)
		face1.set_facecolor((0, 0, 1, 0.5))
		face2.set_facecolor((0, 0, 1, 0.5))
		face3.set_facecolor((0, 0, 1, 0.5))
		ax1.add_collection3d(face1)
		ax1.add_collection3d(face2)
		ax1.add_collection3d(face3)
		ax1.view_init(30, 120)
		
def mc3_example_3_1_func(x,y):
    return (y)

def mc3_example_3_1(option):
	nxy = 101
	x = np.linspace(0.,2.,nxy)
	y = 2.*x
	X,Y = np.meshgrid(x, y)
	Z = mc3_example_3_1_func(X, Y)

	verts = []
	verts.append([x[-1],y[0],0.])
	for i in range(nxy):
		verts.append([x[i],y[i],0.])
	verts.append([x[-1],y[0],0.])

	if option != 'default':
		verts1 = []
		verts2 = []
		verts3 = []
		
		if option == 'x':
			tmp = np.int(nxy/2)
			verts1.append([x[tmp],y[0],0.])
			for i in range(tmp):
				verts1.append([x[tmp],y[i],mc3_example_3_1_func(x[tmp],y[i])])
			verts1.append([x[tmp],y[tmp],0.])

			tmp = np.int(nxy/4)
			verts2.append([x[tmp],y[0],0.])
			for i in range(tmp):
				verts2.append([x[tmp],y[i],mc3_example_3_1_func(x[tmp],y[i])])
			verts2.append([x[tmp],y[tmp],0.])

			tmp = np.int(nxy/1.2)
			verts3.append([x[tmp],y[0],0.])
			for i in range(tmp):
				verts3.append([x[tmp],y[i],mc3_example_3_1_func(x[tmp],y[i])])
			verts3.append([x[tmp],y[tmp],0.])	

		if option == 'y':
			tmp = np.int(nxy/2)
			verts1.append([x[tmp],y[tmp],0.])
			for i in range(nxy-tmp):
				verts1.append([x[tmp+i],y[tmp],mc3_example_3_1_func(x[tmp+i],y[tmp])])
			verts1.append([x[-1],y[tmp],0.])

			tmp = np.int(nxy/4)
			verts2.append([x[tmp],y[tmp],0.])
			for i in range(nxy-tmp):
				verts2.append([x[tmp+i],y[tmp],mc3_example_3_1_func(x[tmp+i],y[tmp])])
			verts2.append([x[-1],y[tmp],0.])

			tmp = np.int(nxy/1.2)
			verts3.append([x[tmp],y[tmp],0.])
			for i in range(nxy-tmp):
				verts3.append([x[tmp+i],y[tmp],mc3_example_3_1_func(x[tmp+i],y[tmp])])
			verts3.append([x[-1],y[tmp],0.])
		
	fig = plt.figure(figsize=(16, 8))
	if option == 'default':
		ax1 = fig.add_subplot(121, projection='3d')
		ax1.set_xlabel('x')
		ax1.set_ylabel('y')
		ax1.set_zlabel('f(x,y)')
		ax1.set_zlim([0,4])
		ax1.set_title('Surface')
		ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.plasma, linewidth=0, antialiased=False)
		ax1.view_init(30, 325)
		ax2 = fig.add_subplot(122)
		ax2.set_xlabel('x')
		ax2.set_ylabel('y')
		ax2.set_title('Integration region')
		ax2.plot([x[0],x[-1]],[y[0],y[0]], 'k--o')
		ax2.plot([x[-1],x[-1]],[y[0],y[-1]], 'k--o')
		ax2.plot([x[-1],x[0]],[y[-1],y[0]], 'k--o')
		r = Poly3DCollection([verts], linewidth=1, alpha=0.5)
		r.set_facecolor((0.2, 0.2, 0.2, 0.5))
		ax1.add_collection3d(r)
	else:
		ax1 = fig.add_subplot(111, projection='3d')
		ax1.set_xlabel('x')
		ax1.set_ylabel('y')
		ax1.set_zlabel('f(x,y)')
		ax1.set_zlim([0,4])
		#ax1.plot_wireframe(X, Y, 0*Z, rstride=nxy, cstride=nxy, color='k', linewidth=1.0, antialiased=True)
		ax1.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='black', linewidth=0.4)
		face1 = Poly3DCollection([verts1], linewidth=1, alpha=0.5)
		face2 = Poly3DCollection([verts2], linewidth=1, alpha=0.5)
		face3 = Poly3DCollection([verts3], linewidth=1, alpha=0.5)
		face1.set_facecolor((0, 0, 1, 0.5))
		face2.set_facecolor((0, 0, 1, 0.5))
		face3.set_facecolor((0, 0, 1, 0.5))
		ax1.add_collection3d(face1)
		ax1.add_collection3d(face2)
		ax1.add_collection3d(face3)
		ax1.view_init(30, 315)
		r = Poly3DCollection([verts], linewidth=1, alpha=0.5)
		r.set_facecolor((0.2, 0.2, 0.2, 0.5))
		ax1.add_collection3d(r)