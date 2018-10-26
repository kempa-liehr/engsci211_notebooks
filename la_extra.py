#import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from numpy import ma

#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.patches import Polygon
#from matplotlib.collections import PatchCollection
#from glob import glob
#from copy import copy
#from ipywidgets import interact, fixed


def example_1():
	n = 101
	x1 = np.linspace(0.,1.5,n)
	y1 = 2.*x1-1.
	x2 = np.linspace(0.,2.,n)
	y2 = (3.-x2)/2.

	fig = plt.figure(figsize=(16, 8))
	ax1 = fig.add_subplot(111)
	ax1.set_xlabel('x')
	ax1.set_ylabel('y')
	line1 = ax1.plot(x1,y1, 'r-', label='2x-y=1')
	line2 = ax1.plot(x2,y2, 'b-', label='x+2y=3')
	ax1.axhline(linewidth=2, color='k')
	ax1.axvline(linewidth=2, color='k')
	ax1.plot(1,1, 'ko')
	ax1.plot([1,1],[0,1], 'k--')
	ax1.plot([0,1],[1,1], 'k--')
	ax1.legend(fontsize=20)
    
    
def jacobi_1(x0, y0, display):
	x = [x0]
	y = [y0]
	tol = 0.00005
	print(tol)
	tol = 5.e-5
	print(tol)
    print("Iteration xold    yold    xnew    ynew")
	for i in range(100):
		x[i+1] = 0.5*y[i]+0.5
		y[i+1] = -0.5*x[i]+1.5
		print("%02d        %6.4f  %6.4f  %6.4f  %6.4f" % (i+1,xold,yold,xnew,ynew))
		if np.abs(x[i+1]-1.) <= tol and np.abs(y[i+1]-1.) <= tol:
			break        