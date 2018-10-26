import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
import ipywidgets as wid
import traitlets
from ipywidgets import interact, fixed


def mc_example_2_1(ts,te,tstep):
	# set time and position vector
	t = np.arange(ts,te+tstep,tstep)
	rx = t
	ry = t

	# plot object motion as dots/line
	fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))
	ax1.set_xlabel('x(t)')
	ax1.set_ylabel('y(t)')
	ax1.set_xlim([-0.1,4.1])
	ax1.set_ylim([-0.1,4.1])
	ax1.plot(rx,ry,'k-o')
	
	
def mc_example_2_2(ts,te,tstep):
	# set time and position vector
	t = np.arange(ts,te+tstep,tstep)
	rx = -2+2*np.cos(t)
	ry = 2+2*np.sin(t)

	# plot object motion as dots/line
	fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))
	ax1.set_xlabel('x(t)')
	ax1.set_ylabel('y(t)')
	ax1.set_xlim([-4.1,0.1])
	ax1.set_ylim([-0.1,4.1])
	ax1.plot(rx,ry,'k-o')

def mc_example_2_3(ts,te,tstep):	
	# set time and position vector
	t = np.arange(ts,te+tstep,tstep)
	rx = np.cos(t)
	ry = np.sin(t)
	rz = t

	# plot object motion as dots/line
	fig = plt.figure(figsize=(14, 8))
	ax1 = fig.add_subplot(111, projection='3d')
	ax1.set_xlabel('x(t)')
	ax1.set_ylabel('y(t)')
	ax1.set_zlabel('z(t)')
	ax1.set_xlim([-1.1,1.1])
	ax1.set_ylim([-1.1,1.1])
	ax1.set_zlim([-0.1,2.*np.pi+0.1])
	ax1.plot(rx,ry,rz,'k-o')
	
	
def mc_example_2_4(ts,te,tstep):
	# set time and position vector
	t = np.arange(ts,te+tstep,tstep)
	rx = t*0-2
	ry = 5*np.cos(t)
	rz = np.sin(t)

	# plot object motion as dots/line
	fig = plt.figure(figsize=(14, 8))
	ax1 = fig.add_subplot(121, projection='3d')
	ax1.set_xlabel('x(t)')
	ax1.set_ylabel('y(t)')
	ax1.set_zlabel('z(t)')
	ax1.set_xlim([-2.1,-1.9])
	ax1.set_ylim([-5.1,5.1])
	ax1.set_zlim([-1.1,1.1])
	ax1.plot(rx,ry,rz,'k-o')

	ax2 = fig.add_subplot(122)
	ax2.set_xlabel('y(t)')
	ax2.set_ylabel('z(t)')
	ax2.set_xlim([-5.1,5.1])
	ax2.set_ylim([-1.1,1.1])
	ax2.plot(ry,rz,'k-o',label='x=-2')
	ax2.legend()
	
def mc_example_2_5(ts,te,tstep):
	# set time and position vector
	t = np.arange(ts,te+tstep,tstep)
	rx = t+np.sin(t)
	ry = 1+np.cos(t)

	# plot object motion as dots/line
	fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))
	ax1.set_xlabel('x(t)')
	ax1.set_ylabel('y(t)')
	ax1.set_xlim([-0.1,9.1])
	ax1.set_ylim([-0.1,2.1])
	ax1.plot(rx,ry,'k-o')

def mc_example_3_1(ts,te,tstep):
	# set time and position vector
	t = np.arange(ts,te+tstep,tstep)
	rx = np.cos(t)
	ry = np.sin(t)
	rz = t
	vx = -1.*np.sin(t)
	vy = np.cos(t)
	vz = 1.

	# plot object trajectory
	fig = plt.figure(figsize=(14, 8))
	ax1 = fig.add_subplot(121, projection='3d')
	ax1.set_xlabel('x(t)')
	ax1.set_ylabel('y(t)')
	ax1.set_zlabel('z(t)')
	ax1.set_xlim([-1.1,1.1])
	ax1.set_ylim([-1.1,1.1])
	ax1.set_zlim([-0.1,2*np.pi+0.1])
	ax1.plot(rx,ry,rz,'k-o',label='r(t)')
	ax1.legend()

	# plot object velocity
	ax2 = fig.add_subplot(122, projection='3d')
	ax2.set_xlabel('dx/dt')
	ax2.set_ylabel('dy/dt')
	ax2.set_zlabel('dz/dt')
	ax2.set_xlim([-1.1,1.1])
	ax2.set_ylim([-1.1,1.1])
	ax2.set_zlim([0.9,1.1])
	ax2.plot(vx,vy,vz,'b-o',label='v(t)')
	ax2.legend()

def mc_example_4_1(ts,te,tstep):
	# set time and position vector
	t = np.arange(ts,te+tstep,tstep)
	rx = 2.*np.cos(t)
	ry = 2.*np.sin(t)
	rz = 0*t
	dist = 2.*t

	# plot object trajectory
	fig = plt.figure(figsize=(14, 8))
	ax1 = fig.add_subplot(121, projection='3d')
	ax1.set_xlabel('x(t)')
	ax1.set_ylabel('y(t)')
	ax1.set_zlabel('z(t)')
	ax1.set_xlim([-2.1,2.1])
	ax1.set_ylim([-2.1,2.1])
	ax1.set_zlim([-0.1,0.1])
	ax1.plot(rx,ry,rz,'k-o',label='r(t)')
	ax1.legend()

	# plot object distance travelled
	ax2 = fig.add_subplot(122)
	ax2.set_xlabel('t')
	ax2.set_ylabel('s')
	ax2.set_xlim([-0.1,np.pi+0.1])
	ax2.set_ylim([-0.1,2.*np.pi+0.1])
	ax2.plot(t,dist,'k-o',label='s(t)')
	ax2.legend()

def mc_example_4_2(ts,te,tstep):
	# set time and position vector
	t = np.arange(ts,te+tstep,tstep)
	rx = 2.*t
	ry = 3.*np.sin(2.*t)
	rz = 3.*np.cos(2.*t)
	dist = 2.*np.sqrt(10.)*t

	# plot object trajectory
	fig = plt.figure(figsize=(14, 8))
	ax1 = fig.add_subplot(121, projection='3d')
	ax1.set_xlabel('x(t)')
	ax1.set_ylabel('y(t)')
	ax1.set_zlabel('z(t)')
	#ax1.set_xlim([-2.1,2.1])
	#ax1.set_ylim([-2.1,2.1])
	#ax1.set_zlim([-0.1,0.1])
	ax1.plot(rx,ry,rz,'k-o',label='r(t)')
	ax1.legend()

	# plot object distance travelled
	ax2 = fig.add_subplot(122)
	ax2.set_xlabel('t')
	ax2.set_ylabel('s')
	#ax2.set_xlim([-0.1,np.pi+0.1])
	#ax2.set_ylim([-0.1,2.*np.pi+0.1])
	ax2.plot(t,dist,'k-o',label='s(t)')
	ax2.legend()

def mc_example_arclength(ts,te,tstep):
	# set time and position vector
	t = np.arange(ts,te+tstep,tstep)
	rx = np.sin(t)
	ry = np.cos(t)
	dist = t

	# plot object trajectory
	fig = plt.figure(figsize=(14, 8))
	ax1 = fig.add_subplot(121)
	ax1.set_xlabel('x(t)')
	ax1.set_ylabel('y(t)')
	ax1.set_xlim([-0.1,1.1])
	ax1.set_ylim([-1.1,1.1])
	ax1.plot(rx,ry,'k-o',label='r(t)')
	ax1.legend()

	# plot object distance travelled
	ax2 = fig.add_subplot(122)
	ax2.set_xlabel('t')
	ax2.set_ylabel('s')
	ax2.set_xlim([-0.1,np.pi+0.1])
	ax2.set_ylim([-0.1,np.pi+0.1])
	ax2.plot(t,dist,'k-o',label='s(t)')
	ax2.legend()

def mc_example_5_1(ts,te,tstep):
	# set time and position vector
	t = np.arange(ts,te+tstep,tstep)
	rx = np.cos(t)*np.cos(t)
	ry = -2.*np.sin(2.*t)
	rz = t*t
	
	# tangent vector i.e. velocity
	vx = -2.*np.cos(t)*np.sin(t)
	vy = -4.*np.cos(2.*t)
	vz = 2.*t
	
	# unit tangent vector
	norm = 1./(vx*vx+vy*vy+vz*vz)
	utvx = norm*vx
	utvy = norm*vy
	utvz = norm*vz
	
	# plot object trajectory
	fig = plt.figure(figsize=(14, 8))
	ax1 = fig.add_subplot(111, projection='3d')
	ax1.set_xlabel('x(t)')
	ax1.set_ylabel('y(t)')
	ax1.set_zlabel('z(t)')
	ax1.set_xlim([-0.1,1.1])
	ax1.set_ylim([-2.1,0.1])
	ax1.set_zlim([-0.1,np.pi*np.pi+0.1])
	ax1.quiver(rx[-1],ry[-1],rz[-1],utvx[-1],utvy[-1],utvz[-1], pivot='middle',normalize=True, label='unit tangent vector')
	ax1.plot(rx,ry,rz,'ko',label='r(t)')
	ax1.legend()
	
def mc_example_7_1(ts,te,tstep):
	# set time and position vector
	t = np.arange(ts,te+tstep,tstep)
	rx = t
	ry = 2.*t

	# set force vector field
	X, Y = np.meshgrid(np.arange(0,1,0.1), np.arange(0,2,0.1))
	U = Y*Y
	V = -1*(X*X)

	
	# plot force field as arrows and object motion as blue dots/line
	fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
	ax1.quiver(X, Y, U, V, units='width')
	ax1.plot(rx,ry,'b-o')
	
def mc_example_7_3():
	# set force vector field
	X, Y = np.meshgrid(np.arange(-1,1,0.1), np.arange(-1,1,0.1))
	U = -1.*Y
	V = X*Y

	# set time and position vector
	t = np.linspace(0,np.pi/2.,11)
	rx = np.cos(t)
	ry = np.sin(t)

	# plot force field as arrows and object motion as blue dots/line
	fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
	ax1.quiver(X, Y, U, V, units='width')
	ax1.plot(rx,ry,'b-o')

def mc_example_7_4():
	# set force vector field
	X, Y, Z = np.meshgrid(np.arange(-0.2,1.2,0.2),
						  np.arange(-0.2,1.2,0.2),
						  np.arange(-0.2,1.2,0.2))
	U = X
	V = -1*Z
	W = 2*Y

	# set time and position vector for each segment
	t = np.linspace(0,1,11)
	rx1 = t
	ry1 = t
	rz1 = t*0
	rx2 = t*0+1
	ry2 = t*0+1
	rz2 = t
	rx3 = 1-t
	ry3 = 1-t
	rz3 = 1-t

	# plot object motion as dots/line
	fig = plt.figure(figsize=(14, 8))
	ax1 = fig.add_subplot(111, projection='3d')
	ax1.set_xlabel('x(t)')
	ax1.set_ylabel('y(t)')
	ax1.set_zlabel('z(t)')
	ax1.quiver(X, Y, Z, U, V, W, length=0.1, normalize=True)
	ax1.plot(rx1,ry1,rz1,'b-o')
	ax1.plot(rx2,ry2,rz2,'b-o')
	ax1.plot(rx3,ry3,rz3,'b-o')