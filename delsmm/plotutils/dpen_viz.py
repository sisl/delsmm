#
# dpen_viz.py
#

import matplotlib.pyplot as plt
import numpy as np
from delsmm.plotutils.colorline import colorline
from scipy.interpolate import interp1d

def dpen_endeffector(q, l1, l2):
	"""
	Compute end-effector position for double pendulum.
	Args:
		q (torch.tensor): (B,T,2) joint angles
		l1 (float): arm 1 length
		l2 (float): arm 2 length
	Returns:
		x (torch.tensor): (B,T) x positions of end effector
		y (torch.tensor): (B,T) y positions of end effector
	"""

	q1 = q[...,0]
	q2 = q[...,1]

	x = q1.sin() * l1 + q2.sin() * l2
	y = -q1.cos() * l1 - q2.cos() * l2
	
	return x, y

def dpen_viz(ax, q, l1, l2):
	"""
	Viz a double pendulum trajectory.
	Args:
		ax (plt.Axis): axis to plot on
		q (torch.tensor): (1,T,2) joint angles
		l1 (float): arm 1 length
		l2 (float): arm 2 length
	Returns:
		None
	"""

	x, y = dpen_endeffector(q, l1, l2)
	x = x[0].detach().numpy()
	y = y[0].detach().numpy()
	T = len(x)
	t = np.arange(T)
	tsp = np.linspace(0., t.max(), 10*T)

	xsp = interp1d(t, x, kind='cubic')(tsp)
	ysp = interp1d(t, y, kind='cubic')(tsp)

	colorline(xsp, ysp, ax=ax, linewidth=1, cmap=plt.get_cmap('Reds'))

	qT = q[0,-1].detach().numpy()
	elbow_x = l1 * np.sin(qT[0])
	elbow_y = -l1 * np.cos(qT[0])
	eef_x = elbow_x + l2 * np.sin(qT[1])
	eef_y = elbow_y - l2 * np.cos(qT[1])

	points_x = np.array([0., elbow_x, eef_x])
	points_y = np.array([0., elbow_y, eef_y])

	plt.plot(points_x, points_y, color='k', marker="o", linewidth=3)

	return

