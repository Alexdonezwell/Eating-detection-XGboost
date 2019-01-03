import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from beyourself.cleanup.necklace import _parse_binary, leanForward, duplicate_second_to_millisecond
from beyourself.cleanup.quaternion import _axisangle_to_q, _qv_mult


def test_proximity_ambient():
	test = '1499778491,  2000283dcccccd3e4ccccd3e99999a3ecccccd3f0000003f19999a3f333333 0'
	results = _parse_binary(test, True)

	proximity = results['proximity']
	ambient = results['ambient']

	aX = results['aX']
	aY = results['aY']
	aZ = results['aZ']

	qW = results['qW']
	qX = results['qX']
	qY = results['qY']
	qZ = results['qZ']

	lf = results['leanForward']

	assert(proximity == 32)
	assert(ambient == 40)

	assert_almost_equal(aX, 0.1)
	assert_almost_equal(aY, 0.2)
	assert_almost_equal(aZ, 0.3)

	assert_almost_equal(qW, 0.4)
	assert_almost_equal(qX, 0.5)
	assert_almost_equal(qY, 0.6)
	assert_almost_equal(qZ, 0.7)


def test_quaternion():
	x_axis_unit = (1, 0, 0)
	y_axis_unit = (0, 1, 0)
	z_axis_unit = (0, 0, 1)

	q = _axisangle_to_q(y_axis_unit, np.pi / 3 * 2)
	v = _qv_mult(q, z_axis_unit)

	assert_almost_equal(leanForward((0.707, 0, 0.707,0)), 90)
	assert_almost_equal(leanForward((1,0,0,0)), 0)
	assert_almost_equal(leanForward((np.cos(30/180*np.pi),0,np.sin(30/180*np.pi),0)), 60)


def test_reliability():

	a = np.array([1000,1000,1000,2000,2000,2000,2000,4000,4000])
	newa = duplicate_second_to_millisecond(a)

	assert_array_almost_equal(newa, np.array([1000,1333,1666,2000,2250,2500,2750,4000,4500]))