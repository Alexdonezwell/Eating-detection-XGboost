from .context import mining
from numpy.testing import assert_almost_equal
from mining.periodic import *


def check_absolute_periodic(a, pmin, pmax):
	for i in range(len(a) - 1):
		diff = a[i + 1] - a[i]
		assert(diff >= pmin and diff <= pmax)
		

def test_absolute_error_periodic():
	a = [1,1.5,2,3,4,7,10,13,16,19]

	pmin = 0.89
	pmax = 1.11

	out = absolute_error_periodic_subsequence(a, pmin, pmax)

	for chunk in out:
		sequence = [a[t] for t in chunk]
		check_absolute_periodic(sequence, pmin, pmax)


def test_periodic_stat():
	a = [1,2,3,4]

	stat = get_periodic_stat(a)

	assert_almost_equal(stat['pmin'], 1)
	assert_almost_equal(stat['pmax'], 1)
	assert_almost_equal(stat['eps'], 0)