from beyourself.data import get_necklace
from beyourself.core.util import assert_monotonic
from numpy.testing import assert_equal
import numpy as np


def test_dataapi():
	start = 1502153929861
	end = start + 10000
	df = get_necklace('P108', start, end)

	assert ((df.Time >= start).all() and (df.Time <= end).all())

	assert_monotonic(df.Time.as_matrix())


