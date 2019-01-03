from beyourself.core.util import *
from numpy.testing import assert_equal
from datetime import datetime, timedelta


def test_sync_relative_time():
    matching = {}
    matching['relative'] = "00:18:21.000"
    matching['absolute'] = "2017-08-07 20:14:21.377"

    dt = sync_relative_time(timedelta_from_str("00:02:49.484"), matching)

    assert (datetime_to_str(dt) == "2017-08-07 19:58:49.861")


def test_epoch_to_datetime():

    epoch = 1503572400000
    dt = epoch_to_datetime(epoch)
    assert epoch == datetime_to_epoch(dt)


def test_epoch_to_relative_str():
    dt = epoch_to_relative_str(10800000)
    assert (dt == "03:00:00.000")

    dt = epoch_to_relative_str(10830000)
    assert (dt == "03:00:30.000")
