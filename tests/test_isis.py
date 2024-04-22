import pytest
from itertools import product

import scipp as sc
import numpy as np
import sciline as sl


from ess.polarization.isis.isis import extract_sample_data, Period, SpinToPeriod
from ess.polarization.base import spin_up, spin_down, ReducedSampleDataBySpinChannel, ReducedDataByRunSectionAndWavelength, Up, Down


@pytest.fixture
def seed():
    return np.random.seed(123)


@pytest.fixture
def reduced_isis_data(seed):
    return sc.DataArray(
        sc.ones(dims=['events'], shape=(10,)),
        coords=dict(
            period=sc.array(dims=['events'], values=np.random.randint(0, 4, 10))
        ),
    ).group('period')


def test_extract_sample_data(reduced_isis_data):
    sc.testing.assert_identical(
        extract_sample_data(
            reduced_isis_data,
            1,
            1,
            {(1, 1): 1}
        ),
        reduced_isis_data['period', 1],
    )


def test_extract_sample_data_in_pipeline(reduced_isis_data):
    pl = sl.Pipeline([extract_sample_data])
    pl[Up] = 1
    pl[Down] = -1
    pl[SpinToPeriod] = {(a, b): i for i, (a, b) in enumerate(product((1, -1), (1, -1)))}
    pl[ReducedDataByRunSectionAndWavelength] = reduced_isis_data

    sc.testing.assert_identical(
        pl.compute(ReducedSampleDataBySpinChannel[Up, Up]), reduced_isis_data['period', 0]
    )
    sc.testing.assert_identical(
        pl.compute(ReducedSampleDataBySpinChannel[Up, Down]), reduced_isis_data['period', 1]
    )
    sc.testing.assert_identical(
        pl.compute(ReducedSampleDataBySpinChannel[Down, Up]), reduced_isis_data['period', 2]
    )
    sc.testing.assert_identical(
        pl.compute(ReducedSampleDataBySpinChannel[Down, Down]), reduced_isis_data['period', 3]
    )
