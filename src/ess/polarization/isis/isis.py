from typing import Mapping, NewType, TypeVar, Tuple

import numpy as np
import sciline as sl
import scipp as sc


from ..base import PolarizerSpin, AnalyzerSpin, ReducedDataByRunSectionAndWavelength, ReducedSampleDataBySpinChannel


Period = NewType('Period', int)

SpinToPeriod = NewType('SpinToPeriod', Mapping[Tuple[PolarizerSpin, AnalyzerSpin], Period])


def extract_sample_data(
    data: ReducedDataByRunSectionAndWavelength,
    polarizer_spin: PolarizerSpin,
    analyzer_spin: AnalyzerSpin,
    spin_to_period_mapping: SpinToPeriod,
) -> ReducedSampleDataBySpinChannel[PolarizerSpin, AnalyzerSpin]:

    period = spin_to_period_mapping[(polarizer_spin, analyzer_spin)]
    return ReducedSampleDataBySpinChannel[PolarizerSpin, AnalyzerSpin](
        data['period', period]
    )


providers = [
    extract_sample_data,
]
