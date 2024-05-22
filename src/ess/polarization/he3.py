# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from typing import Generic, Literal, NewType, TypeVar

import sciline as sl
import scipp as sc

from .types import (
    Down,
    PolarizationCorrectedSampleData,
    ReducedSampleDataBySpinChannel,
    Up,
)
from .uncertainty import broadcast_with_upper_bound_variances

Depolarized = NewType('Depolarized', int)
Polarized = NewType('Polarized', int)
"""Polarized either up or down, don't care."""
PolarizationState = TypeVar('PolarizationState', Polarized, Depolarized)


Analyzer = NewType('Analyzer', str)
Polarizer = NewType('Polarizer', str)
Cell = TypeVar('Cell', Analyzer, Polarizer)

DirectBeamNoCell = NewType('DirectBeamNoCell', sc.DataArray)
"""Direct beam without cells and sample as a function of wavelength."""


class He3CellPressure(sl.Scope[Cell, sc.Variable], sc.Variable):
    """Pressure for a given cell."""


class He3CellLength(sl.Scope[Cell, sc.Variable], sc.Variable):
    """Length for a given cell."""


class He3CellTemperature(sl.Scope[Cell, sc.Variable], sc.Variable):
    """Temperature for a given cell."""


class He3FillingTime(sl.Scope[Cell, sc.Variable], sc.Variable):
    """Filling wall-clock time for a given cell."""


class He3CellTransmissionFraction(
    sl.ScopeTwoParams[Cell, PolarizationState, sc.DataArray], sc.DataArray
):
    """Transmission fraction for a given cell"""


class He3TransmissionEmptyGlass(sl.Scope[Cell, sc.Variable], sc.Variable):
    """Transmission of the empty glass for a given cell."""


class He3DirectBeam(
    sl.ScopeTwoParams[Cell, PolarizationState, sc.DataArray], sc.DataArray
):
    """
    Direct beam data for a given cell and spin state as a function of wavelength.
    """


class He3Opacity0(sl.Scope[Cell, sc.Variable], sc.Variable):
    """Opacity at 1 Angstrom for a given cell."""


class He3OpacityFunction(Generic[Cell]):
    """Wavelength-dependent opacity function for a given cell."""

    def __init__(self, opacity0: sc.Variable):
        self._opacity0 = opacity0.to(unit='1/Angstrom')

    @property
    def opacity0(self) -> sc.Variable:
        return self._opacity0

    def __call__(self, wavelength: sc.Variable) -> sc.Variable:
        scale = broadcast_with_upper_bound_variances(
            self.opacity0, sizes=wavelength.sizes
        )
        return sc.DataArray(
            (scale * wavelength).to(unit='', copy=False),
            coords={'wavelength': wavelength},
        )


def he3_opacity_from_cell_params(
    pressure: He3CellPressure[Cell],
    length: He3CellLength[Cell],
    temperature: He3CellTemperature[Cell],
) -> He3Opacity0[Cell]:
    """Opacity 0 for a given cell, estimated from pressure and cell length."""
    from scipp.constants import Boltzmann as k_B

    he3_neutron_absorption_cross_section_at_1_angstrom = 2966.0e-24 * sc.Unit(
        'cm^2/Angstrom'
    )
    # Try to convert to Kelvin, since we get a more cryptic error message later if we
    # do not, in case the user passes in a temperature in degree Celsius.
    temperature = temperature.to(unit='K')

    opacity0 = (
        he3_neutron_absorption_cross_section_at_1_angstrom
        / (k_B * temperature)
        * pressure
        * length
    )
    return He3Opacity0[Cell](opacity0)


def he3_opacity_function_from_cell_opacity(
    opacity0: He3Opacity0[Cell],
) -> He3OpacityFunction[Cell]:
    """
    Opacity function for a given cell, based on pressure and cell length.

    Note that this can alternatively be defined via neutron beam data, see
    :py:func:`he3_opacity_function_from_beam_data`.
    """
    return He3OpacityFunction[Cell](opacity0)


def he3_opacity_function_from_beam_data(
    transmission_empty_glass: He3TransmissionEmptyGlass[Cell],
    transmission_fraction: He3CellTransmissionFraction[Cell, Depolarized],
    opacity0_initial_guess: He3Opacity0[Cell],
) -> He3OpacityFunction[Cell]:
    """
    Opacity function for a given cell, based on direct beam data.

    Note that this can alternatively be defined via cell parameters, see
    :py:func:`he3_opacity_function_from_cell_opacity`. The cell opacity is used as an
    initial guess for the fit.
    """

    def intensity(wavelength: sc.Variable, opacity0: sc.Variable) -> sc.Variable:
        opacity = He3OpacityFunction[Cell](opacity0)
        return transmission_empty_glass * sc.exp(-opacity(wavelength))

    transmission_fraction = transmission_fraction.copy(deep=False)
    transmission_fraction.coords['wavelength'] = sc.midpoints(
        transmission_fraction.coords['wavelength']
    )

    popt, _ = sc.curve_fit(
        ['wavelength'],
        intensity,
        transmission_fraction,
        p0={'opacity0': opacity0_initial_guess},
    )
    return He3OpacityFunction[Cell](sc.values(popt['opacity0']).data)


class He3PolarizationFunction(Generic[Cell]):
    """Time-dependent polarization function for a given cell."""

    def __init__(self, C: sc.Variable, T1: sc.Variable):
        self._C = C
        self._T1 = T1

    @property
    def C(self) -> sc.Variable:
        return self._C

    @property
    def T1(self) -> sc.Variable:
        return self._T1

    def __call__(self, time: sc.Variable) -> sc.Variable:
        return sc.DataArray(self.C * sc.exp(-time / self.T1), coords={'time': time})


@dataclass
class He3TransmissionFunction(Generic[Cell]):
    """Wavelength- and time-dependent transmission for a given cell."""

    opacity_function: He3OpacityFunction[Cell]
    polarization_function: He3PolarizationFunction[Cell]
    transmission_empty_glass: He3TransmissionEmptyGlass[Cell]

    def __call__(
        self,
        *,
        time: sc.Variable,
        wavelength: sc.Variable,
        plus_minus: Literal['plus', 'minus'],
    ) -> sc.Variable:
        opacity = self.opacity_function(wavelength)
        polarization = self.polarization_function(time)
        if plus_minus == 'minus':
            polarization *= -1.0
        return self.transmission_empty_glass * sc.exp(-opacity * (1.0 + polarization))


def transmission_incoming_unpolarized(
    *,
    transmission_empty_glass: sc.Variable,
    opacity: sc.Variable,
    polarization: sc.Variable,
) -> sc.Variable:
    return transmission_empty_glass * sc.exp(-opacity) * sc.cosh(opacity * polarization)


def compute_transmission_fraction_from_direct_beam(
    direct_beam_no_cell: DirectBeamNoCell,
    direct_beam_polarized: He3DirectBeam[Cell, PolarizationState],
) -> He3CellTransmissionFraction[Cell, PolarizationState]:
    """
    Compute the transmission fraction for a given cell and polarization state.

    This is defined as the ratio of the direct beam with the cell to the direct beam
    without the cell. The result is a function of wavelength and time.

    Note that this is possible only if the main detector is used to measure direct
    beam data. If direct beam is computed from monitors then, e.g., the SANS
    transmission fraction (as computed by a regular SANS workflow) should be used
    directly. Note that the regular SANS workflow also normalized to an empty beam
    run, make sure to not perform the division twice.
    """
    return He3CellTransmissionFraction[Cell, PolarizationState](
        direct_beam_polarized / direct_beam_no_cell
    )


def get_he3_transmission_from_fit_to_direct_beam(
    transmission_fraction: He3CellTransmissionFraction[Cell, Polarized],
    opacity_function: He3OpacityFunction[Cell],
    transmission_empty_glass: He3TransmissionEmptyGlass[Cell],
) -> He3TransmissionFunction[Cell]:
    """
    Return the transmission function for a given cell.

    This is composed from the opacity-function and the polarization-function.
    The implementation fits a time- and wavelength-dependent equation and returns
    the fitted T(t, lambda).

    DB_pol/DB = T_E * cosh(O(lambda)*P(t))*exp(-O(lambda))
    """

    def expected_transmission(
        wavelength: sc.Variable, time: sc.Variable, C: sc.Variable, T1: sc.Variable
    ) -> sc.Variable:
        opacity = opacity_function(wavelength)
        polarization_function = He3PolarizationFunction[Cell](C=C, T1=T1)
        polarization = polarization_function(time)
        return transmission_incoming_unpolarized(
            transmission_empty_glass=transmission_empty_glass,
            opacity=opacity,
            polarization=polarization,
        )

    transmission_fraction = transmission_fraction.copy(deep=False)
    transmission_fraction.coords['wavelength'] = sc.midpoints(
        transmission_fraction.coords['wavelength']
    )

    popt, _ = sc.curve_fit(
        ['wavelength', 'time'],
        expected_transmission,
        transmission_fraction,
        p0={'C': sc.scalar(0.8, unit=''), 'T1': sc.scalar(400000.0, unit='s')},
    )
    # TODO Consider including variances from fit
    polarization_function = He3PolarizationFunction[Cell](
        C=sc.values(popt['C']).data, T1=sc.values(popt['T1']).data
    )
    return He3TransmissionFunction[Cell](
        opacity_function=opacity_function,
        polarization_function=polarization_function,
        transmission_empty_glass=transmission_empty_glass,
    )


def get_he3_transmission_from_fit_to_direct_beam_from_polarized_incoming_beam(
    transmission_fraction: He3CellTransmissionFraction[Cell, Polarized],
    opacity_function: He3OpacityFunction[Cell],
    transmission_empty_glass: He3TransmissionEmptyGlass[Cell],
) -> He3TransmissionFunction[Cell]:
    """
    Return the transmission function for a given cell.

    This is composed from the opacity-function and the polarization-function.
    The implementation fits a time- and wavelength-dependent equation and returns
    the fitted T(t, lambda).

    DB_pol/DB = T_E * cosh(O(lambda)*P(t))*exp(-O(lambda))
    """

    def expected_transmission(
        wavelength: sc.Variable, time: sc.Variable, C: sc.Variable, T1: sc.Variable
    ) -> sc.Variable:
        polarization_function = He3PolarizationFunction[Cell](C=C, T1=T1)
        return He3TransmissionFunction(
            opacity_function=opacity_function,
            polarization_function=polarization_function,
            transmission_empty_glass=transmission_empty_glass,
        )(time=time, wavelength=wavelength, plus_minus='plus')

    transmission_fraction = transmission_fraction.copy(deep=False)
    transmission_fraction.coords['wavelength'] = sc.midpoints(
        transmission_fraction.coords['wavelength']
    )

    popt, _ = sc.curve_fit(
        ['wavelength', 'time'],
        expected_transmission,
        transmission_fraction,
        p0={'C': sc.scalar(0.8, unit=''), 'T1': sc.scalar(400000.0, unit='s')},
    )
    # TODO Consider including variances from fit
    polarization_function = He3PolarizationFunction[Cell](
        C=sc.values(popt['C']).data, T1=sc.values(popt['T1']).data
    )
    return He3TransmissionFunction[Cell](
        opacity_function=opacity_function,
        polarization_function=polarization_function,
        transmission_empty_glass=transmission_empty_glass,
    )


def compute_direct_beam(
    data: sc.DataArray,
    q_range: sc.Variable,
    background_q_range: sc.Variable,
) -> sc.DataArray:
    """
    Compute background-subtracted direct beam function.

    The input must be normalized data, not counts.
    """
    if data.bins.unit != '':
        raise ValueError(f'Input data must be normalized, got unit {data.unit}.')
    if q_range.max() > background_q_range.min():
        raise ValueError('Background range must be after direct beam range.')
    if q_range.min() < sc.scalar(0.0, unit='1/angstrom'):
        raise ValueError('Q-range must be positive.')
    q_range = q_range**2
    background_q_range = background_q_range**2
    start_db = q_range[0]
    stop_db = q_range[-1]
    start_bg = background_q_range[0]
    stop_bg = background_q_range[-1]
    # Simple approach for now: Assume we can treat this as rotation invariant
    qx = data.bins.coords['Qx']
    qy = data.bins.coords['Qy']
    data.bins.coords['Q_squared'] = qx**2 + qy**2
    # The input is binned in time and wavelength, we simply take the per-bin mean
    # without changes.
    beam_region = data.bins['Q_squared', start_db:stop_db].bins.mean()
    background = data.bins['Q_squared', start_bg:stop_bg].bins.mean()
    return beam_region - background


DirectBeamQRange = NewType('DirectBeamQRange', sc.Variable)
"""Q-range defining the direct beam region in a direct beam measurement."""

DirectBeamBackgroundQRange = NewType('DirectBeamBackgroundQRange', sc.Variable)
"""Q-range defining the direct beam background region in a direct beam measurement."""

ReducedDirectBeamDataNoCell = NewType('ReducedDirectBeamDataNoCell', sc.DataArray)


class ReducedDirectBeamData(
    sl.ScopeTwoParams[Cell, PolarizationState, sc.DataArray], sc.DataArray
):
    """Direct beam data for a given cell, as a function of wavelength and time."""


def direct_beam(
    data: ReducedDirectBeamDataNoCell,
    q_range: DirectBeamQRange,
    background_q_range: DirectBeamBackgroundQRange,
) -> DirectBeamNoCell:
    """
    Returns the direct beam function without any cells.

    The result is background-subtracted and returned as function of wavelength.
    Other dimensions of the input are preserved. In particular, the time dimension,
    corresponding to different direct beam measurements, is preserved.
    """
    return DirectBeamNoCell(
        compute_direct_beam(
            data=data,
            q_range=q_range,
            background_q_range=background_q_range,
        )
    )


def direct_beam_with_cell(
    data: ReducedDirectBeamData[Cell, PolarizationState],
    q_range: DirectBeamQRange,
    background_q_range: DirectBeamBackgroundQRange,
) -> He3DirectBeam[Cell, PolarizationState]:
    """
    Returns the direct beam function for a given cell.

    The result is background-subtracted and returned as function of wavelength and
    wall-clock time. The time dependence is coarse, i.e., due to different time
    intervals at which the direct beam is measured.
    """
    return He3DirectBeam[Cell, PolarizationState](
        compute_direct_beam(
            data=data,
            q_range=q_range,
            background_q_range=background_q_range,
        )
    )


def correct_sample_data_for_polarization(
    upup: ReducedSampleDataBySpinChannel[Up, Up],
    updown: ReducedSampleDataBySpinChannel[Up, Down],
    downup: ReducedSampleDataBySpinChannel[Down, Up],
    downdown: ReducedSampleDataBySpinChannel[Down, Down],
    transmission_polarizer: He3TransmissionFunction[Polarizer],
    transmission_analyzer: He3TransmissionFunction[Analyzer],
) -> PolarizationCorrectedSampleData:
    """
    Apply polarization correction for the case of He3 polarizers and analyzers.

    There will be a different version of this function for handling the supermirror
    case, since transmission is not time-dependent but spin-flippers need to be
    accounted for.
    """
    # 1. Apply polarization correction (matrix inverse)
    # 2. Compute weighted mean over time and wavelength, bin into Q-bins

    # Pseudo code:
    # result = [0,0,0,0]
    # for j in range(4):
    #     for i, channel in enumerate((upup, updown, downup, downdown)):
    #         da = PA_inv[j, i] * channel
    #         da *= weights  # weights from error bars or event counts?
    #         result[j] += da.bins.concat('time', 'wavelength').hist(Qx=100, Qy=100)
    raise NotImplementedError()


providers = (
    he3_opacity_from_cell_params,
    he3_opacity_function_from_beam_data,
    get_he3_transmission_from_fit_to_direct_beam,
    direct_beam,
    direct_beam_with_cell,
    correct_sample_data_for_polarization,
)


def He3CellWorkflow(in_situ: bool = True) -> sl.Pipeline:
    """
    Workflow performing polarization correction for beam lines with two He3 cells.

    Parameters
    ----------
    in_situ : bool
        Whether to use an in-situ definition of the cell opacity based on cell
        parameters, or an ex-situ definition based on direct beam data. The latter
        requires a direct-beam measurement with depolarized cells.
    """
    steps = (
        he3_opacity_from_cell_params,
        get_he3_transmission_from_fit_to_direct_beam,
        direct_beam,
        direct_beam_with_cell,
        correct_sample_data_for_polarization,
    )
    workflow = sl.Pipeline(providers=steps)
    if in_situ:
        workflow.insert(he3_opacity_function_from_cell_opacity)
    else:
        workflow.insert(he3_opacity_function_from_beam_data)
    return workflow
