import pytest
import numpy as np
from hiro_models.one_qubit_model import *
from qutip import *
from hiro_models.utility import assert_serializable
import scipy.integrate
import random


def test_sd_bcf_norm():
    random.seed(0)

    for _ in range(4):
        model = QubitModel(
            ω_c=random.uniform(0.5, 4),
            s=random.choice([1.0, 0.1, 0.5, 0.3]),
            T=random.uniform(0.1, 4),
            δ=1,
        )
        assert_serializable(model)

        bcf = model.bcf
        assert np.abs(
            scipy.integrate.quad(lambda t: bcf(t).imag, 0, np.inf)[0] * model.bcf_norm
        ) == pytest.approx(1)
        assert np.sum(model.bcf_coefficients()[0]) == pytest.approx(bcf(0), 1e-2)

        tsd = model.thermal_spectral_density
        assert tsd is not None
        assert pytest.approx(tsd(1)) == model.spectral_density(1) * 1 / np.expm1(
            1 / float(model.T)
        )

        tcorr = model.thermal_correlations
        assert tcorr is not None

        # tests if the normalization of tcorr is correct by
        # calculating the fourier transform
        model.thermal_process


def test_sd_bcf_norm_two_bath():
    np.random.seed(0)

    for _ in range(4):
        model = QubitModelMutliBath(
            ω_c=list(np.random.uniform(0.5, 4, 2)),
            s=list(np.random.choice([1.0, 0.1, 0.5, 0.3], 2)),
            T=list(np.random.uniform(0.1, 4, 2)),
            δ=[1, 1],
        )
        assert_serializable(model)

        for i in range(2):
            bcf = model.bcf(i)
            assert np.sum(model.bcf_coefficients()[0][i]) == pytest.approx(bcf(0), 1e-2)

            tsd = model.thermal_spectral_density(i)
            assert tsd is not None
            assert pytest.approx(tsd(1)) == model.spectral_density(i)(1) * 1 / np.expm1(
                1 / float(model.T[i])
            )

            tcorr = model.thermal_correlations(i)
            assert tcorr is not None

            # tests if the normalization of tcorr is correct by
            # calculating the fourier transform
            model.thermal_process(i)
