import pytest
import random
from two_qubit_model import *
from qutip import *
from utility import assert_serializable


class TestBasicConfigs:
    def test_xx_zz(self):
        model = TwoQubitModel(ω_2=2, γ=0.4, δ=[0.1, 0.2], s_vec=[[0, 1], [0, 1]])
        assert_serializable(model)

        assert model.local_system(0) == 1 / 2 * (tensor(sigmaz(), identity(2)))
        assert model.local_system(1) == (tensor(identity(2), sigmaz()))

        for j in [1, 1000]:
            model.j[0, 0] = j
            assert model.system == 1 / 2 * (
                tensor(sigmaz(), identity(2)) + tensor(identity(2), sigmaz()) * 2
            ) + 0.4 / 2 * tensor(sigmax(), sigmax())

        for s in [1, 10]:
            model.s_vec[0][1] = s
            model.s_vec[1][1] = s

            assert model.bath_coupling(0) == tensor(sigmaz(), identity(2))
            assert model.bath_coupling(1) == tensor(identity(2), sigmaz())

        assert model.bcf_scale(0) == 0.1 ** 2
        assert model.bcf_scale(1) == 0.2 ** 2

    def test_xy_zz(self):
        model = TwoQubitModel(ω_2=0.11, γ=10, δ=[23, 123], s_vec=[[0, 1], [0, 1]])
        model.j[0, 0] = 0
        model.j[0, 1] = 123

        assert_serializable(model)

        for j in [1, 1000]:
            model.j[0, 1] = j
            assert model.system == 1 / 2 * (
                tensor(sigmaz(), identity(2)) + tensor(identity(2), sigmaz()) * 0.11
            ) + 10 / 2 * tensor(sigmax(), sigmay())

        for s in [1, 10e4]:
            model.s_vec[0][1] = s
            model.s_vec[1][1] = s

            assert model.bath_coupling(0) == tensor(sigmaz(), identity(2))
            assert model.bath_coupling(1) == tensor(identity(2), sigmaz())

    def test_yx_zx(self):
        model = TwoQubitModel(ω_2=0.11, γ=10, δ=[23, 123], s_vec=[[0, 1], [1, 0]])
        model.j[0, 0] = 0
        model.j[1, 0] = 123

        assert_serializable(model)

        for j in [1, 1000]:
            model.j[1, 0] = j
            assert model.system == 1 / 2 * (
                tensor(sigmaz(), identity(2)) + tensor(identity(2), sigmaz()) * 0.11
            ) + 10 / 2 * tensor(sigmay(), sigmax())

        for s in [1, 10e4]:
            model.s_vec[0][1] = s
            model.s_vec[1][0] = s

            assert model.bath_coupling(0) == tensor(sigmaz(), identity(2))
            assert model.bath_coupling(1) == tensor(identity(2), sigmax())

    def test_yx_xz(self):
        model = TwoQubitModel(ω_2=0.11, γ=10, δ=[23, 123], s_vec=[[1, 0], [0, 1]])
        model.j[0, 0] = 0
        model.j[1, 0] = 123

        assert_serializable(model)

        for j in [1, 1000]:
            model.j[1, 0] = j
            assert model.system == 1 / 2 * (
                tensor(sigmaz(), identity(2)) + tensor(identity(2), sigmaz()) * 0.11
            ) + 10 / 2 * tensor(sigmay(), sigmax())

        for s in [1, 10e4]:
            model.s_vec[0][0] = s
            model.s_vec[1][1] = s

            assert model.bath_coupling(0) == tensor(sigmax(), identity(2))
            assert model.bath_coupling(1) == tensor(identity(2), sigmaz())

    def test_xz_xz(self):
        model = TwoQubitModel(ω_2=0.11, γ=10, δ=[23, 123], s_vec=[[1, 0], [0, 1]])
        model.j[0, 0] = 0

        assert_serializable(model)

        for j in [1, 1e5]:
            model.j[0, 2] = j
            assert model.system == 1 / 2 * (
                tensor(sigmaz(), identity(2)) + tensor(identity(2), sigmaz()) * 0.11
            ) + 10 / 2 * tensor(sigmax(), sigmaz())

        for s in [1, 10e4]:
            model.s_vec[0][0] = s
            model.s_vec[1][1] = s

            assert model.bath_coupling(0) == tensor(sigmax(), identity(2))
            assert model.bath_coupling(1) == tensor(identity(2), sigmaz())


def test_sd_bcf_norm():
    random.seed(0)
    for _ in range(4):
        model = TwoQubitModel(
            ω_c=[random.uniform(0.5, 4), random.uniform(0.5, 4)],
            s=random.choices([1.0, 0.1, 0.5, 0.3], k=2),
            T=[random.uniform(0.1, 4), random.uniform(0.1, 4)],
        )
        assert_serializable(model)

        for i in range(2):
            assert model.spectral_density(i).integral() == pytest.approx(np.pi)
            assert model.bcf(0)(0) == 1
            assert np.sum(model.bcf_coefficients()[0][i]) == pytest.approx(1, rel=1e-2)

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
