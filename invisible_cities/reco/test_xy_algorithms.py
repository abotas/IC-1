import numpy as np
from pytest import fixture

from invisible_cities.reco.params import Cluster
from invisible_cities.core.system_of_units_c import units
from invisible_cities.reco.xy_algorithms import corona, barycenter



@fixture
def toy_sipm_signal():
    xs = np.array([65, -65]) * units.mm
    ys = np.array([65, -65]) * units.mm
    qs = np.ones(2) * 5   * units.pes
    return xs, ys, qs

def test_barycenter():
    l   = 1000 # EVEN
    xst = np.arange(l)
    yst = np.arange(l)
    qst = np.concatenate([np.ones(l//2), np.zeros(l//2)])
    cluster = barycenter(xst, yst, qst)[0]
    assert cluster.X == (l / 2.0 - 1) / 2.0
    assert cluster.Y == (l / 2.0 - 1) / 2.0
    assert cluster.Q ==  l / 2.0

def test_barycenter2(toy_sipm_signal):
    xs, ys, qs = toy_sipm_signal
    cluster = barycenter(xs, ys, qs)[0]
    assert qs[0] == qs[1]
    assert cluster.X == xs.mean()
    assert cluster.Y == ys.mean()
    assert cluster.Q == qs.sum()

def test_corona_multiple_clusters(toy_sipm_signal):
    xs, ys, qs = toy_sipm_signal
    clusters = corona(xs, ys, qs, msipm=1, rmax=15*units.mm, T=4.9*units.pes)
    assert len(clusters) == 2
    for i in range(len(xs)):
        assert clusters[i].X == xs[i]
        assert clusters[i].Y == ys[i]
        assert clusters[i].Q == qs[i]

def test_corona_msipm(toy_sipm_signal):
    xs, ys, qs = toy_sipm_signal
    assert len(corona(xs, ys, qs, msipm=2)) == 0

def test_corona_threshold_for_local_max(toy_sipm_signal):
    xs, ys, qs = toy_sipm_signal
    assert len(corona(xs, ys, qs,
                      msipm =  1,
                      T     =  5.1*units.pes,
                      rmax  = 15  *units.mm )) == 0

def test_corona_rmax(toy_sipm_signal):
    xs, ys, qs = toy_sipm_signal
    assert len(corona(xs, ys, qs,
                      msipm =  1,
                      T     =   4.9*units.pes,
                      rmax  =  15  *units.mm)) == 2

    assert len(corona(xs, ys, qs,
                      msipm = 1,
                      T     = 4.9    *units.pes,
                      rmax  = 1000000*units.m)) == 1

def test_corona_barycenter_are_same_with_one_cluster(toy_sipm_signal):
    xs, ys, qs = toy_sipm_signal
    c_clusters = corona(xs,ys,qs, rmax=10*units.m, msipm=1, T=4.9*units.pes)
    b_clusters = barycenter(xs,ys,qs)
    c_cluster = c_clusters[0]
    b_cluster = b_clusters[0]

    assert len(c_cluster)  == len(b_cluster)
    assert c_cluster.X     == b_cluster.X
    assert c_cluster.Y     == b_cluster.Y
    assert c_cluster.Q     == b_cluster.Q
    assert c_cluster.Nsipm == b_cluster.Nsipm
    assert c_cluster.Xrms  == b_cluster.Xrms
    assert c_cluster.Yrms  == b_cluster.Yrms
