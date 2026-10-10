import numpy as np

from wd2 import geometry as G
from wd2.frame import radar_range_estimates
from wd2.track import IoUTracker, kalman_cv_smooth


def test_fit_affine_recovers_line_with_outliers():
    x = np.linspace(1, 50, 100)
    y = 1.3 * x + 2.0
    y[::10] += 40  # 10% outliers
    s, t = G.fit_affine(x, y)
    assert abs(s - 1.3) < 1e-6 and abs(t - 2.0) < 1e-5


def test_robust_median_drops_background():
    z = np.r_[np.full(20, 20.0), np.full(5, 80.0)]
    assert G.robust_median(z) == 20.0


def test_ego_motion_detects_moving_platform():
    az = np.linspace(-0.6, 0.6, 400)
    radar = {"x": 30 * np.cos(az), "y": 30 * np.sin(az), "h": -5.0 * np.cos(az)}
    V, r2, inl, is_ego = G.ego_motion_fit(radar)
    assert is_ego and abs(V - 5.0) < 0.1


def test_ego_motion_static_scene_not_ego():
    rng = np.random.default_rng(0)
    az = rng.uniform(-0.6, 0.6, 400)
    radar = {"x": 30 * np.cos(az), "y": 30 * np.sin(az), "h": rng.normal(0, 0.05, 400)}
    assert not G.ego_motion_fit(radar)[3]


def test_doppler_transfer_anisotropic():
    lidar = np.array([[20.0, 0, 0], [20.0, 0, 1.2], [20.0, 0.9, 0], [30.0, 0, 0]])
    radar = np.array([[20.0, 0, 0]])
    out = G.doppler_transfer(lidar, radar, np.array([1.0]))
    assert np.allclose(out[:3], 1.0) and np.isnan(out[3])


def test_radar_range_estimates_nearest_cluster():
    z = np.r_[np.full(4, 18.0) + np.arange(4) * 0.1, np.full(10, 70.0)]
    e = radar_range_estimates(z, prior=17.0)
    assert abs(e["nearest"] - 18.15) < 0.2 and abs(e["prior"] - 18.15) < 0.2 and e["median"] > 60


def test_tracker_and_kalman():
    tr = IoUTracker()
    a = tr.update(0, np.array([[0, 0, 10, 10.0]]))
    b = tr.update(1, np.array([[1, 0, 11, 10.0]]))
    assert a == b
    t = np.arange(50) * 0.1
    z = np.stack([t * 1.0, np.zeros(50), 20 + np.zeros(50)], 1)
    p, v = kalman_cv_smooth(t, z, np.full(50, 0.1))
    assert abs(v[25, 0] - 1.0) < 0.05
