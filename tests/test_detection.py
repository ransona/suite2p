import numpy as np

from suite2p.detection import sparsedetect


def test_sparsery_forced_scale_controls_vcorr_and_peak_search():
    rng = np.random.default_rng(2)
    mov = rng.normal(scale=0.1, size=(60, 32, 32)).astype("float32")
    mov[10:30, 15:18, 15:18] += 5
    sdmov = mov.std(axis=0).astype("float32") + 1e-3

    settings = dict(highpass_neuropil=3, threshold_scaling=0.1, max_ROIs=3)
    scale1, stat1 = sparsedetect.sparsery(
        mov.copy(), sdmov, spatial_scale=1, **settings
    )
    scale4, _ = sparsedetect.sparsery(
        mov.copy(), sdmov, spatial_scale=4, **settings
    )

    assert scale1["spatscale_pix"] == 6
    assert scale4["spatscale_pix"] == 48
    assert not np.allclose(scale1["Vcorr"], scale4["Vcorr"])
    assert len(stat1) > 0
    assert all(stat["footprint"] == 1 for stat in stat1)
