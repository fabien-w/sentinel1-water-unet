import torch

from s1water.model import KERAS_BN_EPS, UNetS1Water


def test_parameter_count():
    # 34,600,641 weights + biases + BN gamma/beta, as in the converted Keras weights
    m = UNetS1Water()
    assert sum(p.numel() for p in m.parameters()) == 34_600_641
    assert sum(isinstance(x, torch.nn.Conv2d) for x in m.modules()) == 28


def test_output_shape_and_range():
    m = UNetS1Water().eval()
    x = torch.rand(1, 4, 64, 64)
    with torch.no_grad():
        y = m(x)
        yc = m(x, crop=8)
    assert y.shape == (1, 1, 64, 64)
    assert yc.shape == (1, 1, 48, 48)
    assert torch.all((y >= 0) & (y <= 1))
    assert torch.allclose(y[..., 8:-8, 8:-8], yc)


def test_keras_batchnorm_epsilon():
    m = UNetS1Water()
    eps = {bn.eps for bn in m.modules() if isinstance(bn, torch.nn.BatchNorm2d)}
    assert eps == {KERAS_BN_EPS}


def test_state_dict_roundtrip(tmp_path):
    from s1water.model import load_model

    m = UNetS1Water()
    torch.save(m.state_dict(), tmp_path / "w.pt")
    m2 = load_model(tmp_path / "w.pt")
    for (k, a), (_, b) in zip(m.state_dict().items(), m2.state_dict().items(), strict=True):
        assert torch.equal(a, b), k
