from targetbook import cli


def test_demo_prints_the_waterfall(capsys):
    cli.main(["demo", "--seed", "1"])
    out = capsys.readouterr().out
    for stage in ("paper", "mandate", "vol_cap", "schedule", "costs", "carry", "bands"):
        assert f"\n{stage} " in out
    assert "staggered" in out
