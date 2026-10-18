"""Parse physical-hand output switches without loading a hardware runtime."""

import sys

import pytest

from g1_teleop.bridge.cli import parse_args


@pytest.mark.parametrize(
    "options, expected",
    [
        ([], False),
        (["--enable-inspire-hand-dds"], True),
        (["--disable-inspire-hand-dds"], False),
        (["--no-inspire-hand-dds"], False),
    ],
)
def test_physical_hand_dds_is_opt_in(monkeypatch, options, expected):
    monkeypatch.setattr(sys, "argv", ["bridge", *options])

    args = parse_args()

    assert args.enable_inspire_hand_dds is expected
    assert args.enable_inspire_hand_sim is False


@pytest.mark.parametrize("disable", ["--disable-inspire-hand-dds", "--no-inspire-hand-dds"])
@pytest.mark.parametrize("enable_first", [True, False])
def test_conflicting_physical_hand_switches_are_rejected(monkeypatch, disable, enable_first):
    options = ["--enable-inspire-hand-dds", disable]
    if not enable_first:
        options.reverse()
    monkeypatch.setattr(sys, "argv", ["bridge", *options])

    with pytest.raises(SystemExit) as error:
        parse_args()

    assert error.value.code == 2


@pytest.mark.parametrize("disable", ["--disable-inspire-hand-dds", "--no-inspire-hand-dds"])
@pytest.mark.parametrize(
    "sim_option, sim_enabled",
    [("--enable-inspire-hand-sim", True), ("--no-inspire-hand-sim", False)],
)
def test_disabling_physical_hands_preserves_sim_selection(
    monkeypatch, disable, sim_option, sim_enabled
):
    monkeypatch.setattr(sys, "argv", ["bridge", disable, sim_option])

    args = parse_args()

    assert args.enable_inspire_hand_dds is False
    assert args.enable_inspire_hand_sim is sim_enabled
