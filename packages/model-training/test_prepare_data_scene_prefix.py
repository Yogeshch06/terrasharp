from pathlib import Path

from prepare_data import derive_scene_prefix


def test_scene_prefix_uses_tile_id_and_date_from_safe_name() -> None:
    safe_dir = Path("S2A_MSIL2A_20260823T053241_N0512_R105_T43PCS_20260823T122218.SAFE")

    assert derive_scene_prefix(safe_dir) == "T43PCS_20260823"
