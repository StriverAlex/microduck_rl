import importlib.util
import sys
from pathlib import Path

import pytest

from mjlab_microduck.tasks.microduck_ball_slalom_env_cfg import (
    make_microduck_ball_slalom_env_cfg,
)
from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import BALL_FIELD_TEXTURE

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "render_ball_slalom_multi", REPO / "scripts" / "render_ball_slalom_multi.py"
)
assert SPEC is not None and SPEC.loader is not None
render = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = render
SPEC.loader.exec_module(render)


def test_multi_env_replay_cfg_uses_irregular_course_and_world_camera():
    scenario = render.GENERALIZATION_SCENARIOS["five_irregular"]
    cfg = render.configure_multi_env_cfg(
        make_microduck_ball_slalom_env_cfg(play=True),
        scenario,
        num_envs=8,
        seed=789,
    )

    assert cfg.scene.num_envs == 8
    assert cfg.scene.env_spacing == 3.0
    assert cfg.commands["body_pose"].cone_x == scenario.cone_x
    assert cfg.viewer.origin_type == cfg.viewer.OriginType.WORLD
    assert cfg.viewer.entity_name is None
    assert cfg.viewer.body_name is None
    assert cfg.viewer.max_extra_envs == 7
    assert (cfg.viewer.width, cfg.viewer.height) == (1280, 720)
    assert cfg.scene.terrain.textures[0] is BALL_FIELD_TEXTURE
    assert cfg.scene.terrain.materials[0].texrepeat == (1.0, 1.0)
    assert cfg.scene.terrain.materials[0].reflectance == 0.0


def test_multi_env_replay_requires_more_than_one_environment():
    with pytest.raises(ValueError, match="at least two"):
        render.configure_multi_env_cfg(
            make_microduck_ball_slalom_env_cfg(play=True),
            render.GENERALIZATION_SCENARIOS["five_irregular"],
            num_envs=1,
            seed=789,
        )
