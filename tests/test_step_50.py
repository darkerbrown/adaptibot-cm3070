from adaptibot import controller, run_config

def test_step_50():
    cfg = run_config.RunConfig(
        render_mode="rgb_array",
        steps=50,
        seed=123,
        gui=False,
        model_path="ppo_fix_continuous_action.cleanrl_model",
    )
    metrics = controller.run(cfg)
    assert isinstance(metrics, dict)
    assert metrics.get("steps", 0) >= 50
