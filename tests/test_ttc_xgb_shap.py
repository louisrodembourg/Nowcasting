from src.correlation.ttc_xgb_shap import (
    FEATURE_COLS,
    TARGET_COL,
    _make_synthetic_dataset,
    run_full_pipeline,
    split_train_test_chronological,
)


def test_synthetic_dataset_matches_the_model_schema():
    df = _make_synthetic_dataset(n_episodes=30)
    assert set(FEATURE_COLS + [TARGET_COL]) <= set(df.columns)
    assert (df[TARGET_COL] >= 1.0).all()


def test_split_is_chronological():
    df = _make_synthetic_dataset(n_episodes=40)
    *_, df_train, df_test = split_train_test_chronological(df, test_frac=0.25)
    assert len(df_test) == 10
    assert df_train["start_date"].max() < df_test["start_date"].min()


def test_full_pipeline_learns_the_synthetic_signal(tmp_path):
    results = run_full_pipeline(df=_make_synthetic_dataset(n_episodes=80), output_dir=tmp_path)
    assert results["metrics"]["r2"] > 0.3
    for key in ("shap_summary_path", "shap_waterfall_path", "scatter_path"):
        assert results[key].exists()
