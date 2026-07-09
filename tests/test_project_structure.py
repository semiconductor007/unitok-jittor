from pathlib import Path


def test_required_project_files_exist():
    root = Path(__file__).resolve().parents[1]
    required = [
        "README.md",
        "requirements.txt",
        "environment.yml",
        "configs/unitok_tiny.yaml",
        "docs/pytorch_code_analysis.md",
        "jittor_unitok/models/mcq.py",
        "jittor_unitok/models/attention_projection.py",
        "jittor_unitok/models/tokenizer.py",
        "scripts/prepare_demo_data.py",
        "scripts/plot_loss.py",
    ]
    missing = [path for path in required if not (root / path).exists()]
    assert not missing

