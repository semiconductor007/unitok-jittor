#!/usr/bin/env bash
set -e
python -m jittor_unitok.engine.train_tokenizer --config configs/unitok_tiny.yaml --no-cuda

