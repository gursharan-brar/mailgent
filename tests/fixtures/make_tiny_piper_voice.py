"""Regenerate tiny_piper_voice.onnx(.json): a stand-in Piper voice for tests.

It has the real Piper model interface (inputs `input`, `input_lengths`, `scales`; float audio
output) but no neural network: it emits 256 samples of a constant level per phoneme id. It
lets tests run the real piper-tts library (espeak phonemizer, onnxruntime, WAV writing)
offline. It does not produce speech.

    pip install onnx && python tests/fixtures/make_tiny_piper_voice.py
"""

import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper
from piper.phoneme_ids import DEFAULT_PHONEME_ID_MAP

HERE = Path(__file__).parent
SAMPLES_PER_ID = 256

nodes = [
    helper.make_node("Cast", ["input"], ["as_float"], to=TensorProto.FLOAT),
    helper.make_node("Mul", ["as_float", "level"], ["scaled"]),
    helper.make_node("Tile", ["scaled", "repeats"], ["tiled"]),
    helper.make_node("Unsqueeze", ["tiled", "axis1"], ["output"]),
]
graph = helper.make_graph(
    nodes,
    "tiny_piper_voice",
    inputs=[
        helper.make_tensor_value_info("input", TensorProto.INT64, [1, "n"]),
        helper.make_tensor_value_info("input_lengths", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("scales", TensorProto.FLOAT, [3]),
    ],
    outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, "t"])],
    initializer=[
        helper.make_tensor("level", TensorProto.FLOAT, [], [0.001]),
        helper.make_tensor("repeats", TensorProto.INT64, [2], [1, SAMPLES_PER_ID]),
        helper.make_tensor("axis1", TensorProto.INT64, [1], [1]),
    ],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, HERE / "tiny_piper_voice.onnx")

config = {
    "audio": {"sample_rate": 16000},
    "espeak": {"voice": "en-us"},
    "inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8},
    "phoneme_type": "espeak",
    "phoneme_id_map": DEFAULT_PHONEME_ID_MAP,
    "num_symbols": 256,
    "num_speakers": 1,
    "piper_version": "test-fixture",
}
(HERE / "tiny_piper_voice.onnx.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
print("wrote", HERE / "tiny_piper_voice.onnx")
