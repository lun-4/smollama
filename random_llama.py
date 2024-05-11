#!/usr/bin/env python3
import sys
import random
import json
import os
from pathlib import Path
from enum import IntEnum

import numpy as np


if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / "gguf-py"))
import gguf


from gguf import GGUFWriter  # noqa: E402


class SentencePieceTokenTypes(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


n_embd = 2048
n_ff = 1000
n_blocks = 2  # aka n_layers
n_vocab = 32000
n_head = 32
assert n_head == 32  # TODO make it work for different n_head values
n_head_kv = 4
n_embd_head_k = 64
n_embd_head_v = 64
assert n_embd_head_k % n_head_kv == 0
n_embd_gqa = int(n_embd_head_v * n_head_kv)
assert n_embd_gqa == 256
n_expert = 0


def write_tensors(gguf_writer) -> None:
    map = gguf.TensorNameMap(gguf.MODEL_ARCH.LLAMA, n_blocks)
    llama_tensors = gguf.MODEL_TENSORS[gguf.MODEL_ARCH.LLAMA]
    for bid in range(n_blocks):
        print("block", bid)
        for tensor_type in llama_tensors:
            # if it has {bid} in its name, then cook!
            tensor_name = gguf.TENSOR_NAMES[tensor_type]
            tensor_full_name = None
            bias_shape = None

            if "{bid}" in tensor_name:
                tensor_full_name = tensor_name.format(bid=bid)
                tensor_shape = (32,)
                if tensor_name.endswith("attn_norm"):
                    tensor_shape = (n_embd,)
                elif tensor_name.endswith("attn_q"):
                    tensor_shape = (n_embd, n_embd)
                    bias_shape = (n_embd,)
                elif tensor_name.endswith("attn_k"):
                    tensor_shape = (n_embd_gqa, n_embd)
                    bias_shape = (n_embd_gqa,)
                elif tensor_name.endswith("attn_v"):
                    tensor_shape = (n_embd_gqa, n_embd)
                    bias_shape = (n_embd_gqa,)
                elif tensor_name.endswith("attn_output"):
                    tensor_shape = (n_embd, n_embd)
                    bias_shape = (n_embd,)
                elif tensor_name.endswith("ffn_norm"):
                    tensor_shape = (n_embd,)
                elif tensor_name.endswith("ffn_gate"):
                    tensor_shape = (n_ff, n_embd)
                elif tensor_name.endswith("ffn_down"):
                    tensor_shape = (n_embd, n_ff)
                elif tensor_name.endswith("ffn_up"):
                    tensor_shape = (n_ff, n_embd)
                elif tensor_name.endswith("attn_rot_embd"):
                    # what the fuck is this and why does llama.cpp lists it (on an internal tensor map) yet doesn't actually load it?
                    # if i emit this on the model file, llama.cpp will explode saying that the layers found and layers loaded
                    # are different. of course they are, you're not loading attn_rot_embd!!
                    tensor_full_name = None
                    tensor_shape = (n_embd, n_embd)
                elif any(
                    # no experts allowed
                    tensor_name.endswith(suffix)
                    for suffix in (
                        "ffn_gate_inp",
                        "ffn_gate_exps",
                        "ffn_down_exps",
                        "ffn_up_exps",
                        "ffn_gate_exp",
                        "ffn_down_exp",
                        "ffn_up_exp",
                    )
                ):
                    # ignore these, not MoE
                    continue
                else:
                    raise Exception(f"TODO {tensor_name}")
            elif bid == 0:
                tensor_full_name = tensor_name
                if "token_embd" in tensor_name:
                    tensor_shape = (n_vocab, n_embd)
                elif "output_norm" in tensor_name:
                    tensor_shape = (n_embd,)
                elif "output" in tensor_name:
                    tensor_shape = (n_vocab, n_embd)
                else:
                    tensor_full_name = None

            if tensor_full_name:
                print("create", tensor_full_name + ".weight", "shape", tensor_shape)
                # tensor? i made it the fuck up
                tensor_value = np.ones(tensor_shape, dtype=np.float32) * random.uniform(
                    -5, 5
                )
                gguf_writer.add_tensor(tensor_full_name + ".weight", tensor_value)
                if bias_shape:
                    print("create", tensor_full_name + ".bias", "shape", bias_shape)
                    another_tensor_value = np.ones(
                        bias_shape, dtype=np.float32
                    ) * random.uniform(-5, 5)
                    gguf_writer.add_tensor(
                        tensor_full_name + ".bias", another_tensor_value
                    )
            else:
                print("skip", tensor_name)


def writer_example() -> None:
    # Example usage with a file
    gguf_writer = GGUFWriter("/tmp/example.gguf", "llama")

    gguf_writer.add_architecture()
    gguf_writer.add_name("cooking-500M")
    gguf_writer.add_block_count(n_blocks)
    gguf_writer.add_context_length(2048)
    gguf_writer.add_embedding_length(n_embd)
    gguf_writer.add_feed_forward_length(n_ff)
    gguf_writer.add_head_count(n_head)
    gguf_writer.add_head_count_kv(n_head_kv)
    gguf_writer.add_rope_freq_base(10000.0)
    gguf_writer.add_layer_norm_rms_eps(1e-05)
    gguf_writer.add_file_type(gguf.GGMLQuantizationType.F16)
    gguf_writer.add_expert_count(n_expert)

    gguf_writer.add_tokenizer_model("llama")

    with open("./tokenizer.json", "r") as fd:
        tokenizer = json.load(fd)

    assert tokenizer["version"] == "1.0"

    tokens = []
    scores = []
    toktypes = []
    for token_str, token_id in tokenizer["model"]["vocab"].items():
        tokens.append(token_str)
        scores.append(token_id)
        toktypes.append(SentencePieceTokenTypes.NORMAL)

    gguf_writer.add_token_list(tokens)
    gguf_writer.add_token_scores(scores)
    gguf_writer.add_token_types(toktypes)

    write_tensors(gguf_writer)

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()

    gguf_writer.close()


if __name__ == "__main__":
    writer_example()
