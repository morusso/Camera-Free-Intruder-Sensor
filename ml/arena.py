#!/usr/bin/env python3
"""Estimate the TFLite Micro tensor arena of an int8 model (step 3.7).

Activation tensors (the model input and every operator output) live in the arena; weights stay in flash.
Each activation is alive from the operator that writes it to the last operator that reads it. Tensors
that are never alive at the same time share memory. The placement follows the greedy-by-size idea of
the TFLite Micro planner, with 16-byte alignment.

This is a LOWER bound for the arena: TFLite Micro also keeps persistent data (operator state,
quantization parameters, bookkeeping) and some operators need scratch buffers. So the budget check
adds a reserve. The real number is read on the device with arena_used_bytes() in step 4.3.

Example:
    python ml/arena.py ml/runs/20261112_cnn_s0/model_int8.tflite
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import typer

ALIGN = 16
BUDGET_KB = 80


@dataclass
class Buffer:
    name: str
    size: int      # bytes, before alignment
    first: int     # index of the operator that writes it (0 for the model input)
    last: int      # index of the last operator that reads it


def aligned(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def plan(buffers: List[Buffer]) -> Tuple[Dict[str, int], int]:
    """Greedy-by-size placement. Returns (offset of every buffer, total arena size in bytes)."""
    placed: List[Tuple[Buffer, int]] = []
    offsets: Dict[str, int] = {}
    for b in sorted(buffers, key=lambda b: (-b.size, b.first)):
        size = aligned(b.size)
        # Buffers that are alive at the same time as b: they block their address range.
        busy = sorted((off, off + aligned(o.size)) for o, off in placed
                      if not (o.last < b.first or b.last < o.first))
        offset = 0
        for start, end in busy:
            if offset + size <= start:
                break  # fits into the gap before this block
            offset = max(offset, end)
        placed.append((b, offset))
        offsets[b.name] = offset
    total = max((off + aligned(b.size) for b, off in placed), default=0)
    return offsets, total


def buffers_from_tflite(tflite_bytes: bytes) -> Tuple[List[Buffer], List[str]]:
    """Activation buffers and the list of operators of a TFLite model."""
    import tensorflow as tf

    # No default delegates: XNNPACK would replace part of the graph with one DELEGATE node,
    # but TFLite Micro runs every operator as it is stored in the file.
    interp = tf.lite.Interpreter(
        model_content=tflite_bytes,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    interp.allocate_tensors()
    details = {d["index"]: d for d in interp.get_tensor_details()}
    ops = interp._get_ops_details()  # private API, but the only way to get the operator graph

    first: Dict[int, int] = {i["index"]: 0 for i in interp.get_input_details()}
    last: Dict[int, int] = {}
    for n, op in enumerate(ops):
        for t in op["outputs"]:
            first.setdefault(int(t), n)
        for t in op["inputs"]:
            last[int(t)] = n
    for o in interp.get_output_details():
        last[o["index"]] = len(ops)  # the caller reads the result after the last operator

    out = []
    for idx, f in first.items():
        d = details[idx]
        size = int(np.prod(d["shape"])) * np.dtype(d["dtype"]).itemsize
        out.append(Buffer(d["name"], size, f, last.get(idx, f)))
    return out, [op["op_name"] for op in ops]


def report(tflite_bytes: bytes, reserve_kb: float) -> Tuple[str, bool]:
    buffers, op_names = buffers_from_tflite(tflite_bytes)
    _, arena = plan(buffers)
    naive = sum(aligned(b.size) for b in buffers)
    total_kb = arena / 1024 + reserve_kb
    ok = total_kb <= BUDGET_KB
    lines = [
        f"model (flash)          : {len(tflite_bytes) / 1024:.1f} KB",
        f"activation buffers     : {len(buffers)}",
        f"without sharing        : {naive / 1024:.1f} KB",
        f"planned tensor arena   : {arena / 1024:.1f} KB   (lower bound)",
        f"+ reserve for TFLM     : {reserve_kb:.0f} KB",
        f"= estimate             : {total_kb:.1f} KB   [{'PASS' if ok else 'FAIL'}: <= {BUDGET_KB} KB]",
        "operators for the resolver: " + ", ".join(sorted(set(op_names))),
        "(real value: arena_used_bytes() on the device, step 4.3)",
    ]
    return "\n".join(lines), ok


app = typer.Typer(add_completion=False, help="Estimate the TFLite Micro tensor arena of an int8 model.")


@app.command()
def main(
    model: Path = typer.Argument(..., exists=True, dir_okay=False, help="model_int8.tflite"),
    reserve_kb: float = typer.Option(16.0, min=0.0, help="Allowance for persistent data and scratch buffers."),
) -> None:
    text, ok = report(model.read_bytes(), reserve_kb)
    typer.echo(text)
    if not ok:
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
