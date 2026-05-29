#!/usr/bin/env python3
"""
zimage_worker.py — ZImage subprocess worker for oh-my-comic.

This script is launched as a child process by app.py when
IMAGE_GENERATION_MODE=zimage_hybrid and ZIMAGE_RUN_MODE=subprocess.

It loads ZImagePipeline, generates all requested images serially,
and writes results to a JSON file after EACH image so the parent
process can update the UI progressively (one image at a time).

When this process exits, the OS reclaims all CPU RAM and CUDA memory
that ZImage allocated — including accelerate offload hooks, transformer
buffers, flash attention workspaces, and PyTorch CPU allocator caches.
This ensures Qwen Edit (loaded in the parent process afterwards) is not
slowed down by ZImage memory residue.

Usage (called by app.py, not directly):
    python zimage_worker.py --tasks <tasks_json_file> --results <results_json_file>
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def _write_results_atomic(results_path: Path, results: list):
    """Write results atomically: write to .tmp then rename."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = results_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)
    tmp_path.replace(results_path)


def main():
    parser = argparse.ArgumentParser(description="ZImage worker process")
    parser.add_argument("--tasks",   required=True, help="Path to tasks JSON file")
    parser.add_argument("--results", required=True, help="Path to results JSON file")
    args = parser.parse_args()

    tasks_path   = Path(args.tasks)
    results_path = Path(args.results)

    if not tasks_path.exists():
        print(f"[zimage_worker] ERROR: tasks file not found: {tasks_path}", file=sys.stderr)
        sys.exit(1)

    with open(tasks_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    cfg   = payload.get("config", {})
    tasks = payload.get("tasks", [])

    if not tasks:
        print("[zimage_worker] No tasks, exiting.", file=sys.stderr)
        _write_results_atomic(results_path, [])
        sys.exit(0)

    # ── Load ZImage ──────────────────────────────────────────────────────────
    model_path         = cfg.get("model_path", "")
    dtype_str          = cfg.get("dtype", "bfloat16")
    steps              = int(cfg.get("steps", 9))
    guidance_scale     = float(cfg.get("guidance_scale", 0.0))
    attention_backend  = cfg.get("attention_backend", "flash")
    negative_prompt    = cfg.get("negative_prompt", "")
    enable_cpu_offload = bool(cfg.get("enable_cpu_offload", True))

    print(f"[zimage_worker] Loading ZImage from: {model_path}", file=sys.stderr)
    print(f"[zimage_worker] dtype={dtype_str}, steps={steps}, "
          f"guidance_scale={guidance_scale}, cpu_offload={enable_cpu_offload}",
          file=sys.stderr)

    try:
        import torch
        from diffusers import ZImagePipeline

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
        }
        torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)

        pipe = ZImagePipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        pipe.to("cuda")

        if attention_backend:
            try:
                pipe.transformer.set_attention_backend(attention_backend)
                print(f"[zimage_worker] Attention backend: {attention_backend}", file=sys.stderr)
            except Exception as e:
                print(f"[zimage_worker] WARNING: attention backend '{attention_backend}' failed: {e}",
                      file=sys.stderr)

        if enable_cpu_offload:
            pipe.enable_model_cpu_offload()
            print("[zimage_worker] CPU offload enabled", file=sys.stderr)

    except Exception as e:
        print(f"[zimage_worker] FATAL: failed to load ZImage: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Write all tasks as failed immediately so parent can update UI
        results = [
            {
                "seg_id":     t.get("seg_id"),
                "image_type": t.get("image_type"),
                "item_id":    t.get("item_id"),
                "status":     "failed",
                "url":        None,
                "filename":   t.get("filename"),
            }
            for t in tasks
        ]
        _write_results_atomic(results_path, results)
        sys.exit(1)

    # ── Generate images serially, writing results after each image ───────────
    results = []
    generated_dir = Path(cfg.get("generated_dir", "static/generated"))
    generated_dir.mkdir(parents=True, exist_ok=True)

    for idx, task in enumerate(tasks):
        seg_id     = task.get("seg_id")
        image_type = task.get("image_type")
        item_id    = task.get("item_id")
        prompt     = task.get("prompt", "")
        filename   = task.get("filename", f"zimage_{idx}.png")
        width      = int(task.get("width",  768))
        height     = int(task.get("height", 1024))

        print(f"[zimage_worker] Generating {idx+1}/{len(tasks)}: {item_id} "
              f"({width}x{height})", file=sys.stderr)

        out_path = generated_dir / filename
        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
            )
            image = result.images[0]
            image.save(out_path)
            del image
            del result

            url = f"/static/generated/{filename}"
            results.append({
                "seg_id":     seg_id,
                "image_type": image_type,
                "item_id":    item_id,
                "status":     "done",
                "url":        url,
                "filename":   filename,
            })
            print(f"[zimage_worker] Saved: {filename}", file=sys.stderr)

        except Exception as e:
            print(f"[zimage_worker] ERROR generating {filename}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            results.append({
                "seg_id":     seg_id,
                "image_type": image_type,
                "item_id":    item_id,
                "status":     "failed",
                "url":        None,
                "filename":   filename,
            })

        # Write results after EACH image so parent can update UI progressively.
        # Atomic write (tmp -> rename) prevents parent from reading a partial file.
        _write_results_atomic(results_path, results)
        print(f"[zimage_worker] Results updated ({len(results)}/{len(tasks)})", file=sys.stderr)

    print(f"[zimage_worker] Done. {len(results)} tasks processed.", file=sys.stderr)
    # Process exits here — OS reclaims all ZImage memory.


if __name__ == "__main__":
    main()