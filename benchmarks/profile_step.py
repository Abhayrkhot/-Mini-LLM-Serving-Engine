"""Where a decode step's time goes, with and without async scheduling.

For each batch size B, B requests decode in lockstep and we measure, per step:

  step   wall time of ``engine.step()``, i.e. the inter-token latency
  GPU    device time of the step's GPU work alone: measured with CUDA events
         while a spin kernel keeps the GPU busy until the whole step has been
         queued, so CPU launch gaps do not count
  CPU    the engine's own CPU time per step: scheduling and launching
         (``schedule`` + ``execute_async``), applying tokens (``update``) and
         bookkeeping, i.e. everything except waiting for the GPU

Synchronously a step costs about CPU + GPU; with async scheduling, about
max(CPU, GPU). ``--kernels B`` also lists the GPU kernels of one decode step
at batch size B, by total time.

    python benchmarks/profile_step.py --model Qwen/Qwen2.5-0.5B-Instruct --batch-sizes 1 4 16 64 256
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from typing import Any

import torch

from minillm import EngineConfig, LLMEngine, SamplingParams


class PhaseTimer:
    """Wraps the engine's scheduler and executor to time each phase."""

    def __init__(self, engine: LLMEngine) -> None:
        self.times: dict[str, float] = {"launch": 0.0, "wait": 0.0, "apply": 0.0, "replay": 0.0}
        sched, executor = engine.scheduler, engine.executor
        schedule, update, execute_async = sched.schedule, sched.update, executor.execute_async

        def timed(name: str, fn: Any) -> Any:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                t0 = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    self.times[name] += time.perf_counter() - t0

            return wrapper

        class Handle:
            def __init__(self, inner: Any) -> None:
                self.result = timed("wait", inner.result)

        sched.schedule = timed("launch", schedule)  # type: ignore[method-assign]
        sched.update = timed("apply", update)  # type: ignore[method-assign]
        executor.execute_async = timed(  # type: ignore[method-assign]
            "launch", lambda scheduled: Handle(execute_async(scheduled))
        )
        graphs = getattr(executor, "graphs", None)  # CUDA-graph replays, part of "launch"
        for graph in graphs.graphs.values() if graphs is not None else ():
            graph.replay = timed("replay", graph.replay)

    def reset(self) -> None:
        self.times = dict.fromkeys(self.times, 0.0)


def spin_cycles_per_ms() -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(1_000_000)  # warm-up
    start.record()
    torch.cuda._sleep(10_000_000)
    end.record()
    end.synchronize()
    return 10_000_000 / start.elapsed_time(end)


def gpu_step_ms(engine: LLMEngine, steps: int, cycles_per_ms: float) -> float:
    """Device time per step, from steps launched while a spin kernel holds the GPU."""
    runner = engine.executor
    times = []
    for _ in range(steps):
        output = engine.scheduler.schedule()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(int(20 * cycles_per_ms))  # longer than queuing one step takes
        start.record()
        handle = runner.execute_async(output.scheduled)
        end.record()
        tokens = handle.result()
        engine.scheduler.update(output, tokens)
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def add_requests(engine: LLMEngine, batch: int, args: argparse.Namespace) -> list[str]:
    rng = random.Random(batch)
    vocab = engine.executor.vocab_size
    params = SamplingParams(max_tokens=10 * args.steps + 64, temperature=0.0, ignore_eos=True)
    return [
        engine.add_request([rng.randrange(vocab) for _ in range(args.prompt_len)], params)
        for _ in range(batch)
    ]


def finish(engine: LLMEngine, ids: list[str]) -> None:
    for rid in ids:
        engine.abort_request(rid)
    engine.block_manager.reset_prefix_cache()


def kernel_table(engine: LLMEngine, batch: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    """GPU kernels of one decode step at ``batch``, grouped by name."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as torch_profile

    ids = add_requests(engine, batch, args)
    engine.async_scheduling = False
    for _ in range(args.warmup):
        engine.step()
    steps = 20
    with torch_profile(activities=[ProfilerActivity.CUDA]) as prof:
        gpu_step_ms(engine, steps, args.cycles_per_ms)
    rows = []
    for ev in prof.key_averages():
        if ev.device_type != DeviceType.CUDA or "sleep" in ev.key or "spin" in ev.key:
            continue
        total = getattr(ev, "self_device_time_total", None) or ev.self_cuda_time_total
        rows.append({"kernel": ev.key, "us": total / steps, "launches": ev.count / steps})
    finish(engine, ids)
    return sorted(rows, key=lambda r: -r["us"])


def profile(
    engine: LLMEngine, timer: PhaseTimer, batch: int, args: argparse.Namespace
) -> dict[str, Any]:
    ids = add_requests(engine, batch, args)
    row: dict[str, Any] = {"batch": batch}
    for mode in ("sync", "async"):
        engine.async_scheduling = mode == "async"
        for _ in range(args.warmup):  # finish the prefills, settle into decode
            engine.step()
        timer.reset()
        walls = []
        for _ in range(args.steps):
            t0 = time.perf_counter()
            engine.step()
            walls.append(time.perf_counter() - t0)
        row[f"{mode}_step_ms"] = 1e3 * statistics.median(walls)
        for phase, total in timer.times.items():
            row[f"{mode}_{phase}_ms"] = 1e3 * total / args.steps
        row[f"{mode}_cpu_ms"] = 1e3 * (sum(walls) - timer.times["wait"]) / args.steps
    engine.async_scheduling = False
    engine.step()  # collect the step in flight
    row["gpu_ms"] = gpu_step_ms(engine, args.steps, args.cycles_per_ms)
    finish(engine, ids)
    return row


def mixed_steps(
    engine: LLMEngine, batch: int, prompt: int, args: argparse.Namespace
) -> dict[str, Any]:
    """GPU time of a step that mixes ``batch`` decodes with one ``prompt``-token
    chunk, next to a decode-only step of the same batch and the chunk alone, plus
    a kernel breakdown of the mixed step."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as torch_profile

    ids = add_requests(engine, batch, args)
    engine.async_scheduling = False
    for _ in range(args.warmup):
        engine.step()
    decode_ms = gpu_step_ms(engine, 20, args.cycles_per_ms)
    rng = random.Random(prompt)
    vocab = engine.executor.vocab_size

    def chunk_ms(runs: int, alone: bool) -> list[float]:
        times = []
        for _ in range(runs):
            rid = engine.add_request(
                [rng.randrange(vocab) for _ in range(prompt)],
                SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True),
            )
            if alone:  # the chunk without the decodes: hold them back for one step
                held = engine.scheduler.running
                engine.scheduler.running = []
            times.append(gpu_step_ms(engine, 1, args.cycles_per_ms))
            if alone:
                engine.scheduler.running = held + engine.scheduler.running
            engine.abort_request(rid)
        return times

    alone = statistics.median(chunk_ms(10, alone=True))
    mixed = statistics.median(chunk_ms(10, alone=False))
    with torch_profile(activities=[ProfilerActivity.CUDA]) as prof:
        chunk_ms(5, alone=False)
    kernels: dict[str, float] = {}
    for ev in prof.key_averages():
        if ev.device_type != DeviceType.CUDA or "sleep" in ev.key or "spin" in ev.key:
            continue
        total = getattr(ev, "self_device_time_total", None) or ev.self_cuda_time_total
        kernels[ev.key] = total / 5 / 1e3
    finish(engine, ids)
    top = sorted(kernels.items(), key=lambda kv: -kv[1])[:8]
    return {
        "batch": batch,
        "prompt": prompt,
        "decode_ms": decode_ms,
        "chunk_alone_ms": alone,
        "mixed_ms": mixed,
        "top_kernels_ms": top,
    }


def cprofile_steps(engine: LLMEngine, batch: int, args: argparse.Namespace) -> str:
    """Python profile of ``args.steps`` async decode steps at ``batch``, top functions."""
    import cProfile
    import io
    import pstats

    ids = add_requests(engine, batch, args)
    engine.async_scheduling = True
    for _ in range(args.warmup):
        engine.step()
    prof = cProfile.Profile()
    prof.enable()
    for _ in range(args.steps):
        engine.step()
    prof.disable()
    engine.async_scheduling = False
    engine.step()
    finish(engine, ids)
    out = io.StringIO()
    pstats.Stats(prof, stream=out).sort_stats("tottime").print_stats(35)
    return out.getvalue()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--no-fused-kernels", dest="fused_kernels", action="store_false")
    p.add_argument("--quantization", choices=["w4a16"], default=None)
    p.add_argument("--kernels", type=int, default=None, metavar="B", help="kernel breakdown at B")
    p.add_argument("--cprofile", type=int, default=None, metavar="B", help="Python profile at B")
    p.add_argument(
        "--mixed",
        default=None,
        metavar="B,P",
        help="GPU time of B decodes plus one P-token prompt chunk in one step",
    )
    p.add_argument("--output", default=None)
    args = p.parse_args()

    engine = LLMEngine(
        EngineConfig(
            model=args.model,
            max_num_seqs=max(
                [*args.batch_sizes, int(args.mixed.split(",")[0]) + 1 if args.mixed else 1]
            ),
            max_num_batched_tokens=max(8192, args.prompt_len * max(args.batch_sizes) // 8),
            enforce_eager=args.enforce_eager,
            fused_kernels=args.fused_kernels,
            quantization=args.quantization,
        )
    )
    timer = PhaseTimer(engine)
    args.cycles_per_ms = spin_cycles_per_ms()
    rows = [profile(engine, timer, b, args) for b in args.batch_sizes]

    print(f"\n{args.model}, {args.prompt_len}-token prompts: ms per decode step")
    print("| batch | GPU | CPU | step, sync | step, async |")
    print("|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['batch']} | {r['gpu_ms']:.2f} | {r['async_cpu_ms']:.2f} "
            f"| {r['sync_step_ms']:.2f} | {r['async_step_ms']:.2f} |"
        )
    result: dict[str, Any] = {"model": args.model, "prompt_len": args.prompt_len, "rows": rows}
    if args.kernels:
        kernels = kernel_table(engine, args.kernels, args)
        total_us = sum(k["us"] for k in kernels)
        launches = sum(k["launches"] for k in kernels)
        print(
            f"\nGPU kernels per decode step at batch {args.kernels}: {launches:.0f} launches, "
            f"{total_us / 1e3:.2f} ms in total"
        )
        print("| kernel | us per step | launches per step |")
        print("|---|---:|---:|")
        for k in kernels[:20]:
            print(f"| `{k['kernel'][:70]}` | {k['us']:.0f} | {k['launches']:.0f} |")
        result["kernels"] = {"batch": args.kernels, "rows": kernels}
    if args.mixed:
        b, prompt = (int(x) for x in args.mixed.split(","))
        m = mixed_steps(engine, b, prompt, args)
        print(
            f"\nMixed step: {b} decodes + a {prompt}-token chunk: {m['mixed_ms']:.2f} ms GPU; "
            f"decodes alone {m['decode_ms']:.2f} ms, chunk alone {m['chunk_alone_ms']:.2f} ms"
        )
        for name, ms in m["top_kernels_ms"]:
            print(f"  {ms:7.3f} ms  {name[:90]}")
        result["mixed"] = m
    if args.cprofile:
        print(f"\nPython profile, {args.steps} async decode steps at batch {args.cprofile}:")
        print(cprofile_steps(engine, args.cprofile, args))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
