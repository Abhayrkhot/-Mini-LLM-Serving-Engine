# Mini LLM Serving Engine

## Overview

This project implements a **mini LLM serving engine simulator** using
plain Python and `asyncio`.  It models the key components of a real
inference server—request queuing, a concurrency‑limited scheduler and
token‑level decoding—without relying on any external web framework.

In production, LLM serving systems such as vLLM, TensorRT‑LLM and
SGLang accept streams of generation requests, schedule them onto
available GPU capacity and return tokens as soon as they are generated.
At their core, these systems perform three tasks:

1. **Queue requests.**  Incoming prompts are queued until resources are
   available.  Real servers often process hundreds or thousands of
   concurrent requests.
2. **Schedule decoding.**  A scheduler selects which requests can be
   decoded in parallel based on GPU memory constraints and latency
   budgets.  The model generates one token at a time for all active
   requests.
3. **Stream tokens.**  Tokens are sent back to the caller as soon as
   they are produced to minimise latency.

The `InferenceEngine` class in this project provides a simplified version
of these steps.  It uses an `asyncio.Semaphore` to limit the number of
concurrent decode tasks (mimicking a GPU with limited capacity) and
emits one token at a time with a configurable delay to simulate
generation latency.  When all tokens have been produced, a final JSON
summary is yielded with timing information.  This design helps you
understand how queuing, scheduling and streaming interact without
needing a GPU or external dependencies.

## Why this matters

Inference engineering roles require understanding more than just how to
call an API.  Engineers must design systems that can accept many
concurrent requests, optimise throughput under GPU memory constraints,
respect latency service‑level agreements (SLAs) and stream responses
incrementally.  Building a toy serving engine gives hands‑on experience
with these constraints and vocabulary: request queues, schedulers,
prefill versus decode, concurrency limits and streaming responses.  A
simple simulator like this lets you observe how changing the concurrency
limit or decode delay affects throughput and latency, which is
fundamental to designing scalable inference backends.

## Features

* **Concurrency‑limited decoding.**  The `InferenceEngine` uses an
  `asyncio.Semaphore` to limit the number of concurrent decode tasks.
  This models a GPU with limited capacity: if the semaphore is set to
  `2`, only two requests will be decoded at the same time, while others
  wait in the queue.
* **Token streaming.**  Tokens are yielded one by one with a
  configurable delay to simulate model decoding latency.  After the
  last token, a JSON summary containing the request ID, start/end
  timestamps and duration is emitted.
* **Metrics tracking.**  The engine collects simple per‑request metrics
  such as start time, end time, number of tokens and duration.  These
  can be inspected via the `metrics` property after a batch of
  requests completes.
* **Demonstration CLI.**  Running the module directly executes a demo
  that submits a few concurrent requests, prints each streamed token and
  prints the collected metrics.  This helps you see how concurrent
  requests interleave and how the semaphore affects throughput.
* **Test suite** using `pytest` and `asyncio` to validate that the
  engine streams tokens correctly and enforces the concurrency limit.

## Project structure

```text
03-mini-llm-serving-engine/
├── server.py         # Defines the InferenceEngine and demo CLI
├── tests/
│   └── test_server.py
├── requirements.txt
└── README.md
```

## Installation

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the demo

You can run the demonstration CLI to see how the engine behaves under
concurrent load.  Execute the module directly:

```bash
python server.py
```

The script will submit three concurrent requests and print each token
as it is generated.  At the end, it prints a list of per‑request
metrics.  You can modify the prompts, number of tokens or the
`max_active` and `decode_delay` parameters in the code to experiment
with different concurrency limits and latencies.

## Running the tests

Run the unit tests with `pytest`:

```bash
pytest -q
```

The tests exercise the `InferenceEngine` directly.  They verify that
tokens are streamed in order and that the concurrency limit enforces
serialisation when the semaphore size is set to one.  This ensures that
the simulator correctly models request queuing and scheduling.

## Resume bullet

Built a mini LLM serving engine simulator using `asyncio`, featuring a
request queue, semaphore‑controlled concurrency limit, token streaming
and per‑request metrics, with a CLI demonstration and `pytest` tests to
validate queuing and throughput behaviour.
