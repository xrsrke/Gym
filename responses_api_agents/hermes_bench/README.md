# HermesBench

HermesBench evaluates Hermes-Agent skills, tool use, memory, compaction,
safety, and artifact creation. It has 140 tasks: 92 automated, 33 hybrid,
7 LLM judge, and 8 architecture diagram.

## Benchmark Behavior

Preparation uses the original `scripts/lib_tasks.py`, task definitions,
fixtures, setup, grading, rubrics, and weights. Each rollout uses one fresh
agent sandbox and one fresh grader sandbox.

The 61 main-suite tasks use `lib_grading.grade_task`. The 79 skills tasks use
`hermes_skills_runner.call_grade`. Architecture tasks use the source judge.
The Hermes harness keeps the source prompt, toolsets, `max_turns: 90`, timeout
multiplier, memory, session, and skills-runner behavior. Only Hermes-Agent is
supported.

The adapter normalizes two malformed Excalidraw rubrics. A missing required
artifact is a valid model failure with reward 0. Other grading infrastructure
failures set `score_valid: false` and `mask_sample: true`.

The ZIP's `scores_and_models.txt` is not extracted. Judge credentials come
from runtime environment variables.

## Data Preparation

Requires Python 3.12 or newer, `uv`, and `npm`. Pass
`hermesbench-eval-clean.zip` directly to the script without extracting it.
Its SHA-256 is
`9f2edc889680801a5e77e2c13d18f62eb91b97384600964b03ebd2c749bd944e`.

One task also requires
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) at
commit `ca192cfb773915c9d8113352b8646d3ce9329424`.

```bash
uv run --frozen python responses_api_agents/hermes_bench/prepare.py \
  /path/to/hermesbench-eval-clean.zip \
  --hermes-agent-path /path/to/hermes-agent
```

This writes the JSONL, fixtures, and grading support to
`responses_api_agents/hermes_bench/data/`. The directory is ignored by Git.
The ZIP is not needed after preparation. Use `--task` or `--limit` for a
smaller dataset.

## Quickstart

```bash
export HERMES_BENCH_IMAGE=docker://registry.example/hermes-bench:pinned-tag
```

```bash
ng_run \
  "+config_paths=[responses_api_agents/hermes_bench/configs/hermes_bench_hermes.yaml,nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

```bash
ng_collect_rollouts \
  +agent_name=hermes_bench_hermes \
  +input_jsonl_fpath=responses_api_agents/hermes_bench/data/hermes_bench.jsonl \
  +output_jsonl_fpath=results/hermes_bench.jsonl \
  +limit=5
```

For Apptainer, replace the OpenSandbox provider config with:

```text
responses_api_agents/hermes_bench/configs/sandbox_apptainer.yaml
```

`agent_runtime_source: auto` builds the pinned Hermes-Agent runtime on Linux
x86_64. Use a baked or portable runtime on macOS.

The image needs `bash`, `tar`, Python 3, Git, PyYAML, `requests`, task
dependencies, and browser rendering dependencies. The default config forwards
`NOUS_API_KEY` and `OPENROUTER_API_KEY`.

## Test

```bash
uv run --frozen --extra dev pytest -q responses_api_agents/hermes_bench/tests
```
