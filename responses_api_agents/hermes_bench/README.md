# HermesBench

This adapter runs each HermesBench task in its own Gym sandbox. It keeps the
benchmark task definitions, fixtures, setup scripts, verifiers, rubrics, and
score aggregation from the source benchmark.

The source ZIP contains 140 tasks:

- 92 automated
- 33 hybrid
- 7 LLM judge
- 8 architecture diagram

The broad task groups are Hermes skills, tool gateway and tool calling,
memory and compaction, and safety.

## Execution

Each rollout uses two fresh sandboxes with the same configured provider:

1. The agent sandbox receives only the task fixture, task metadata, and agent
   runtime. It runs Gym's Hermes-Agent in `/sandbox/workspace`.
2. The completed workspace and trajectory are downloaded.
3. A fresh grader sandbox receives the completed workspace and the original
   HermesBench grading libraries.
4. The original main-suite or skills-suite verifier path computes the reward.

The outer sandbox is the isolation boundary. The inner Hermes terminal backend
is therefore `local`, which avoids a nested container while preserving the
task-visible isolated filesystem. Changing `sandbox_provider` selects
Apptainer, OpenSandbox, Docker, or another Gym sandbox provider without code
changes.

## Fidelity

Preparation imports the source benchmark's `scripts/lib_tasks.py`. It does not
reimplement task parsing. Generated rows preserve task prompts, per-task
toolsets, timeouts, setup scripts, grading code, rubrics, weights, and
architecture judge configuration.

The source `run_all.sh` splits the benchmark into 61 main-suite tasks and 79
Hermes skills tasks. Main-suite tasks use `lib_grading.grade_task`. Skills
tasks use the source `hermes_skills_runner.call_grade`, including its optional
`run_record`, and its original aggregation rules. Architecture tasks use the
source architecture judge.

The default Hermes settings match the source ZIP's Gym config:

- `max_turns: 90`
- core toolsets `terminal`, `file`, `code_execution`, `memory`, `skills`, and
  `session_search`
- the source skills-first system prompt

Tasks with an explicit `toolsets` field override the default toolset list while
retaining the source Gym config's global skills toolset. The default timeout
multiplier is 2.0, matching `run.sh` in the source benchmark.

The shared Gym Hermes agent normally disables memory, context files, and
session persistence. `preserve_hermes_behavior: true` corrects those settings
inside the task sandbox. Main-suite tasks use Hermes' source defaults with
memory and context enabled. Skills tasks keep the source skills runner's
`--ignore-rules` behavior. Both paths persist the session so trajectory-based
verifiers receive the complete run record. Only Hermes-Agent is supported
because the source benchmark specifically evaluates that harness.

Two source-data defects receive narrow adapter repairs. The two plain-numbered
Excalidraw rubrics are converted to the format expected by the source judge
parser. A missing required image or architecture artifact is a valid model
failure with reward 0. All other infrastructure grading failures return reward
0 with `score_valid: false` and `mask_sample: true`.

The ZIP's `scores_and_models.txt` is never extracted. Judge credentials are
accepted only through runtime environment variables. Safety-task `.env`
fixtures remain because they are graded adversarial inputs, not harness
credentials.

## Prepare

One architecture task references a Hermes Agent checkout that is not included
in the ZIP. Supply that fixture explicitly when preparing the full dataset.

```bash
python responses_api_agents/hermes_bench/prepare.py \
  /path/to/hermesbench-eval-clean.zip \
  --source-path \
  task_architecture_diagram_kanban_telegram_functions_extreme=/path/to/hermes-agent
```

The source checkout used for the supplied benchmark was commit
`ca192cfb773915c9d8113352b8646d3ce9329424`.

Preparation writes ignored local artifacts under `data/`:

- `hermes_bench.jsonl`
- one fixture archive per task
- one allowlisted grading support archive

Use `--task TASK_ID`, `--limit N`, or both for small iterations.

## Run

Set an image explicitly. The adapter does not choose or infer one.

```bash
export HERMES_BENCH_IMAGE=docker://registry.example/hermes-bench:latest

ng_run \
  "+config_paths=[responses_api_agents/hermes_bench/configs/hermes_bench_hermes.yaml,nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

ng_collect_rollouts \
  +agent_name=hermes_bench_hermes \
  +input_jsonl_fpath=responses_api_agents/hermes_bench/data/hermes_bench.jsonl \
  +output_jsonl_fpath=results/hermes_bench.jsonl \
  +limit=5
```

For Apptainer, replace only the provider config path:

```text
responses_api_agents/hermes_bench/configs/sandbox_apptainer.yaml
```

An inline provider also works:

```yaml
sandbox_provider:
  apptainer: {}
```

If the policy model is not reachable from inside the sandbox at Gym's resolved
URL, set `NEMO_GYM_SANDBOX_MODEL_BASE_URL` to its sandbox-reachable URL.

`agent_runtime_source: auto` builds and uploads the pinned Hermes-Agent runtime
once. Set it to `baked` when the image already contains Gym and Hermes-Agent, or
to a local archive or URL containing a portable runtime rooted at `bin/python`.
The default auto builder targets Linux x86_64 and must run on a matching Linux
host. Set `NGHB_PYTHON_ARCH` on a matching Linux host for another supported
target. On macOS, use a prebuilt archive, URL, or baked runtime.

## Image contract

The configured image must provide:

- a POSIX shell, `bash`, `tar`, and Python 3
- `git` for the six GitHub code review setup scripts
- dependencies required by task setup scripts
- PyYAML for the source skills runner
- `requests` for judged tasks
- browser and rendering dependencies for architecture and p5.js tasks
- any tool gateway credentials and services required by enabled toolsets

The agent and grader may use different images through
`grader_sandbox_spec.image`. For p5.js, configure `P5_RENDER_CMD` in
`grader_env` when nested Docker is unavailable.

## Judge credentials

The default config forwards only these runtime environment variables:

- `NOUS_API_KEY`
- `OPENROUTER_API_KEY`

The original grader prefers Nous when both are present. `NOUS_API_BASE_URL`,
`OPENROUTER_API_BASE_URL`, `P5_RENDER_CMD`, and other original grader variables
can be added through `grader_env`. Do not put keys in the dataset or image.

## Test

```bash
uv run --frozen --extra dev pytest -q responses_api_agents/hermes_bench/tests
```
