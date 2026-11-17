# Rework architecture and validation

## Canonical stages (one question at a time)

The canonical task graph is `mrdre_baseline.runtime.runner.task_plan`.

- `gen1` calls the official upstream LangChain Open Deep Research graph once.
- `cov1` uses the original per-criterion checklist evaluator.
- `contentfb` selects one unsatisfied checklist item using original simulation.
- `formatfb` selects three of 21 seeds and asks the original simulation LLM.
- `gen2_content`, `gen2_format`, `gen2_reflect` independently revise R1.
- `cov2_*`, `rubric_*`, `cite_*`, and `formatinc` use unchanged score prompts.
- Content INC and BRK are computed by the original Appendix B.4 equations.
- Final aggregation uses the dataset **sample mean** on completed IDs only.

Stages have distinct output files and resume by ID. The runner keeps one
*logical* question in flight, but it overlaps dependent-independent stages across
Agent and judge lanes on separate GPUs. On a one-GPU host both lanes share a
single executor to protect available VRAM.

## Reproducibility and benchmark comparability

Run manifests block mixing different data/model/ODR settings. For clean
comparisons, use the same model, parameters, evaluation judge, feedback seed,
research graph commit, context length, web-search source, and report budgets
across baseline and intervention. Runtime context is recorded by bootstrap,
not included in the benchmark identity: **do not compare a truncated 8192-token
local run to a 16384-token cloud run as though they were identical**.

`Open Deep Research` is downloaded unmodified, pinned by commit at resumption.
The CLI assumes a server reachable at 127.0.0.1:2024. Search uses a localhost
MCP service, one agent thread for each report, and new threads for revisions.

## Performance mechanisms and limits

1. Keep Ollama, LangGraph, MCP resident between questions.
2. Cache repeated LLM prompts and fetched URLs using the existing cache.
3. Do not repeat completed JSONL stage calls on resume.
4. Adapt judge concurrency to currently free VRAM; never exceed configured
   Ollama concurrency. Keep one GPU worker on 6 GB.
5. For two GPU instances, pipelined agent and evaluator work; if judge uses
   vLLM, its internal continuous batching/prefix caching may increase
   throughput but must be measured, and the model should be frozen.
6. Persist per-stage wall time and CPU/VRAM snapshots.
7. Retain original one-criterion, one-claim and one-URL judging to protect
   evaluation semantics; bulk changes are research interventions, not silent
   throughput tweaks.

## Known constraints

- The live Graph, Ollama and MCP integration require GPU/Internet and were not
  exercised by the offline unit tests.
- Existing Ollama servers are reused and cannot be reconfigured by bootstrap;
  restart them with desired environment settings if required.
- vLLM is optional and needs an independently installed CUDA-compatible vLLM.
  It uses a separate judge model/HF deployment; treat as a different benchmark
  configuration until equivalence is experimentally validated.
- A content-feedback branch cannot be generated if every checklist item is
  already ideal; the sample remains incomplete instead of fabricating feedback.
- `--budget-minutes` may interrupt the current stage, leaving a valid partial
  checkpoint; that question is not counted completed until resume finishes it.
