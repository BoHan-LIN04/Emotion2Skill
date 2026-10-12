# Emotion2Skill

**LLM-Internal Emotion Signals for Adaptive Skill Selection and Evolution**

Emotion2Skill extracts 27-dimensional internal emotion features from Qwen3, maps them to a learned confidence-gated routing summary, and uses trajectory transitions to rewrite skill SOPs. The released implementation follows the supplied AAAI 2027 manuscript protocol.

## Reported manuscript results

These values are reported in the supplied manuscript. They are not measurements produced by the software repair or by the CPU regression tests.

| Backbone | ALFWorld Avg. (%) | WebShop Score | WebShop Succ. (%) |
|---|---:|---:|---:|
| Qwen3-8B | 53.1 ± 1.6 | 64.7 | 33.3 ± 0.6 |
| Qwen3-14B | 57.0 ± 1.6 | 66.8 | 37.0 ± 0.6 |

The manuscript additionally reports 69.2% MATH accuracy on 500 problems and 71.8% MBPP pass@1 on 374 tasks. Original task manifests, checkpoints, corpus, baseline implementations and per-run traces have not been supplied with this release; their status is recorded in [artifacts.json](artifacts.json). Newly generated assets do not establish identity with the original runs.

## Installation

Use Python 3.10 or newer; CPU regressions were checked with Python 3.11.

```bash
git clone https://github.com/BoHan-LIN04/Emotion2Skill.git
cd Emotion2Skill
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,analysis,data]'
pytest tests/ -q
```

[requirements-tested.txt](requirements-tested.txt) records the direct dependency versions tested on macOS arm64. It is a CPU verification environment, not a recovered original H800 environment. Transformers must be at least 4.51 for Qwen3; the implementation currently targets its 4.x API.

For WebShop, follow the [official setup](https://github.com/princeton-nlp/WebShop#-setup), including product/goal data, the spaCy model and search index; record the exact upstream revision and product-pool settings. For ALFWorld, install its official package and download the data with `alfworld-download`. Set `datasets.alfworld.data_path` in the YAML to the downloaded root. The wrappers use the official APIs and stop if a real environment cannot initialize.

MBPP executes generated code in a separate process with a timeout and resource limits. These limits are not an operating-system sandbox. Run formal MBPP evaluation inside a dedicated container or VM with no credentials or network access.

## Prepare fixed datasets

Fetch and convert the official GoEmotions snapshot used in the preparation check:

```bash
python scripts/prepare_goemotions.py --download \
  --revision 5b381db4139c3cbf25cc9a698c5fe9754215fae2
```

The preparation script removes the neutral label, excludes neutral-only examples, preserves original example IDs, checks split overlap, and records source revision and SHA256. The snapshot produces 3,834 validation and 3,821 test examples. Layer selection reads validation only; fixed-layer test evaluation is separate.

Benchmark manifests must identify the actual train/dev/test tasks and dataset revision. The preparation tool preserves supplied splits and never invents a replacement test subset:

```bash
python scripts/prepare_benchmarks.py --benchmark webshop \
  --source data/webshop_splits.json --dataset-revision SOURCE_COMMIT \
  --output manifests/webshop.json
python scripts/prepare_benchmarks.py --benchmark alfworld \
  --source data/alfworld --dataset-revision DATASET_REVISION \
  --output manifests/alfworld.json
```

WebShop source JSON contains `train`, `dev`, `test` lists of canonical goal indices and optional `environment` settings for the exact product pool. ALFWorld preparation reads solvable official `json_2.1.1` games under train/valid_seen/valid_unseen and retains relative gamefile paths and hashes. MATH/MBPP source JSON contains explicit split lists of original records: a canonical `task_id`/`id`, MATH `problem` and `answer`/`solution`, or MBPP `text` and `test_list`. Use the same preparation command with `--benchmark math` or `mbpp` and an appropriate source/output. Preserve the original paper task lists when reproducing its numbers.

## Build extractor and shared encoder

```bash
python scripts/build_extractor.py --config configs/default.yaml
python train_encoder.py --config configs/default.yaml --env mixed
```

The story builder varies explicit scenarios under greedy decoding, applies sentence-count and target-top-3 classification filters, rejects embedding cosine duplicates above 0.95, and requires exactly 100 passing stories per concept. Failed builds retain resumable evidence and cannot become a valid extractor. The 14B extractor uses the completed canonical Qwen3-8B corpus.

Extraction uses prefix exclusion at token 50, top-10 neutral PCA components from 100 neutral stories, and validation-only layer selection. Online features come from the final token of the real non-thinking chat generation prefix. Model and dataset provenance are retained with the extractor.

Warm-up runs a text-only ReAct skill policy while still recording real emotion vectors. The default uses 250 training episodes per benchmark, filters low-signal steps, and trains one encoder per backbone for reuse across all four benchmarks. Training uses AdamW, lr 0.001, weight decay 0.0001, batch 64, up to 50 epochs and patience 5 on an episode-disjoint validation subset. The objective combines contrastive loss, prototype classification (weight 0.5), and success-label confidence BCE (weight 1). Cluster-to-template semantic mappings, the canonical 12-template bank and source hashes are stored in the checkpoint.

## Run main experiments

```bash
bash scripts/run_webshop_8b.sh 0,1
bash scripts/run_alfworld_8b.sh 0,1
bash scripts/run_qwen3_14b.sh webshop 0,1
python evaluate.py --config configs/default.yaml --env math --no-evolution
python evaluate.py --config configs/default.yaml --env mbpp --no-evolution
```

Each evolution round uses 50 distinct WebShop and 50 distinct ALFWorld training tasks, sharing the encoder and keeping separate libraries. The default is three rounds with underperformance threshold 0.4. Warm-up/evolution/test IDs and checkpoint source metadata are checked for overlap; final test libraries are frozen. The rewriter uses real transition windows, LLM diagnostics, and a 200-word SOP limit. Long diagnostics are processed in budgeted batches while preserving the complete source windows and generated summaries.

YAML configuration is actually loaded; CLI options override it. The 14B YAML inherits the base protocol and sets matching model, extractor and encoder identities. Three runs use the same task subset in distinct recorded orders. Run directories retain effective config, task order, seed, backend, checkpoint hashes, complete observations/actions/emotion states, grading results, and initial/evolved SOP snapshots. Reported uncertainty uses sample standard deviation.

## Ablations and analysis

```bash
bash scripts/run_ablations.sh 0,1
bash scripts/run_tau_sweep.sh 0,1
bash scripts/run_round_analysis.sh 0,1
```

The ablations separately replace contrastive directions with plain class means, bypass the encoder with top-3 labels, disable evolution, or use outcome-only rewriting. Generic PCA-27 is fitted on training activations; Shuffled Emotion breaks training feature/context pairings. Control encoders are retrained and validated against their extractor mode and shuffle metadata. Threshold selection uses dev tasks, while individually specified test settings support the manuscript sensitivity table.

```bash
python scripts/analyze_results.py --logs results/RUN/.../episodes.jsonl \
  --out results/analysis.json --figures results/figures
python scripts/export_activations.py --config configs/default.yaml \
  --split test --layer 24 --out results/probe_layer24.npz
python scripts/analyze_results.py --probe_files results/probe_layer24.npz \
  --out results/probe_analysis.json --figures results/figures
```

Analysis reads actual traces/activations. It supports co-activation, confidence histograms, case trajectories, held-out round curves, t-SNE/Fisher probes, MATH/MBPP comparisons, task-cluster bootstrap intervals and paired permutation tests. Baseline/candidate comparisons require matching task and run IDs. Test is the default analysis partition; dev analysis requires explicit `--split dev` and is marked ineligible for paper test metrics. Synthetic runs cannot enter either analysis or paper aggregation.

Use `--annotation_events` to export 200 sampled events for model/human annotation, then pass supplied boolean judgments and human annotations via `--judgments` and `--human_annotations`. Coherence rates and Cohen's kappa are computed from these files, never inserted as assumed values. The local manuscript figure generator only produces schematics without data; empirical plots require measured inputs.

## Baselines

ZeroShot, ReAct, Reflexion and ExpeL have explicit manuscript-protocol runners. ZeroShot sees no skill SOPs; ReAct preserves Thought/Action history; Reflexion performs actual failed-task reflection with a recorded retry budget. ExpeL requires 50 disjoint recorded training experiences and real offline insight extraction before top-5 BM25 retrieval:

```bash
python -m emotion2skill.baselines build-memory --logs TRAIN_EPISODES.jsonl \
  --out checkpoints/expel_memory.json --model Qwen/Qwen3-8B --device cuda
python evaluate.py --method ExpeL --baseline_artifact checkpoints/expel_memory.json
```

These are manuscript-protocol adaptations, not a claim to reproduce every detail of upstream algorithms. `TextSimilarity` is a separately named lexical skill baseline. It is not labeled MASA.

MASA, EvoSkill (the manuscript cites AlignEvoSkill) and Trace2Skill require a genuine upstream implementation and source-pinned artifact manifest:

```bash
python evaluate.py --method MASA --external_adapter package.module:factory \
  --baseline_artifact artifacts/masa_manifest.json
```

The manifest must contain `method`, the cited primary `source_url`, exact `revision`, `benchmark`, `source_split: train`, `source_task_ids`, `protocol_config`, and nonempty `implementation_files` / `artifacts` lists of `{path, sha256}`. The actual factory source must be hash-covered and return the shared runner interface and matching method/revision metadata. Missing integrations fail explicitly. Their original implementations and experimental assets are not bundled here.

Primary sources: [ReAct](https://arxiv.org/abs/2210.03629), [Reflexion](https://github.com/noahshinn/reflexion), [ExpeL](https://github.com/LeapLabTHU/ExpeL), [MASA](https://arxiv.org/abs/2605.30723), [AlignEvoSkill v2](https://arxiv.org/abs/2506.23149v2), [Trace2Skill](https://arxiv.org/abs/2603.25158). AlignEvoSkill manifests must cite v2: the same arXiv identifier's v1 describes V-SYNTHESIS. Official upstream code is available for [MASA](https://github.com/jianxiangyu/MASA_) and [Trace2Skill](https://github.com/Qwen-Applications/Trace2Skill); connecting these sources to this benchmark protocol still requires a validated adapter and original training artifacts.

## Tests and smoke runs

```bash
pytest tests/ -q
python evaluate.py --env webshop --mock --n_episodes 2 --n_runs 1 --no-evolution
```

The regression suite checks actual data flow, confidence updates, checkpoint compatibility, split isolation, official adapter interfaces, mathematical/code grading, evolution budgets, provenance and analysis. Explicit mock runs test plumbing and are labeled synthetic; their scores are not benchmark results. CI runs the CPU regression suite.

## License and citation

Code is distributed under the [MIT License](LICENSE). Project-generated Emotion Stories use [CC BY 4.0](data/LICENSE); third-party datasets retain their own licenses. [CITATION.cff](CITATION.cff) describes this software release. The supplied manuscript is under review; this repository does not assert an accepted AAAI publication.
