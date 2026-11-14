# TacHarness Loop

A benchmark-independent diagnostic loop with native UniVTAC ACT, DP and π0.5 adapters. Codex CLI uses `gpt-6-astra` with `medium` reasoning. Harness evolution is deliberately excluded.

All new files live in this directory. Source repositories, environments and shared model weights are read-only. No SHA/checksum mechanism is used. Source provenance is recorded in `vendor/MANIFEST.json`. The user separately authorized temporary simulator IPC objects in `/dev/shm`.

## Run

Use the existing environment; do not install dependencies into it:

```bash
cd /root/autodl-tmp/gujie/tacharness-loop
export PYTHONDONTWRITEBYTECODE=1
/root/autodl-tmp/gujie/UniVTAC/.venv/bin/python th.py init
/root/autodl-tmp/gujie/UniVTAC/.venv/bin/python -m unittest discover -s tests -v
/root/autodl-tmp/gujie/UniVTAC/.venv/bin/python -m adapters.smoke --policy act --action finetune_more --steps 50 --evaluate
```

After all adapter acceptance checks pass:

```bash
/root/autodl-tmp/gujie/UniVTAC/.venv/bin/python -m orchestrator.experiment all --workers 2
```

`all` starts baseline test concurrently with the 18 initial harness diagnoses, freezes the token budget, then runs the three target groups. Individual commands are `calibrate`, `baseline-test`, and `loop --group univtac_hole_act_rgbt`. **Do not start overlapping experiment supervisors.** This is real training/evaluation, not a dry run.

## Diagnostics and accounting

The raw condition exposes `raw.list/read`; harness additionally exposes `th.*`. Neither condition receives a playbook or precomputed evidence injection. First-round inputs are saved verbatim and reused. Requirement validation checks citations and structured facts independently, including stop decisions. `value_json` encodes the raw cited value; JSON fields use RFC6901 pointers and text uses `line:N`.

Each round allows one initial call plus one error retry or correction. Each group allows six CLI invocations and 150 recorded model responses. Response IDs, not tool calls, determine model-request count. CLI usage is observable at response completion: an in-flight response that crosses a request/token cap is fully charged and rejected, then the process is terminated; this is not a pre-request billing reservation. Forty MCP tool executions are allowed per call; the next call receives a final-answer warning and further tool calls terminate the session. Shell execution events terminate it as an isolation violation.

The token budget is `ceil(3 * P95 * 1.5)`, using linear P95 over the 18 initial harness calls. Input, cached input and output tokens are reported separately. Cached input is a subset of input. First-round target harness calls count toward each group's formal budget. Raw-condition calls belong to the separately reported diagnostic ablation. Failed first outputs remain in the diagnostic comparison; only the formal loop may use one correction.

## Prompt versions

Newly created inputs use [repair-v2](harness/prompts/repair-v2.txt). It requires examining feedback failures, training curves, representative trajectories and available imagery before choosing an evidence-linked experiment. It distinguishes a testable hypothesis from a proven cause; using the best validation checkpoint is not sufficient to stop. A stop must explain why each permitted action is unavailable, exhausted or unsupported, unless feedback is fully successful or the budget prevents useful action. These are model instructions, not a programmatic proof of causal correctness. The existing schema/fact/action verifier remains the execution gate.

Existing contexts without a `prompt_version` remain `diagnostic-v1`; their exact prompts and sealed gates are preserved. CLI calls read the saved input prompt, so a later template edit cannot silently change a retry or a paired raw/harness input. Later rounds inherit the first round's version. Formal calibration records the version and the loop rejects a version mismatch with the frozen budget.

The isolated development command is:

```bash
PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/gujie/UniVTAC/.venv/bin/python -m orchestrator.prompt_preview
```

It diagnoses DP and π0.5 once each with `gpt-6-astra / medium`, reuses completed preview receipts on rerun, and writes only under `runs/prompt-revisions/repair-v2/`. It does not train, evaluate, reopen gates, or count previews as formal first-round results. Its 150-response / 1,200,000-token ceiling is an engineering limit, not a new calibrated formal group budget. A formal experiment using v2 requires its own 18 initial harness calls for calibration and identical paired raw/harness inputs; the existing v1 experiment is not silently converted.

## Training and evaluation

Weights load strictly and optimizers reset. Validation IDs, normalization, architecture and inference remain fixed. UniVTAC DP has no EMA. Updates are optimizer steps, including DP. Additional data are converted into `runs/data/` using the original preprocessing.

`save_at_steps` is available only on the adapter's trusted control interface. Agent schema rejects it. Control schedules are ACT/π0.5 `[320, 680, 1000]`, DP `[1400, 2975, 4375]`, without resetting training between saves.

Feedback uses seeds 20001–20030. Test uses 20031–20060 and is entirely outside the MCP allowlist. A confirmed reset error before policy control can be retried once using the same seed. Policy failures are never retried. Baseline test starts early; gate selects solely on feedback, ties favor earlier rounds. After gate sealing the selected test is automatic. Selecting baseline reuses its registered test, including an in-flight run.

## Recovery and status

Completed artifacts and seed results are reused. Each training, diagnosis and evaluation process has explicit started/result records. Incomplete training or an ambiguous started evaluation fails closed for reconciliation instead of blindly repeating policy trials. Successful rounds, gate files and test registry entries are retained. Never manually delete failed attempts to make a run appear successful.

`python th.py audit` performs the original-directory modification-time check. Missing `/root/autodl-tmp/gujie/lab` is recorded; it is not created. Changes to protected directories stop execution without automatic repair.

Local Codex authentication copies are private runtime material: `runtime/` and `runs/` must not be published. Open-source the code and sanitized aggregate reports, not credentials or raw Codex state.

See `STATUS.json` for actual completed checks and remaining work; the existence of code is not evidence that an experiment passed.

## Other benchmark adapters

Implement `adapters.base.BenchmarkAdapter` on the benchmark's server, then register
`{"libero": "adapters.libero:LIBERO"}` in `configs/adapters.json` there. Run `th.py init --benchmark libero`, followed by `orchestrator.experiment ... --benchmark libero`. The adapter owns raw file discovery, episode normalization, validation curves, facts, demonstration checks, training, and native evaluation. It must expose only feedback sources to `raw_files`, preserve native evaluation rules, and declare a stable `protocol` identifier. This checkout does not claim a working LIBERO backend.

UniVTAC reset retry provenance is in `configs/evaluation-protocols.json`. Only the three verified target groups may launch evaluation; other groups can still run diagnostics. Verify their baseline retry rules before enabling additional evaluations.

For paired diagnostics and a live receipt-based report:

```bash
/root/autodl-tmp/gujie/UniVTAC/.venv/bin/python -m orchestrator.ablation --workers 2
/root/autodl-tmp/gujie/UniVTAC/.venv/bin/python th.py status
```

`orchestrator.acceptance_campaign` resumes the nine native action checks, the registered baseline tests, and target gates. It reuses the six completed 50-update training checks; it never treats a failed policy episode as a retry opportunity. The current campaign is already running, so inspect status before launching another supervisor.

Diagnostic metrics count all 18 first calls per condition. Citation-weighted evidence validity is reported alongside mean per-call validity (missing/empty outputs score zero), missing-output counts, and accepted requirement counts. The primary factual-error count covers success count, termination distribution, and best validation step; all additional fact-field errors are reported separately. These measures do not establish causal root-cause accuracy.

To regenerate the comparison without invoking the model:

```bash
/root/autodl-tmp/gujie/UniVTAC/.venv/bin/python -m orchestrator.ablation --collect-only
```

The native campaign supervisor identity is recorded in `runs/acceptance/supervisor.json`; native worker identities and commands are recorded per attempt. See [ARCHITECTURE.md](ARCHITECTURE.md) for the external benchmark contract and [reports/diagnostic-comparison.md](reports/diagnostic-comparison.md) for the completed first-diagnosis comparison.

## Verified implementation and active completion

[ACCEPTANCE.md](ACCEPTANCE.md) records the completed native action checks and actual model gate decisions. One pre-task Isaac startup failure was retained and separately recovered during technical acceptance. The recovery helper cannot replace reset or policy trials and is not called by the formal loop.

`orchestrator.finalize_campaign` is the coordinated completion process: it seals the target gates, waits for the already registered baseline tests, preserves the interrupted initial supervisor state, and publishes final reports. Do not start an additional experiment while these registered tests are still running.

## Demonstration replay and repair (v3)

New inputs default to `demo-repair-v3`. `raw.list(kind="demos")` lists source seeds and split membership; `raw.read` supports bounded HDF5 numeric slices, RGB/tactile frames, literal log search and tail reads. Raw and harness conditions share this access. Only the repair-stage second CLI call exposes `sim.try_repair` and `sim.verify_repair`.

An initial `finetune_data` diagnosis may request up to three eligible demonstrations. The trusted supervisor copies them into independent files and replays each original. The second CLI slot can then submit at most two local eight-dimensional joint-target patches and independently verify one successful candidate per demo. All starts are charged; there are at most four simulations per demo and twelve per round. CLI corrections and repair share the second slot. Simulator calls count toward the existing forty-tool cap; no model polling or extra CLI invocation is used while waiting.

Patches stay inside the declared zero-based half-open action window. Offset patches use a sine taper; replacement patches provide one target per frame. Command jumps may not exceed the larger of twice the original maximum per-axis jump and 0.05 rad (arm) / 0.005 m (gripper); the worker separately checks native joint limits and uses `force=False`. Full replay length and native success rules are fixed before each trial. Both candidate and independent verification must complete and succeed; if the original already succeeds, both must reduce reference tracking RMSE. An initialization failure is not classified as a demonstration defect.

Every candidate records fresh initial/after-action observations, four RGB/tactile streams, commands and outcomes. Training labels retain the baseline next-observed-qpos convention. Sealed manifests under `runs/datasets/` reference untouched originals read-only and retain independent physical copies for changed demonstrations; no mutable symlink/hardlink aliases are used. All three policy loaders resolve the explicit manifest and record their actual sources in the training receipt. Validation IDs and normalization stay fixed.

Use a separate experiment namespace, never replace old inputs or gates:

```bash
export TH_EXPERIMENT_ID=demo-repair-v3
export PYTHONDONTWRITEBYTECODE=1
/root/autodl-tmp/gujie/UniVTAC/.venv/bin/python th.py init
/root/autodl-tmp/gujie/UniVTAC/.venv/bin/python -m orchestrator.experiment calibrate --workers 2
/root/autodl-tmp/gujie/UniVTAC/.venv/bin/python -m orchestrator.ablation --workers 2
/root/autodl-tmp/gujie/UniVTAC/.venv/bin/python -m orchestrator.experiment loop --group univtac_hole_dp_rgbt
```

The eighteen initial calibration calls remain GPU-free; a version's raw/harness first inputs are identical. Replay GPU time is recorded separately from training updates and policy feedback trials. Existing baseline tests remain reusable. `orchestrator.repair_acceptance` is explicitly synthetic: it injects a local offset into an owned copy, checks model-directed repair, and never counts it as a real dataset defect or measured policy improvement. Synthetic data manifests cannot be used by formal groups.

LIBERO can implement `DemonstrationRepairAdapter` on its own server. No LIBERO simulator or remote environment is accessed here.

Simulation is dispatched through a per-session trusted Unix-socket broker. It accepts only typed `try`/`verify` operations for registered demo IDs; it cannot execute arbitrary commands. This keeps CUDA workers outside the model's inherited sandbox restrictions while preserving per-worker write confinement, model shell prohibition, and the shared simulation ledger. An early development attempt failed before actions with CUDA error 304 when launched directly below the model process; that failed candidate remains charged and recorded.

The completed v3 development evidence is in [reports/demo-repair-v3.json](reports/demo-repair-v3.json): 51 unit tests, real GPT-6 medium diagnosis/repair, an independently verified synthetic local repair, 20 native updates for each policy loader, and the model-selected DP 500-update action. Failed development attempts remain included. These are implementation checks, not a measured benchmark success-rate improvement.
