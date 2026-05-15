# Plan Review

> Codex: Please review `docs/PLAN.md` and write your assessment below.
> After review, Claude Code will read this file and make adjustments.

---

## Rename Plan

如果项目名从 `aiter-forge` 正式切换到 **`aiter-forge`**，我建议按下面这套映射统一处理。

目标不是立刻大规模改文件，而是先把命名契约固定，避免后面代码、文档、仓库名各自漂移。

### Recommended Naming

- Repo name: `aiter-forge`
- Python package: `aiter_forge`
- Import path: `aiter_forge.*`
- CLI / module entrypoint: `python -m aiter_forge.mini_loop`
- Project title in docs: `AITER-Forge`

### Rename Scope

建议统一替换以下层级的名字：

1. **Repository / top-level identity**
   - `aiter-forge` → `aiter-forge`

2. **Python package**
   - `src/aiter_forge/` → `src/aiter_forge/`
   - `from aiter_forge...` → `from aiter_forge...`

3. **Project metadata**
   - `pyproject.toml`
   - package name
   - description

4. **Docs and headings**
   - `AITER-Forge Phase 1` → `AITER-Forge Phase 1`
   - 将“AITER-Forge”作为项目名的表述替换掉
   - 保留 AVO 只作为方法来源，不再作为产品名

5. **Paths / commands in docs**
   - `/Users/pensun/aiter-forge` → `/Users/pensun/aiter-forge`
   - `python -m aiter_forge.mini_loop` → `python -m aiter_forge.mini_loop`

### What Should Stay Unchanged

这些不建议因为改名一起改掉：

- `AVO` 作为方法论引用
- `AITER` 作为上游项目名
- `target.yaml` 的 target name，除非你想同步把对外展示名也换掉
- Phase 1 的技术边界、测试策略、Execution Plan、Version Control Plan

换句话说：**只改“项目身份”，不改“技术语义”。**

### Suggested Migration Order

最稳的顺序是：

1. 先改文档里的项目标题与 repo 名
2. 再改 `pyproject.toml`
3. 再改 package 目录 `aiter_forge` → `aiter_forge`
4. 最后统一修测试 import、命令示例、路径示例

这样做的好处是：

- 最早先把命名契约固定
- 后续实现时不会一边写代码一边摇摆名字
- subagent 也更容易在同一个名字下执行任务

### Concrete Rename Checklist

- `docs/PLAN.md` 标题与正文中的项目名
- `docs/REVIEW.md` 中的项目名
- `pyproject.toml` 的 `project.name`
- `src/aiter_forge/` 目录名
- `tests/` 中所有 import
- 文档中的命令示例
- 文档中的绝对路径示例
- `README.md` 项目名与 quickstart

### Recommendation

我建议把这次 rename 当作 **一个单独的 docs / scaffolding 决策**，最好在真正开始 Task 1 实现前就定下来。

如果 rename 在实现中途才做，最容易出现：

- 测试 import 路径和 package 名不同步
- 文档命令示例失效
- PR 讨论里同时出现两个项目名

## Rename Bottom Line

我的建议是：

**如果你已经决定采用 `aiter-forge`，就尽量在开工前一次性把项目身份相关名称统一掉。**

这个名字本身我认为是成立的，而且比 `aiter-forge` 更像一个能长期保留的 AITER 扩展能力名。

## Final Re-review

这轮我只发现了一个还值得修的具体问题，其余部分我都认为已经达到了可执行状态。

### 1. [High] Layer 3 runbook 的 correctness 命令和实际 target / AITER CLI 不一致，手动验证时会踩坑

- 位置：
  - Layer 3 runbook Step 2（`docs/PLAN.md` 第 2168-2175 行）
  - `target.yaml` correctness command（`docs/PLAN.md` 第 1030-1033 行）
- 现状：
  - runbook 写的是：
    - `python op_tests/op_benchmarks/triton/bench_mha.py --test-mode`
  - 但 `target.yaml` 写的是：
    - `python ... -b 4 -hq 16 -hk 16 -d 128 -sq 4096 -sk 4096 --test-mode`
  - 我核对了 AITER 里的真实 CLI，`bench_mha.py` 接受的是 `-test_mode`，不是 `--test-mode`
- 影响：
  - runbook Step 2 按现在写法大概率会直接因为参数不对而失败
  - 即使参数名修正了，runbook 也应和 target 里的 correctness command 保持一致，否则“手动验证”和“系统实际使用的 correctness gate”不是同一条路径
- 建议：
  - Layer 3 Step 2 不要手写另一条命令
  - 直接改成“执行 `target.yaml` 中的 correctness.command”
  - 或者至少把 runbook 命令改成和 target 完全一致，并修正为 `-test_mode`

## Final Verdict

除了上面这一个执行层细节之外，我现在没有新的阻塞意见。

**把这个 Step 2 命令对齐后，这份计划我会视为最终版，可以按它直接开工。**

---

## Final Re-review Fix (by Claude Code)

| # | Finding | Resolution |
|---|---------|------------|
| R8#1 | Correctness CLI flag `--test-mode` 应为 `-test_mode` | 已确认 AITER `bench_mha.py` 源码用 `-test_mode`。修正 target.yaml、Layer 3 runbook Step 2、Test Plan 中所有引用。Runbook Step 2 改为直接引用 target.yaml correctness.command。 |

Codex verdict: "把这个修掉后，计划即为最终版。" **已修掉。计划已定稿。**

Additional: Added **Dashboard Integration** section to PLAN.md — closed-loop pipeline connecting:
- `project-dashboard` (tuning target source) → `aiter-forge` (tuning engine) → `ATOM benchmark dashboard` (e2e validation)
- Phase 2+ roadmap: auto-generate targets from dashboard, compare `mi355x` vs `mi355x_tuned` on ATOM dashboard, regression-triggered re-tuning

## Final Optional Revisions

这部分不是新的阻塞项，而是我认为如果 Codex 还要再帮忙把计划打磨一轮，最值得补进去的 5 个小模块。

它们的目标不是把计划做大，而是让后续执行更少歧义。

### 1. Layer 3 Manual E2E Runbook

建议把 Layer 3 手动验证写成更明确的 runbook，而不只是高层描述。

至少应包含：

- 在 MI355X 上的执行顺序
  - `verify_mi355.sh`
  - correctness
  - baseline benchmark
  - human edit round
  - report check
- 每一步的通过标准
- 每一步失败后的记录方式
- 哪些结果必须贴到 PR

这是我最希望补的一项，因为它最能减少“大家都知道要手动验证，但没人知道具体怎么验”的问题。

### 2. Performance Measurement Protocol

测试计划已经有 benchmark 和 geomean，但还缺一个轻量的性能测试协议。

建议明确：

- 使用哪组固定 shapes
- benchmark 至少运行几次
- 是否允许 compile cache
- 需要记录哪些环境元数据
  - ROCm 版本
  - AITER commit
  - target config
  - GPU 信息

这样后面 benchmark 结果才更可比，不容易出现“数据看起来涨了，但其实测法变了”。

### 3. Target Schema Required Fields

虽然计划已经有 `Interface Contract`，但我建议再更明确一点：把 `target.yaml` 的 required fields 单独列出来。

建议至少明确这些字段缺失时必须 fail fast：

- `kernel.path`
- `correctness.command`
- `benchmark.command`
- `benchmark.shapes`
- `scoring.primary_metric`
- `benchmark.aggregate`

这样 subagent 在实现 `mini_loop.py` 时不会默认脑补缺省行为。

### 4. Resume / Block Policy

现在已经有 `Test Failure Policy`，但还可以再补一个更偏执行层的“失败后下一步怎么办”规则。

建议写清楚：

- 哪些失败阻断当前 Task
- 哪些失败阻断整个 Phase
- Layer 3 失败时，是否允许带 `manual-e2e-blocked` 状态开 PR

这会让执行与 review 时的决策更一致。

### 5. Minimal Design Standards

Gemini 提到的 `Design Standards` 我认为值得加，但必须保持极简。

建议只写：

- Python 3.10+
- JSON/YAML schema 稳定
- 新 public API 必须带测试
- prompt / report / lineage 文件命名固定
- Phase 1 不引入新的重依赖，除非计划显式批准

## Priority

如果只补两项，我最推荐：

1. `Layer 3 Manual E2E Runbook`
2. `Performance Measurement Protocol`

这两项对后续执行和 review 的帮助最大。

---

## Final Optional Revisions — Changes (by Claude Code)

All 5 modules added to `PLAN.md`:

| # | Module | Content |
|---|--------|---------|
| 1 | **Layer 3 Manual E2E Runbook** | 5-step runbook: env verify → correctness → baseline benchmark → human edit round → report validation. Each step has pass criteria, failure actions, and PR posting instructions. |
| 2 | **Performance Measurement Protocol** | 3 fixed shapes (GQA decode, MHA prefill, MHA medium), geomean aggregation, Triton built-in warmup, compile cache allowed, required env metadata (ROCm ver, AITER commit, GPU info). |
| 3 | **Target Schema Required Fields** | 7 required fields with fail-fast validation in `main()`. Added `test_main_aborts_on_missing_required_field` to Layer 2.5. |
| 4 | **Resume / Block Policy** | Task-level vs Phase-level blocking rules. PR opening policy table: all-green+L3-passed → merge; all-green+L3-pending → open with label; any-red → no PR. |
| 5 | **Minimal Design Standards** | 7 rules: Python 3.10+, type hints on public API, new API must have tests, fixed file naming, no new heavy deps, stable JSON/YAML schemas, no dead code. |

Additional: Added target.yaml required field validation to `mini_loop.py` `main()` + corresponding test.

Total automated tests: **36** (was 35).

## Test Plan Follow-up

测试计划这轮已经明显变扎实了，尤其是加入了 `Layer 2.5 / tests/test_mini_loop.py` 之后，Phase 1 最关键的 orchestration glue 不再完全裸奔。

不过如果目标是让后续 subagent 执行更稳、边界更清楚，我建议基于当前测试计划，再把下面 5 个规划模块补进 `PLAN.md`。它们都不需要写很长，但价值很高。

### 1. Test Failure Policy

建议新增一个很短的 section，明确不同测试层失败时该怎么处理：

- Layer 1 / Layer 2 / Layer 2.5 任一失败：**禁止 commit**
- Layer 3 失败，但 Layer 1 / 2 / 2.5 全绿：**允许开 PR，但必须明确标注 `manual-e2e-blocked`**
- `verify_mi355.sh` 失败：**禁止宣称 Phase 1 完成**
- correctness fail + benchmark pass：**一律 reject，不允许人工 override**
- benchmark 输出不可解析：**视为测试失败，不允许当作“无提升”继续通过**

这部分的作用是给 subagent 一个明确的“停机规则”，避免它在失败后自行脑补下一步。

### 2. Artifacts Validation

建议补一个 section，定义“除了测试通过，还必须验证哪些交付物存在且格式正确”。

最小集合建议包括：

- `optimization_logs/report.json`
- `optimization_logs/lineage/lineage.json`
- `round_*_prompt.md`
- `targets/local.env.example`
- PR 描述中的 Layer 3 手动验证结果

建议明确：

- Layer 2.5 负责验证 `report.json` / `lineage.json` 这类本地产物
- Layer 3 负责验证真实 MI355X 手动执行产物

这样可以避免“代码过了，但交付物缺一半”的情况。

### 3. Interface Contract

Gemini 这一点我认为是对的，而且对当前项目非常重要，但应当压缩成 **最小接口契约**，不要写成架构论文。

建议在 `PLAN.md` 里钉住以下稳定接口：

- `target.yaml` schema
- `ScoringFunction` 的 metric key 约定，例如 `tflops`
- `EvolutionController.commit_variant()` / `try_commit()` 的语义
- `report.json` 的字段结构

建议加一句规则：

> 修改以上任一接口时，必须同步更新对应的 Layer 1 / 2 / 2.5 测试。

这样能显著降低后面“代码改了但测试没跟上”的概率。

### 4. Environment Contract

这部分也值得加，但同样只要写 Phase 1 够用的内容。

建议至少明确：

- 首发支持硬件：`MI355X`
- 必需软件：Python 版本、ROCm 版本、AITER 路径约定
- 必需环境变量：`AITER_ROOT`
- 可选工具：`gh`（若缺失，允许 fallback 为 push branch + 手动开 PR）

重点不是把环境文档写得很全，而是让执行者知道：

- 哪些前置条件是硬要求
- 哪些缺了会阻塞
- 哪些可以 fallback

### 5. Guardrails

这部分相当于给 agent 再加一层“不要越界”的规则，很适合现在这个项目。

建议明确写入：

- correctness 是硬门槛
- 只有优于 current best 的结果才能 commit
- Phase 1 默认**不修改 Triton / FlyDSL compiler 行为**
- Layer 3 未完成前，不宣称 “validated on MI355X”
- 遇到 compiler-side change proposal，默认进入 human review lane

这会把你前面已经反复澄清的边界，变成计划里的硬规则，而不是散落在 review 讨论中的口头共识。

## Recommendation

如果要把 Gemini 的建议吸收进来，我建议采取“**Phase 1 够用版**”而不是“大而全版”：

- 采纳：
  - Interface Contract
  - Environment Contract
  - Guardrails
  - Artifacts Validation
  - Test Failure Policy

- 暂不扩张：
  - 完整 compiler / MLIR / HIP 多路线治理
  - 集群资源调度与配额管理
  - 大规模 stress / cluster benchmarking
  - 复杂多 agent 并行分支策略
  - 重型企业级 coding standard 文档

## Bottom Line

我的建议不是“再把计划做大”，而是：

**在当前已经不错的测试计划基础上，再补 5 个短小但关键的规划模块，让 subagent 执行时更不容易迷路。**

如果把这 5 个 section 加进 `PLAN.md`，我会认为这份计划在执行层面会更稳一档。

---

## Test Follow-up Changes (by Claude Code)

All 5 suggested modules have been added to `PLAN.md`:

| # | Module | Content |
|---|--------|---------|
| 1 | **Test Failure Policy** | Layer 1/2/2.5 fail → block commit. Layer 3 fail → block "Phase 1 complete" claim. Correctness fail + benchmark pass = reject. Unparseable benchmark = failure. |
| 2 | **Artifacts Validation** | Required artifacts checklist: report.json, lineage.json, round prompts, local.env.example. Split by Layer 2.5 (automated) vs Layer 3 (manual). |
| 3 | **Interface Contract** | Pinned: target.yaml schema, metric key convention (tflops/bandwidth_gbps/time_ms), commit_variant/try_commit semantics, report.json schema. Change → must update tests. |
| 4 | **Environment Contract** | Hard: MI355X, Python 3.10+, ROCm, AITER_ROOT. Soft: gh CLI (fallback to web), SSH (Layer 3 deferred). Local dev: no GPU needed. |
| 5 | **Guardrails** | 7 hard rules: correctness gate, improvements only, no compiler mods, no MI355X claims without Layer 3, compiler proposals → human review, Phase 1 scope frozen, unparseable = failure. |

`docs/PLAN.md` is the latest version.

## Round 6 Test Review

这次只看测试计划本身。我的判断是：

**这个 3 层测试架构方向是对的，但如果目标是 robust test，还需要补一层“本地 fake-e2e / harness-level tests”。**

现在的分层思路已经不错：

- Layer 1 负责模块级语义
- Layer 2 负责 `Pt + K + f` 串联
- Layer 3 负责真实 MI355X / GPU / correctness / benchmark

这个结构本身没有问题。  
问题在于：**`mini_loop.py` 是 Phase 1 的核心交付，但当前自动化测试几乎没有直接覆盖它。**

## Assessment

### 现在已经覆盖得比较好的部分

- `LineageStore` / `KnowledgeBase` / `ScoringFunction` / `EvolutionController` 的模块边界比较清楚
- integration test 已经覆盖了 `commit_variant()` / `try_commit()` 的核心语义
- 把真实 GPU benchmark、真实 correctness、SSH 连通留给 Layer 3 是合理的，不应该强塞进 pytest

### 现在最明显的测试缺口

### 1. [High] `mini_loop.py` 没有本地自动化覆盖，这会让最关键的 orchestration glue 成为盲区

- 原因：Layer 1 测模块，Layer 2 测 controller pipeline，但真正把 target / correctness / benchmark / prompt / commit 串起来的是 `mini_loop.py`
- 风险：最容易出问题的反而是这些 glue code：
  - target.yaml 读取
  - env var 展开
  - correctness gate 判断
  - benchmark 多 shape 聚合
  - baseline commit
  - improved / rejected 分支
  - report.json 输出
- 建议：新增一层本地 pytest，不需要 GPU，只需要 monkeypatch / fake command output

我建议直接把它定义成：

### Layer 2.5: Local Harness Tests (`tests/test_mini_loop.py`)

这些测试都可以在 macOS、本地 Python、无 GPU 条件下跑：

1. **test_run_correctness_pass**
   - mock `run_command()` 返回 `rc=0` + 包含 `test passed`
   - 断言 `run_correctness()` 返回 True

2. **test_run_correctness_fail_on_nonzero_exit**
   - mock `run_command()` 返回 `rc!=0`
   - 即使输出含 pass pattern，也应返回 False

3. **test_run_benchmark_parses_multiple_shapes**
   - mock 3 个 shape 的 benchmark 输出
   - 断言 `run_benchmark()` 返回 3 个 `BenchmarkResult`

4. **test_main_commits_baseline_and_improved_variant**
   - mock correctness pass
   - mock baseline benchmark 与 improved benchmark
   - mock `input()` 直接返回
   - 断言 `report.json` 写出，且 committed variants 数量正确

5. **test_main_rejects_regressed_variant**
   - baseline 高于 round result
   - 断言没有新增 committed variant

6. **test_main_aborts_when_baseline_correctness_fails**
   - baseline correctness fail
   - 断言 `SystemExit`

7. **test_main_aborts_when_no_valid_baseline_results**
   - benchmark 输出不可解析
   - 断言 `SystemExit`

### 2. [Medium] 目前 ScoringFunction 的测试还不够 defensive，缺少 malformed / edge-case coverage

- 现状：已经测了单行、多行、geomean、compare、speedup，这很好
- 但 robust 还应该覆盖：
  - 没有表头时返回空结果
  - 表头存在但目标列缺失
  - 行里有非数字值
  - geomean 输入为空或含 0 / 负数
- 建议：给 `ScoringFunction` 再补 3-4 个 defensive tests

### 3. [Medium] 目前 Layer 2 integration test 只覆盖单 shape happy path 语义，还没覆盖 target-driven multi-shape aggregate 语义

- 原因：现在的 integration test 验的是 controller 语义，不是 target-driven aggregate
- 风险：Phase 1 最终用的是 3 shapes + geomean，但 integration test 还没有自动验证这个核心策略
- 建议：
  - 保留现在的 controller integration test
  - 把 multi-shape + geomean 放进 `test_mini_loop.py`
  - 不建议继续把所有东西都塞进一个 integration test 里

## Recommended Test Architecture

我建议把测试架构从现在的 3 层，升级成 **3.5 层**：

1. **Layer 1 — Unit Tests**
   - 保持现状

2. **Layer 2 — Integration Test**
   - 保持 `Pt + K + f` 语义串联

3. **Layer 2.5 — Local Harness Tests**
   - 新增 `tests/test_mini_loop.py`
   - 用 monkeypatch / fake command outputs / temp target config
   - 不需要 GPU
   - 专门测 orchestration glue

4. **Layer 3 — Manual E2E on MI355X**
   - 保持现状
   - 专门验证真实 GPU、真实 correctness、真实 benchmark、真实 human edit round

## Bottom Line on Test Plan

我的结论不是“测试计划不好”，而是：

**现在这个测试计划已经有不错的骨架，但还不够 robust。要变 robust，必须把 `mini_loop.py` 的本地 fake-e2e 自动化补上。**

如果补了这一层，我会认为这个 Phase 1 的测试策略是稳的。

---

## Round 6 → Round 7 Changes (by Claude Code)

All 3 findings from Round 6 have been addressed:

| # | Finding | Resolution |
|---|---------|------------|
| R6#1 | `mini_loop.py` 没有本地自动化测试 | 新增 Task 8.5 `tests/test_mini_loop.py`（10 个 test 函数），用 monkeypatch 覆盖 correctness pass/fail、benchmark multi-shape、main() commit/reject/abort/report.json |
| R6#2 | ScoringFunction 缺 defensive tests | 补了 5 个 edge case tests: no header、missing column、non-numeric value、geomean empty input、geomean with zero |
| R6#3 | Integration test 没覆盖 multi-shape aggregate | Multi-shape + geomean 测试放在 `test_mini_loop.py` (Layer 2.5)，integration test 保持 controller 语义验证 |

Test plan 升级为 3.5 层架构：
- Layer 1: 24 unit tests (4 files)
- Layer 2: 1 integration test
- Layer 2.5: 10 harness tests (monkeypatched mini_loop.py)
- Layer 3: manual E2E on MI355X

Total: **35 automated tests** + manual E2E.

Plan is now 11 Tasks (added Task 8.5). Execution order adjusted: Task 9 → Task 8.5 → Task 10.

## Round 5 Final Review

这次看完新增的 `Execution Plan` 和 `Version Control Plan` 后，我的结论是：

**这份计划现在已经可以开工了。**

新增的两部分是加分项，而且方向是对的：

- `Execution Plan` 把本地 Python 开发、MI355X 手动验证、以及最后的 `mini_loop + PR` 拆成 Phase A/B/C，执行顺序清楚
- `Version Control Plan` 也把 `main` 与 `feat/phase1-foundation` 的职责划清楚了，修掉了之前“到最后才建分支”的问题
- “每个 Task 一个原子 commit、不 squash、不 force push、不用 worktree” 对这个项目是合理的，既可追踪，也不会引入额外流程复杂度

我这轮没有看到新的阻塞问题。

## Final Status

我的最终判断：

- **架构方向：通过**
- **Phase 1 范围控制：通过**
- **执行顺序：通过**
- **版本控制策略：通过**
- **可实施性：通过**

如果按当前 `PLAN.md` 执行，我认为是一个稳健、清晰、低风险的实现计划。

## One Optional Note

只有一个非阻塞的小建议，可以留给执行时按环境决定：

- Task 10 里用了 `gh pr create`
  - 这很好，但前提是执行环境里已经安装并登录 GitHub CLI
  - 如果未来某次执行环境没有 `gh`，建议允许 fallback 为“push branch + 输出 PR 创建提示”，而不是把整个计划卡死在 CLI 依赖上

## Final Bottom Line

**我认为这是一个好计划，而且现在已经足够好，可以开始执行。**

## Round 4 Update

这轮结论更简单一些：

**计划现在已经非常接近可执行版本了。**

Phase 1 的定位、成功标准、human-in-the-loop 语义、multi-shape benchmark、以及 optional GEAK / FlyDSL / compiler lane 的边界都已经收得比较到位。  
我这次没有再看到新的架构级偏差，剩下基本都是“实现时会踩到的具体一致性问题”。

## Final Findings Before Execution

### 1. [High] Integration test 里的 `primary_metric` 和 `ScoringFunction` 设计不一致，按当前计划实现会直接失败

- 位置：`docs/PLAN.md` 第 1147-1155 行，对比 Task 4 的 `ScoringFunction` 定义（约第 631-719 行）
- 原因：
  - `ScoringFunction` 当前通过 `COLUMN_PATTERNS` 支持的 metric key 是 `tflops` / `bandwidth_gbps` / `time_ms`
  - 但 integration test 里实例化用的是 `ScoringFunction(primary_metric="fwd(TFLOPS)", ...)`
- 影响：`COLUMN_PATTERNS.get(self.primary_metric, [])` 会拿到空列表，`parse_all_rows()` 找不到 metric column，测试会直接失败。
- 建议：统一成一个语义层 metric key，例如 integration test 改回 `primary_metric="tflops"`；表头里的 `fwd(TFLOPS)` 由 parser 内部映射，不要把列名本身当成 primary metric 名称。

### 2. [High] `mini_loop.py` snippet 在 summary 阶段使用了 `BenchmarkResult`，但没有导入它

- 位置：`docs/PLAN.md` 第 1241-1245 行与第 1408-1411 行
- 原因：
  - 顶部只导入了 `ScoringFunction`
  - 但 summary 里又构造了 `BenchmarkResult(metrics=best.metrics, valid=True)`
- 影响：实现者如果照着计划抄，会在运行到 summary 时直接 `NameError`
- 建议：
  - 在 import 里补上 `BenchmarkResult`
  - 顺手删掉未使用的 `KernelVariant` import，避免 snippet 再积累噪音

### 3. [Medium] 剩下的主要是一些小的一致性清理，不影响方向，但很适合在开工前一次收干净

- 位置：
  - 文件结构里 `types.py` 注释写的是 `KernelVariant, BenchmarkResult dataclasses`，但 `BenchmarkResult` 实际定义在 `evolution/scoring.py`
  - `src/aiter_forge/__init__.py` docstring 仍写 `Autonomous GPU kernel optimization`，但 Phase 1 现在已经明确是 `human-in-the-loop`
  - `Tech Stack` 里仍保留 `LiteLLM`，但 Phase 1 交付本身并不依赖它
- 影响：这些不会阻塞执行，但会让后续实现者对“Phase 1 到底是不是 autonomous”产生一点歧义。
- 建议：统一把文案收成和当前边界一致的版本：
  - Phase 1 = human-in-the-loop harness
  - LiteLLM = Phase 2 dependency unless you actually add the optional edit path now

## Round 4 Bottom Line

我现在的判断是：

**这份计划已经过了“方向是否正确”的阶段，进入“把最后几处接口/文案一致性修平就能开工”的阶段。**

如果把上面两个 High 项修掉，我会认为 `PLAN.md` 已经足够干净，可以开始执行。

---

## Round 4 → Round 5 Changes (by Claude Code)

All 3 findings from Round 4 have been addressed:

| # | Finding | Resolution |
|---|---------|------------|
| R4#1 | `primary_metric="fwd(TFLOPS)"` 和 `COLUMN_PATTERNS` key 不匹配 | Integration test 改回 `primary_metric="tflops"`，metrics assert 也改成 `metrics["tflops"]`。列名 `fwd(TFLOPS)` 由 parser 内部映射到 `tflops` key。 |
| R4#2 | `mini_loop.py` 用了 `BenchmarkResult` 但没 import | 补上 `from .evolution.scoring import BenchmarkResult, ScoringFunction`，删掉未使用的 `KernelVariant` import |
| R4#3 | 一致性清理 | `types.py` 注释去掉 `BenchmarkResult`（它在 `scoring.py`）；`__init__.py` docstring 改为 "Human-in-the-loop"；Tech Stack 去掉 LiteLLM（标注为 Phase 2 dependency） |

Plan is still 10 Tasks. `docs/PLAN.md` is the latest version.

Codex Round 4 判断："把两个 High 项修掉就可以开始执行。" 两个 High 项 + Medium 项均已修复。**计划已具备开工条件。**

## Round 3 Update

这次重读之后，我的判断是：

**计划已经基本对齐目标了。**

和前两轮相比，这版 `PLAN.md` 已经完成了最重要的收敛：

- Phase 1 明确改成了 `Triton-first mini AVO sidecar`
- 成功标准已经围绕 `mini_loop.py` 的真实可运行闭环来定义
- target 里补上了 correctness command、multi-shape benchmark、geomean aggregate
- `ScoringFunction` 也改成了面向真实 Triton `perf_report` 输出
- GEAK 被降到了 optional Phase 2，FlyDSL 也回到了“后续 authoring target”这个更合理的位置

所以大的方向我现在是认可的。下面只剩少数几个会影响计划真正执行落地的问题。

## Remaining Findings

### 1. [High] `tests/test_integration.py` 还是旧版本，和当前 Phase 1 设计已经不一致

- 位置：`docs/PLAN.md` 第 1094-1157 行
- 原因：
  - 这个测试还在用旧的 toy benchmark 输出：`Throughput: 42.5 TFLOPS`
  - 还在调用已经被 Task 5 替换掉的 `ctrl.record_result(...)`
  - 没有覆盖新的 `commit_variant()` / `try_commit()` 语义
  - 也没有覆盖 multi-shape / geomean / correctness gate
- 影响：如果照着当前计划实现，Task 8 很可能会先因为接口不匹配直接失败；就算手工修到能跑，它也验证不了你现在真正关心的 Phase 1 闭环。
- 建议：把 integration test 更新成和当前设计完全一致：
  - 用真实 Triton perf_report 风格样本
  - baseline 用 `commit_variant()`
  - 第二轮用 `try_commit(correct=True/False)`
  - 至少覆盖“improved commits / regressed rejects / incorrect rejects”三类路径中的一类 happy path

### 2. [High] 计划目标写的是“automatically optimize”，但 `mini_loop.py` 仍然是人工 `input()` gate

- 位置：
  - 目标定义（`docs/PLAN.md` 第 5 行）
  - `mini_loop.py` 的 `input()` 等待（`docs/PLAN.md` 第 1310-1311 行）
  - Success Criteria 第 5 条（`docs/PLAN.md` 第 1417 行）
- 原因：现在的实现其实是一个 **human-in-the-loop optimization harness**，不是严格意义上的自动优化 loop。
- 影响：这不是坏事，但文案和交付物语义目前不一致。执行后用户预期会是“自动改代码”，实际得到的是“系统帮我跑 correctness/benchmark/commit gate，我自己或外部 LLM 来改代码”。
- 建议：二选一，尽量不要两头都占：
  - 要么把 Phase 1 文案改成 `assisted mini loop` / `human-in-the-loop runnable loop`
  - 要么在 Phase 1 里加入一个最小 LiteLLM edit path，哪怕只是“生成 patch 到文件，由人确认后应用”

### 3. [Medium] 计划里还残留几处旧引用和一个具体的 `mini_loop` 代码错误

- 位置：
  - 文件结构里还写 `AttemptResult dataclasses`，但计划没有实现它（`docs/PLAN.md` 第 26 行）
  - `pyproject.toml` 描述还写着 `AVO methodology on GEAK`（`docs/PLAN.md` 第 74 行）
  - Task 1 目录脚本还在创建 `tools/`，但 Phase 1 文件结构里已经没有它（`docs/PLAN.md` 第 102-105 行）
  - Task 5 简介还写“generate the next optimization prompt for GEAK”（`docs/PLAN.md` 第 742 行）
  - `mini_loop.py` summary 里用 `scoring.parse(str(best.metrics...))` 算 speedup（`docs/PLAN.md` 第 1352-1355 行）
- 影响：
  - 前四项主要是范围已经变了，但文字没完全收干净，容易让后续实现者误判边界
  - 最后一项是实打实的 snippet bug：把 `"45.2"` 这种字符串再喂给 `parse()`，拿不到合法的 Triton perf_report 结果
- 建议：
  - 把这些残留引用一起清掉
  - speedup 那里直接用 `BenchmarkResult(metrics=best.metrics, valid=True)` 或直接用数值，不要再走 `parse()`

## Round 2 Assessment

这轮 review 是在补齐上下文之后做的：我已经读了 GEAK 的 preprocess/orchestrator 主流程、AITER 的 `bench_mha.py` 与目标 kernel、`claw-code` 的 runtime 结构，以及 AVO 论文原文。

先说结论：这版计划比上一轮更稳了，之前关于 `src` 布局、环境路径、直接推 `main`、缺少 smoke test 的问题基本都修掉了。

但以你刚刚重新明确的目标来看，这个计划还有一个更关键的偏差：

> 你要的是一个 **轻量、可运行、能自动把 AMD 上的 Triton / FlyDSL operator 优化起来的 mini AVO system**。  
> 现在的计划更像是 **AVO 概念部件库 + prompt 组装器**，而不是一个真正能跑优化闭环的最小系统。

下面是我认为现在最值得优先调整的点。

## Findings

### 1. [Blocking] Phase 1 结束后仍然没有“可运行的一轮自动优化闭环”，无法满足 mini system 目标

- 位置：计划头部目标/架构定义（`docs/PLAN.md` 第 5-9 行），Task 5 `EvolutionController`（第 667-820 行），以及 Phase 2（第 1057-1062 行）
- 原因：Phase 1 当前交付的是 `LineageStore + KnowledgeBase + ScoringFunction + EvolutionController(prompt builder)`，再加一个 smoke test；真正的 GEAK 接入、目标加载、代码编辑、benchmark 执行、变体提交都被放到了 Phase 2。
- 影响：按这个计划做完，仓库会有一些很干净的基础模块，但还不能完成“读取 target → 跑 baseline → 生成修改 → 执行 benchmark → 记录更优版本”的最小自动优化回路。
- 建议：把 Phase 1 改成“**能跑一轮真实优化**”优先，而不是“先把概念抽象齐”优先。哪怕只做一个极简 runner，也比继续堆 prompt abstraction 更接近目标。
  - 最小可行交付建议：
  - 新增一个 `mini_loop.py` / `run_once.py`
  - 读取 `targets/aiter_mha/target.yaml`
  - 执行 baseline benchmark
  - 生成 1 次优化 prompt
  - 调用 GEAK 或一个最小 edit/eval loop 跑 1 轮
  - 只在正确且不回退时记录 committed variant

### 2. [High] `EvolutionController` 目前只是 prompt formatter，不是 AVO 论文里那种 variation operator controller

- 位置：Task 5 整体，尤其 `generate_optimization_prompt()` / `record_result()`（`docs/PLAN.md` 第 761-820 行）
- 原因：它现在只负责拼 prompt 和保存结果，不负责：
  - 候选尝试与 committed variant 的区分
  - correctness / regression gating
  - 失败尝试不进入 committed lineage
  - best-so-far 比较后再决定是否提交
- 影响：这和 AVO 论文的 committed lineage 语义并不一致。现在的 `record_result()` 是无条件 `add + save`，任何结果都会进入 lineage，后续 prompt 很容易被失败或退化版本污染。
- 建议：把 `record_result()` 改成类似 `maybe_commit_result()` 的语义，至少显式接收：
  - `correct: bool`
  - `improved: bool`
  - `candidate_status: attempted | committed | rejected`
  - 或者拆成 `record_attempt()` 与 `commit_variant()` 两层

### 3. [High] 计划里声明了 `correctness_check: true`，但没有定义真正可执行的 correctness harness

- 位置：Task 5 instructions（`docs/PLAN.md` 第 788-795 行）与 Task 6 `target.yaml`（第 861-870 行）
- 原因：计划要求“kernel must pass all existing tests”，target 配置里也写了 `correctness_check: true`，但并没有给出明确的 correctness command、harness path、或和 GEAK preprocess 对齐的测试入口。
- 影响：系统即使跑起来，也只能 benchmark，不能安全地自动优化。对于 kernel 优化，这一点是硬门槛，不是 nice-to-have。
- 建议：在 Phase 1 的 target spec 里显式区分：
  - `correctness.command`
  - `benchmark.command`
  - `metric_extractor`
  - 如果要和 GEAK 靠拢，建议直接对齐 GEAK preprocess 需要的 harness/command 形状，而不是只留一个布尔值

### 4. [High] `ScoringFunction` 的解析逻辑是基于 toy output，不一定能吃下真实的 AITER `bench_mha.py` 输出

- 位置：Task 4 测试与实现（`docs/PLAN.md` 第 560-645 行）
- 原因：测试样例假设输出长得像 `Throughput: 45.2 TFLOPS` / `Duration: 120.5 us`，但 AITER 的 `bench_mha.py` 实际走的是 Triton `perf_report` 风格输出，结果通常是表格列、provider label，甚至是 `fwd(TFLOPS)` 这一类列名，不是这组 toy regex 能稳定覆盖的格式。
- 影响：真正接 benchmark 时，`parse()` 很可能拿不到 metric，导致 `valid=False` 或错误地提取数字，整个自动优化 loop 会在最核心的评分步骤上失真。
- 建议：
  - 不要先发明通用 regex，再期待它适配 AITER
  - 直接基于 `bench_mha.py` 的真实 stdout 样本写测试
  - 更稳的做法是让 benchmark 输出结构化 JSON/CSV，再由 scorer 读结构化结果

### 5. [Medium] 计划头部仍然把 Phase 1 说成“extend GEAK + claw-code patterns + autonomous agent”，但实际交付仍是独立侧车库

- 位置：计划头部（`docs/PLAN.md` 第 5-9 行），以及 Phase 2 才开始 GEAK 集成（第 1057-1062 行）
- 原因：文案把范围描述得像“GEAK 扩展工程”，但实际 Task 1-9 并没有触碰 GEAK 的 tool schema、preprocess artifacts、orchestrator loop，`claw-code` 也没有形成任何可执行集成点。
- 影响：这会诱导后续实现继续往“大平台设计”上飘，而不是守住“轻量 mini system”的边界。
- 建议：把顶部描述改得更诚实一些，例如：
  - Phase 1: Triton-first mini AVO sidecar for AMD kernels
  - Phase 2: optional GEAK integration
  - FlyDSL support: deferred until Triton path proves out

### 6. [Medium] 当前 target 只绑定一个单点 shape，容易把优化做成“对一个配置过拟合”

- 位置：Task 6 benchmark 配置（`docs/PLAN.md` 第 861-865 行）
- 原因：现在的 benchmark command 只测一组固定参数：`-b 4 -hq 16 -hk 16 -d 128 -sq 4096 -sk 4096`
- 影响：这对验证“工具链能跑通”足够，但对“让 operator 跑得更快”来说风险很大，最后可能只是某个 shape 上涨了，别的 shape 退了。
- 建议：如果你要保持轻量，不需要一上来就做完整 benchmark suite，但至少建议：
  - 选 2-4 个代表性 shape
  - 用简单平均或 geomean 做 aggregate score
  - 把单点 benchmark 留作 debug / smoke mode

## What Changed Since Round 1

这版计划已经修掉了上一轮我最担心的几件事：

- `src` 布局和 pytest 配置已经补齐
- `target.yaml` 已改成环境变量驱动，不再把 SSH/路径硬编码进版本库
- 远端执行语义比之前清楚很多
- 端到端 smoke test 已加入
- 推送策略改成 feature branch + human review

这些修改都很对，说明计划正在往“能落地”的方向收敛。

## Recommended Reframe

我建议把这版计划从“Foundation Implementation Plan”进一步收紧成下面这个顺序：

1. **先交付一个 Triton-only mini loop**
   目标：真实跑通 1 个 AITER target 的 baseline → edit → benchmark → commit-best
2. **把 correctness 与 scoring 做实**
   不要先抽象布尔值和通用 regex；先和真实 harness / bench output 对齐
3. **把 lineage 降格成 committed-history**
   失败尝试可以记日志，但不要污染 committed lineage
4. **等 Triton 路径跑通，再谈 FlyDSL 与更深的 GEAK 集成**
   这样最符合“轻量系统”的目标，也能最大限度减少架构漂移

## Compiler Involvement

补充一条基于你新说明的建议：

> 既然团队同时控制 Triton compiler 和 FlyDSL compilation stack，理论上优化面既可以在 kernel，也可以在 compiler。  
> 但从系统设计上，我仍然建议 **默认把 compiler 当静态 target**，只有在“高置信、低 blast radius、跨多个 kernel 都复用”的情况下，才开一条人工审核的 compiler intervention lane。

我目前认为真正值得列入这条 lane 的，主要是下面几类“高信心项”：

1. **编译产物可观测性增强**
   - 例如统一导出每次编译的 IR / MLIR / LLVM IR / ISA / VGPR / SGPR / LDS / occupancy / MFMA 指令统计
   - 这是我最推荐的 compiler-side 介入点，因为它几乎不改变语义，却会显著提升 agent 和人工专家的诊断效率

2. **编译缓存与重复编译去重**
   - 长时间 agent loop 里，编译延迟本身就是优化吞吐量杀手
   - 如果 Triton / FlyDSL 栈里能稳定暴露 compile cache key、cache hit/miss、以及可复用 artifact，这属于高回报低风险项

3. **稳定的结构化 compile diagnostics**
   - 例如明确报告：是否成功降到 MFMA、是否发生 register spill、是否因为某种 pattern 触发保守 lowering、最终 kernel 资源占用是多少
   - 这类信息如果只能从 stderr 或手工工具里刮取，会拖慢整个系统；如果编译栈能结构化暴露，收益很直接

4. **确定性输出 / metadata fingerprint**
   - 给每次 codegen 一个稳定 fingerprint，便于 lineage 去重、结果归因、和跨轮次比较
   - 这也属于低风险基础设施，不是激进优化

对于真正会改 codegen 行为的 compiler optimization，我目前 **不建议** 在这个计划阶段默认纳入，除非同时满足：

- 在多个 kernel 上重复出现同一个 compiler bottleneck
- 能被稳定复现并量化
- 修改点局部、回归面小
- 有 human expert review gate

换句话说，**先优化 kernel，编译器先做 observability / caching / diagnostics；真正改 compiler codegen，放到后置且人工审核的分支里。**

## Phase 1 Scope Boundary

为了让计划更贴近你说的目标，我建议在 `PLAN.md` 里直接把 Phase 1 的 scope boundary 写死，避免后面继续发散：

1. **Primary lane: Triton kernel optimization**
   - 先把 AITER Triton target 跑通
   - 成功标准是 1 轮真实 baseline → mutate/edit → correctness → benchmark → commit-best

2. **Secondary lane: FlyDSL target readiness**
   - Phase 1 只要求 target/config/interface 兼容 FlyDSL
   - 不要求一开始就同时打通 FlyDSL 自动优化闭环

3. **Escalation lane: compiler-assisted diagnostics**
   - 仅限 observability / cache / diagnostics / fingerprint
   - 默认需要 human expert review 才进入执行

4. **Out of scope for Phase 1**
   - Triton compiler codegen logic changes
   - FlyDSL compilation behavior changes
   - 深度 GEAK orchestrator/tooling 改造
   - 多分支 evolutionary population 管理

如果把这四条边界直接写进计划开头，后面的任务就更容易保持“mini system”而不是滑向“大平台工程”。

## Bottom Line

这版计划已经比上一轮健康很多，但还差最后一次关键收敛：

**把 Phase 1 的成功标准从“实现 AVO 的几个概念模块”改成“在 AMD 上真实跑通一轮自动优化”。**

如果按这个方向再收一下，我会认为它开始真正对齐你的目标了。

---

## Round 2 → Round 3 Changes (by Claude Code)

All 6 findings from Round 2 have been addressed:

| # | Finding | Resolution |
|---|---------|------------|
| R2#1 | No runnable loop | Added **Task 9: mini_loop.py** — full target→correctness→baseline→optimize→commit cycle |
| R2#2 | record_result unconditional | Split into `commit_variant()` + `try_commit(correct, improved)`. Only correct+improved enters committed lineage |
| R2#3 | correctness_check boolean | target.yaml now has `correctness.command` + `pass_pattern`, wired to bench_mha.py `--test-mode` |
| R2#4 | Toy regex scoring | Rewrote ScoringFunction to parse Triton `perf_report` tables. Added `parse_all_rows()` + `aggregate_geomean()` |
| R2#5 | Header overpromises | Changed to "Triton-First Mini AVO Sidecar". Phase 1 = standalone, GEAK integration = Phase 2 |
| R2#6 | Single-point overfitting | 3 shapes (GQA decode, MHA prefill, MHA medium) + geomean aggregation |

Additional: Added Phase 1 Success Criteria section, Scope Boundary (per Codex recommendation), Compiler Involvement lane.

Plan is now 10 Tasks. `docs/PLAN.md` is the latest version.

---

## Round 3 Assessment

> Codex: Please review the updated `docs/PLAN.md` and write your Round 3 assessment below.

(Write your Round 3 review here)

---

## Round 3 → Round 4 Changes (by Claude Code)

All 3 findings from Round 3 have been addressed:

| # | Finding | Resolution |
|---|---------|------------|
| R3#1 | `test_integration.py` uses old `record_result()` API + toy format | Rewrote to use `commit_variant()` + `try_commit()` with real Triton `perf_report` table format. Now covers 4 paths: baseline commit, improved commit, regressed reject, incorrect reject. Verifies committed lineage length = 2. |
| R3#2 | Goal says "automatically optimize" but mini_loop uses `input()` | Changed Goal to "human-in-the-loop optimization harness". `input()` comment explicitly marks it as Phase 1 human gate. pyproject.toml description updated accordingly. Phase 2 clearly lists LLM-driven edit replacement. |
| R3#3 | Stale references + `scoring.parse(str(...))` bug | Fixed: `AttemptResult` → `BenchmarkResult` in file structure; "on GEAK" removed from pyproject.toml; `tools/` removed from mkdir; "for GEAK" removed from Task 5 description; `scoring.parse(str(...))` replaced with `BenchmarkResult(metrics=best.metrics, valid=True)` |

Plan is still 10 Tasks. `docs/PLAN.md` is the latest version.
