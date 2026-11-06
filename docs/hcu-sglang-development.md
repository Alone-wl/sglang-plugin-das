# hcu_sglang 开发迁移记录

> **本文用途**：持续记录本仓 `hcu_sglang` 每次代码迁移或开发完成的步骤、提交范围、实际验证与未完成条件。2026-10-28 起，本仓从仅维护方法推进到同时维护运行时代码；早期方法库限制不再禁止当前包落地。

## 1. 当前布局与接口

| 对象 | 当前定义 |
|---|---|
| 代码目录 / Python 包 | `hcu_sglang/` / `hcu_sglang` |
| distribution / 开发版本 | `hcu-sglang-plugin` / `0.3.2.dev7` |
| general plugin entry point | `hcu = hcu_sglang:register` |
| 显式服务入口 | `python3 -m hcu_sglang` 或 `hcu-sglang-serve`，参数传给官方 launcher |
| 来源插件版本 | `afc6205dd1a0f1979cfab9bf1f3272d2db14f46c`，原包 `sglang_dcu_plugin` |
| 官方接口基线 | `baa9befa381511cd4d56f83980951e5a45fffc6d`，SGLang 0.5.22 |
| 来源清单 | [逐文件迁入快照](hcu-sglang-import-manifest.json)，用于本次迁入审计，不是未来文件完整性锁 |

复制来源仅为输出仓的运行时包。没有复制外部 docs、tools、tests 或其他生成脚本。包内 7 份不参与运行的来源说明 JSON 未放入运行时目录；源路径、哈希与排除理由保存在迁入清单。保留原生资源、继承的注释/docstring、版权与 vendor LICENSE。

原来的 `usercustomize.py` 位于来源包之外，未作为全局 Python 启动文件迁入。新增包内显式 launcher 先安装同一惰性 AITER shim，再转发官方服务；普通 Python 子进程及冷导入不自动加载 Torch/AITER。

## 2. 迁移与提交台账

| 日期 / 步骤 | 完成内容 | 提交记录 | 验证与边界 |
|---|---|---|---|
| 2026-10-28 / M01 运行时代码迁入 | 525 个来源文件；180 个 Python 文件仅改命名空间；239 个源 Python 文件 AST 对照；286 个原生资源/许可证按原字节保留 | [`4d82a16`](https://github.com/Sherlolo/hcu-sglang-plugin/commit/4d82a1663a54c5d4075c5215acc2d91dcfc66750) | 本次静态迁入对照通过；不把旧 GPU/模型结果记为新包验证 |
| 2026-10-28 / M02 安装与启动入口 | 本仓打包定义、`dcu` 入口、显式 launcher、命名空间可选严格加载工具、阶段门禁及当前使用文档 | 同一代码提交 `4d82a16` | 65 项仓库测试、源码/wheel 各 363/363 项 Hook 和 3 个具名接口通过；完整服务未执行 |
| 2026-10-28 / M03 HCU 发现入口 | entry point 改为 `hcu = hcu_sglang:register`；版本 `0.3.2.dev7`；同步当前安装与开发说明 | [`443eb83`](https://github.com/Sherlolo/hcu-sglang-plugin/commit/443eb83eb56d91ec5a86c1c53989419fb0077667) | 源码/wheel 各 363/363 项 Hook、3 个具名接口；旧 `dcu` 名称隔离通过 |

代码提交：`4d82a1663a54c5d4075c5215acc2d91dcfc66750`，标题 `feat: import HCU runtime as hcu_sglang with isolated packaging`，共 557 个变更文件。范围包括 526 个运行时包文件、根打包定义、迁入清单、当前使用文档与 Skill、结构门禁和运行验证工具/测试；未迁入输出仓的 docs、scripts 或 tests。其后提交 `c8ba45a` 仅回填该代码 SHA 并重建导览。

当前入口在 M03 改为 `hcu`；M01/M02 中的 `dcu` 与 dev6 验证结果保留历史含义。

当前代码来源是已存在实现的迁入，不是宣称整个 DAS fork 已完成转换。模型能力、Native、CP/EP/PD 的历史范围仍以对应来源记录为线索，本命名空间必须建立自己的产物与运行证据。

## 3. 当前安装与验证方式

先准备匹配的官方 SGLang、Torch/HCU、DTK 和 vendor 依赖；本次 wheel 不代替该环境。按 [配置管理](configuration.md#1-当前工具的路径配置) 提供真实本机路径，安装时使用隔离环境或 task target，避免同名旧入口重复加载：

```bash
# 从本仓根目录执行；仅安装当前插件，不补齐 vendor/HCU 依赖。
python3 -m pip install -e . --no-deps --no-build-isolation
export SGLANG_PLUGINS=hcu
# 追加本次模型、设备、TP、backend、量化与 Graph 等完整配方。
python3 -m hcu_sglang --help
```

严格加载工具默认保留旧 Python API/命令兼容；验证当前包时显式选择命名空间：

```bash
: "${OFFICIAL:?请设置固定官方源码路径}"
: "${ARTIFACTS:?请设置本次证据目录}"
REPO="$(pwd -P)"
python3 tools/validate_plugin_runtime.py \
  --official-python "$OFFICIAL/python" --plugin-root "$REPO" \
  --plugin-package hcu_sglang \
  --expected-targets docs/hcu-sglang-import-manifest.json --output "$ARTIFACTS/hcu-runtime.json"
```

记录实际解释器、entry point、模块文件来源、目标集合与具名 seam、加载软错误，以及是否误导入旧包。当前 entry point 名为 `hcu`，由固定官方 loader 对 `SGLANG_PLUGINS` 中的名称精确匹配；不是 Python 包名后缀自动扫描，也不保留 `dcu` 别名。升级后须重新安装本仓或新 wheel 以刷新 metadata，再设置 `SGLANG_PLUGINS=hcu`。安装其他插件时同样核对实际白名单和模块来源，不能凭入口存在或目标数量认定通过。源码、隔离 wheel 和完整服务分别验收。

## 4. 实际验证记录

### M01/M02：首次迁入（dev6 / dcu）

验证日期：2026-10-28。实际环境为 Python 3.10.12 与既有 HCU/DTK 依赖容器；官方源码固定在上述 `baa9befa`。安装、构建和仓库测试使用源码所属用户；真实 Hook 加载使用容器原有的可访问设备的执行身份。没有修改设备权限或共享 site-packages。

| 检查 | 实际结果 | ignored 原始证据（相对仓库根） |
|---|---|---|
| 在线来源核验 | 525 个来源/目标 SHA256 一致；official、fork、来源 plugin 的 HEAD 与干净状态保持 | `artifacts/v3-20261028/provenance.json` |
| 结构及仓库测试 | L0：261 个 Python、8 个 Skill；65 项 unittest 通过 | `artifacts/v3-20261028/l0.log`、`tests.log` |
| 源码隔离安装与冷导入 | distribution `hcu-sglang-plugin==0.3.2.dev6`，唯一 `dcu = hcu_sglang:register`；冷导入未载入 Torch/AITER/旧包 | `artifacts/v3-20261028/source-install.log`、`source-cold-import.log` |
| 源码真实加载 | 363 项目标集合完全匹配且全部应用；3 个具名 resolution 接口；软错误与未应用列表为空；未导入旧包 | `artifacts/v3-20261028/source-hooks.json`、`source-hooks.log` |
| wheel 构建与完整性 | wheel 内 526 个包文件路径/字节与源码一致；未携带外部 docs/tools、旧包或全局 Python 启动文件 | `artifacts/v3-20261028/runtime-validation.json`、`build-wheel.log` |
| 隔离 wheel 安装与冷导入 | 从独立 target、仓库外 cwd 导入；唯一新入口；冷导入无 Torch/AITER/旧包 | `artifacts/v3-20261028/wheel-install.log`、`wheel-cold-import.log` |
| wheel 真实加载 | 未将仓库根加入 PYTHONPATH；实际导入来自 wheel target；363/363 项和 3 个具名接口通过，无加载软错误或旧包导入 | `artifacts/v3-20261028/wheel-hooks.json`、`wheel-hooks.log` |
| 显式 CLI 参数入口 | 源码和 wheel 的 `python3 -m hcu_sglang --help` 均 exit 0；只证明参数入口可用 | `artifacts/v3-20261028/source-cli.log`、`wheel-cli.log` |
| 文档与 Skill | 8 个 Skill 元信息、导览快照及链接/锚点校验通过；迁移计划保持原三方快照并链接当前记录 | `artifacts/v3-20261028/local-validation.json`、`guide-qa.json` |

本次 wheel 为 `hcu_sglang_plugin-0.3.2.dev6-py3-none-any.whl`，SHA256：`dc992c7a420148204cee1e0a9274ca590ea555189fb9bc58d74b606e1f720e8a`。它包含 Python 与原生源码资源，实际运行仍依赖兼容 vendor 库及工具链；没有发布公共 PyPI 版本。

复现构建和隔离 wheel 安装（`ARTIFACTS` 为本次独立可写目录）：

```bash
python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir "$ARTIFACTS/wheels" .
python3 -m pip install --no-deps --no-build-isolation --target "$ARTIFACTS/wheel-site" \
  "$ARTIFACTS/wheels/hcu_sglang_plugin-0.3.2.dev6-py3-none-any.whl"
```

wheel 加载进程在仓库外执行，只将固定官方 `python/` 与本次 `wheel-site/` 加入 PYTHONPATH，并将严格验证工具的 `--plugin-root` 指向 `wheel-site/`。目标合同仍使用本仓迁入清单。

**环境排查记录**：首次用无容器 passwd 记录的普通 UID 验证时，Torch 默认缓存落到不可写目录，AITER 随后尝试写不可用的用户 JIT 目录。配置本次任务的 `XDG_CACHE_HOME`、`TORCHINDUCTOR_CACHE_DIR`、`TRITON_CACHE_DIR`、`AITER_JIT_DIR` 后缓存问题消除；首次 AITER 原生依赖构建完成。随后普通 UID 报设备不可用，而容器原有执行身份可见 8 张设备，真实加载因此使用后者。失败尝试日志保存在 `artifacts/v3-20261028/failed-attempts/`，不能将编译耗时或 loader 捕获异常后的退出视为加载通过。

**未执行范围**：新命名空间的模型生成、GPU 数值/Graph/多卡服务及 embedded 产物未在本轮复验。真实 Hook 加载使用了设备环境，但不能代替这些验收；旧命名空间的历史结果仅作为后续回归线索。

### M03：HCU 插件发现入口（dev7 / hcu）

代码提交：`443eb83eb56d91ec5a86c1c53989419fb0077667`（`fix: discover HCU plugin through the hcu entry point`，13 个文件）。

将根打包元数据入口改为 `hcu = hcu_sglang:register`，开发版本更新为 `0.3.2.dev7`。运行时实现及 363 项目标合同不变；同步当前安装说明、开发指南和 Skill，保留旧产物的历史入口。实际执行环境沿用固定官方 `baa9befa` 与 Python 3.10.12 / HCU 容器。源码与 wheel 均从新的隔离 target 安装；构建/安装使用源码所属用户，设备加载使用容器原有执行身份。

| 检查 | 本次结果 | ignored 证据（目录 `artifacts/hcu-entrypoint-20261028/`） |
|---|---|---|
| 安装 metadata 与冷导入 | dev7 仅提供 `hcu → hcu_sglang:register`；源码及 wheel 冷导入均无 Torch/AITER/旧包 | `source-cold-import.log`、`wheel-cold-import.log` |
| `SGLANG_PLUGINS=hcu` 源码加载 | 363/363 目标全部应用、3 个具名接口，无加载软错误或旧包导入 | `source-hooks.json`、`source-hooks.log` |
| `SGLANG_PLUGINS=hcu` wheel 加载 | 从仓库外 cwd、隔离 wheel target 加载，363/363 和 3 个具名接口通过 | `wheel-hooks.json`、`wheel-hooks.log` |
| 旧名称隔离 | 新进程设置 `SGLANG_PLUGINS=dcu`，0 项 Hook，新包未导入 | `old-name-isolation.log` |
| wheel 完整性 | 526 个包文件路径及字节与源码一致，无旧包或全局启动文件 | `runtime-validation.json` |
| 仓库门禁 | L0 与 65 项 unittest 通过；两个更新的 Skill、链接/锚点和导览快照校验通过 | `l0.log`、`tests.log`、`guide-qa.json` |

本次 wheel：`hcu_sglang_plugin-0.3.2.dev7-py3-none-any.whl`，SHA256：`5d61be21697b03c9c5e21cf26c7ec74a161357f7af15e93b0bf2c2e911740ecd`。新版本更新安装元数据后采用 `export SGLANG_PLUGINS=hcu`；dev6 的 `dcu` 结果不作为 dev7 的实际加载证据。本轮未执行模型服务、数值或 Graph 复验。

构建首次使用设备执行身份时被 Git 所有权校验拒绝，已改用源码所属用户完成构建；保留失败日志，未修改全局 Git 信任配置。

### M04：SlimQuant INT4 混合权重接入（开发中）

本轮基线为 official `baa9befa381511cd4d56f83980951e5a45fffc6d`（v0.5.22）、adapter `b8af034fee92a38eca959502e50bcac47bb2a9f3`、plugin `405b8dc9dd3bbde6a85368089875c879b27668a7`。开发分支为 `codex/glm53-slimquant-int4`，尚未提交。实际设备环境使用 DTK 26041 / PyTorch 2.11.0 / HIP 6.3.26113 的独立容器；模型只读挂载，代码、缓存和日志单独保存。

职责：新增 SlimQuant checkpoint 格式、混合精度分派和 INT4 执行路径归 plugin；既有上游算子在 HCU 上的实现沿 adapter。组合运行必须在 Python 启动前设置 `HCU_SLIMQUANT=0`，关闭 adapter 的历史 SlimQuant 注册，并设置 `SGLANG_PLUGINS=hcu`。注册冲突直接报错，避免不同提供者静默抢占量化名称。

本轮改动：

- 补齐 CLI 的 `slimquant_w4a8` / `slimquant_w4a8_marlin` 选项；此前只暴露 `w4a8_int8`。将配置从 vendor 执行模块拆出，配置解析不初始化 LightOp/设备；保留原模块的配置类导出。读取 `ignore`、`moe_int8_layers`，保留官方名称映射和融合投影精度一致性检查。BF16 忽略层、INT8 dense、INT4 routed experts 与指定 INT8 MoE 层分别选取对应方法；NextN 的专家前缀随官方 mapper 变换，不能直接把 draft 层编号 0 当作源 checkpoint 层编号。
- 使用 `DefaultModelLoader._get_all_weights` 的窄 AROUND Hook 转换本配置拥有的权重迭代器。`w4_dir=true` 的 `qweight` 必须为 U8 `[N,K/2]`，`scales` 必须为浮点 `[N,1]`；转换为现有 INT4 内核的高位优先有符号字节和 `scale/16`。原始模型文件不改写，旧 `weight/weight_scale` 格式不变，其他量化方法直接调用原加载路径。
- Triton INT4 MoE 传播 `swiglu_limit`，在 SiLU 前分别裁剪 gate 上界与 up 双侧边界。排序使用官方 SGLang 入口，避免 LightOp 排序 padding 未初始化；调优配置按 vendor 的 E/M 参数顺序调用，K/N 分块满足实际形状，两个 GEMM 的 M 分块不同时分别排序。
- GLM 构造作用域内，共用的 DeepSeek 方法通过 AROUND 调用当前有效上游实现，保留 adapter 包装；实例标记使后续 forward/加载使用同一路径。修复旧 fork constructor 的 `tp_rank/tp_size` 参数与当前 `parallel_group` 接口不匹配，以及 KPool/DCP 状态缺失。其他模型继续使用既有插件方法。
- DSA 缓存 Hook 对齐当前 `index_kpool`、压缩尾部及 DCP 构造参数；压缩索引缓存与对应内存估算调用当前有效上游实现，保留 adapter 的索引布局适配。普通 BF16 索引缓存仍由既有插件路径提供。
- HCU 分页分配和请求表优化改由 plugin 自己持有算子入口。Prefill 优先调用 LightOp 的 `hcu_alloc_extend_kernel`；Decode、请求表写入和末位置读取从固定 DAS 的 `transfer.cu` 抽出函数体，经独立 JIT 扩展加载，不覆盖 adapter 的 `sgl_kernel`，也不加载整套 DAS 算子库。来源与逐文件哈希见 [原生分配来源](hcu-allocation-source.json)。初始化 allocator 时预编译，避免首次 Graph 中编译；空批次直接返回，校验 dtype/device/连续布局；空间不足在 native kernel 启动前拒绝。原有环境开关关闭优化时仍保留当前有效上游路径。
- GLM 的 `index_kpool=4` 保留官方 256-token KV 页配置，使池化后的物理索引页为 64；旧插件只对无池化 DSA 保留 64-token 页覆盖，避免误把 GLM 索引页缩为 16。
- MHC 延迟加载使用插件已有的 CP 模式兼容入口，移除对官方已不存在的 `is_dsa_prefill_cp_round_robin_split` 名称的直接导入。TileLang 的 bool allocation 补丁仅应用于实际提供旧 `Allocate` 接口的版本；镜像中的新版 pass 使用 `tirx.AllocBuffer`，不再套用旧补丁。
- FlashAttention 的 HCU MHA 布局优化不接管带 `qv` 的 MLA 调用；这类调用转交当前有效的官方/adapter 路径，保留 `only_qv`、空 K 和原始缓存布局契约。GLM 的 rope dimension 为 0，不能对其 `k_cache=None` 套用 MHA 的 `.ndim`/布局转换。

实际证据与边界（本轮记录持续更新）：

| 检查 | 结果 |
|---|---|
| 两份 checkpoint 完整性 | INT4 103 个文件、90 个 safetensors；INT8 79 个文件、62 个 safetensors。文件大小和全部 shard header/index 对应关系已核对 |
| 修改前组合加载 | official + adapter + plugin 的 363/363 个 Hook 应用成功，三个具名 resolution seam 存在；adapter 日志另有 JIT 旧补丁告警，此结果不代表模型运行通过 |
| INT4 格式改动后组合加载 | 364/364 个 Hook 全部应用，三个具名 resolution seam 存在，无插件加载错误；后续 GLM 上游分派改动单独复核 |
| GLM 分派改动后组合加载 | 367/367 个 Hook 全部应用，三个具名 resolution seam 存在，无插件加载错误 |
| KPool 构造复验 | 真实 GPU 分配：page size 64/256、pooling 4、压缩尾部与 skip-topk 空层均按官方布局建立 |
| Plugin KV 原生分配 | 从固定 DAS 抽出的独立 JIT 扩展构建成功；LightOp Prefill、plugin Decode/请求表写入/末位置读取在 page size 64/256、batch 0/1/5/65 下与逐项参考一致，非空批次两轮变输入 Decode Graph 重放一致 |
| 实际布局对照 | 抽查第 3/20/44 层 expert 0 的 gate/down，共六组。低位优先解码且使用原始 scale 时，与对应 INT8 反量化权重余弦相似度 0.9869–0.9888；高位优先约为 0。此为格式判定证据，不是整模型精度验收 |
| 首次 GPU 探针 | 失败：LightOp 排序返回的 padding 存在未初始化负 token 索引，首个 GEMM 越界；已保留边界日志，并切换官方排序入口 |
| 修改后单层 GPU 数值 | 实际 INT4 gate/up/down 权重，转换前后反量化值逐元素相等；1/8/32 token 与相同动态 INT8 激活量化的矩阵参考对照，最大相对 RMSE 0.000659、最低 cosine 0.99999976，输出有限；包含输入幅度扩大 10 倍的 clamp 覆盖 |
| TP8 形状单层 GPU 数值 | 实际 INT4 单专家取 TP8 的 256-column intermediate 分片；1/8/32 token 最大相对 RMSE 0.00003725、cosine 为 1。此项只验证该形状，不替代真实八卡归约与模型生成 |
| INT4 TP8 首次完整加载 | 八个 rank 均完成权重加载，每卡权重约 19.64 GiB；随后缓存构造暴露旧 Hook 缺少 `index_kpool` 参数，已修复并完成下述真实权重生成验证 |
| INT4 TP8 dummy 预检 | 已通过八卡模型构造、KV 初始化和 attention backend 初始化；修复 MHC 后进入 KDA/DSA 前向，暴露 MHA Hook 对 `only_qv` 调用的误接管，修复后转入真实权重验证。dummy 权重不用于精度结论 |
| INT8 线性层 | 实际第 0 层 Q 权重、1/8/32 token，官方 W8A8 方法首次调用暴露 adapter 漏绑 `int8_scaled_mm`；adapter 补齐后与相同量化的矩阵参考逐元素一致，Graph 两轮变输入重放与 eager 一致；整服务预检另外暴露上游 ROCm 白名单不含该方法，由 adapter 局部适配，保留格式校验。此项不代表整模型通过 |
| MHC 前向数值 | 单独复现时，首个分歧位于 adapter 的 DTK prenorm GEMM 名称别名，输出平方和与上游契约不符。adapter 改接已有 FP32 split-K 实现后，plugin MHC 的 1/8/32 token 前后处理均通过对照，含 norm 融合的最大相对 RMSE 为 0.000360；融合 norm 的 pre 路径两轮变输入 Graph 重放与 eager 逐元素一致，整模型另行复验 |
| MLA `only_qv` | 通过官方公共入口及 plugin Hook 实际分派；page size 1/256、query 1/8 对照最大相对 RMSE 0.000043，各组两轮变输入 Graph 重放与 eager 逐元素一致 |
| INT4 TP8 eager 基础生成 | 真实 INT4 权重、BF16 KV、context 8192，三条短请求 HTTP 200，闭合 `</think>` 后的完整最终答案分别为 `4`、`Berlin`、`7319`，均正常 stop。初轮未传 reasoning parser，思考与答案同处 content；后续配方显式加 `--reasoning-parser glm45`。这不是并发、长上下文或性能验收 |
| INT8 TP8 eager 基础生成 | 真实 INT8 权重，每卡约 37.36 GiB；运行接口确认 `w8a8_int8`、`glm45`、TP8 与 BF16 KV。三条短请求正文分别为 `4`、`Berlin`、`7319`，思考单列 `reasoning_content`，均 HTTP 200 和正常 stop。加载日志中的 `compressed-tensors` 是 checkpoint 格式名，不是运行方法被替换的证据 |
| INT4 TP8 eager 两请求并发 | BF16 KV、`glm45` 解析器；调度日志确认第二条 Prefill 时第一条仍在运行，Decode 同时有两条请求；分别完整返回 `5`、`Paris`，均 HTTP 200、正常 stop，思考与正文分离 |
| INT8 TP8 eager 两请求并发 | 调度日志确认同批 Prefill `#new-seq: 2`；两请求分别完整返回 `5`、`Paris`，均正常 stop。只覆盖两条短请求，不是吞吐或稳定性压测 |
| 完整服务 Graph、长上下文、MTP、多节点 | 待验证；不沿用历史 FP8 服务结论，不把配置支持视为所有执行后端已验收 |

本轮验收矩阵扩展为单机 IFB INT4 TP8、INT8 TP8，以及 PD Prefill INT8 CP8/EP8 + Decode INT4 DP16/EP16。多节点资源由用户协调释放，不能停止现有占卡任务。

本轮按用户要求不新增单测；交付运行项目 L0 lint、导览/迁移计划快照检查、改动 Python 语法检查及 `git diff --check`。GPU 探针和模型服务属于本次适配验证，结果分别记录。


### M05 GLM E4M3 KV 与框架隔离（验证中）

本轮仅接入 SGLang；不复用 vLLM、DeepInfer 的实现或注册链。adapter 的 INT8 门控与 MLA KV 写入桥位于 `hcu_adapter/bridges/_sglang/`，GLM `V32_NO_ROPE` 读取实现位于 `sgl_kernel/flash_mla_fp8_emu.py`。MHC prenorm GEMM 使用 SGLang 独立维护的 `sglang_mhc_prenorm_emu.py`，不再引用 DeepInfer 使用的 `mhc_emu.py`。

文档的 PD 配方要求 P INT8 / D INT4 均使用 `fp8_e4m3` KV；已通过的 BF16 KV 结果只作为基线。GLM NoPE 的缓存行宽为 528 字节：512 个 E4M3 值和四个 FP32 scale。plugin 的 LightOp 融合写入要求 RoPE 64，不能接管 RoPE 0，现对非 64 维 RoPE 委托当前有效原方法。adapter 修正 packed KV 写入误入 HIP raw-FP8 分支的问题，并提供匹配布局的 sparse MLA 读取；不改变整个 memory_pool 模块的平台判定。

GLM 的 `index_kpool=4` 会将含 tail 的稀疏 Decode 路由至 FA3。上游 `_forward_fa3` 将缓存第 512 字节之后视为 RoPE，而 packed NoPE 的该区域实际是 scale。plugin 为这一组合增加 `_forward_fa3` 的窄 AROUND 路由，将稀疏页表和有效长度交给 adapter 的 `V32_NO_ROPE` 接口；仅限 HCU、packed NoPE、page size 1 的稀疏调用、无 softcap，其他情况保留 `original_fn`。这项额外执行路径在 plugin，不修改上游框架源码。首次服务 warmup 在 logits 因零长度 hidden states 失败；独立公共入口复现确认原 FA3 方法将两条查询返回为 `[0,64,512]`。新 Hook 已经由真实 plugin entry point 注册和应用，page 64/256 的两轮变输入 Graph、独立数值参考均通过。

小张量证据：修复前 scale 区仍为零，读取相对 RMSE 为 1；修复后与官方量化的缓存字节逐字节一致、attention 与独立计算结果相同。page size 64/256 的两轮变输入 Graph 重放与 eager 逐元素一致，包含跨页、有效 top-k 长度变化和零有效位置；LSE 最大绝对误差为 4.77e-7。仅 `V32_NO_ROPE` 新分支在本轮支持范围内。INT4 TP8 eager + E4M3 完整服务已通过三条串行请求（`4`、`Berlin`、`7319`）及两请求并发（`5`、`Paris`），均 HTTP 200、正常 stop，正文与思考分离；日志确认 Decode 同时有两条请求。INT8 TP8 eager + E4M3 随后通过相同的三条串行及两请求并发，均完整答案匹配、正常 stop，调度日志确认双请求同时 Decode。完整服务 Graph、长上下文、MTP、PD 和性能尚未验收。

2026-10-29 的 VMFault 排查已建立单卡、无权重加载的复现：旧 FA3 输出零行，投影后卷积输入为 `[3072,0]`、指针为零，但序列元数据仍为 `[0,24]`；调用真实 causal-conv kernel 后触发 XID 81，进程退出 134。修复路径输出 `[24,8,512]`，后续卷积与 FP32 参考相对 RMSE 为 0.0023463。该实验复现相同非法输入条件，并非原始全模型张量重放。复现脚本、原始日志、输入 dump、源码哈希和交接流程保存在 ignored `artifacts/glm53-int4-20261029/vmfault-handoff/`；本轮测试容器已停止，8 卡回到基础显存占用，交硬件排查。


### M06 GLM PD INT4/INT8 DeepGEMM 接入（验证中）

目标为 P INT8 CP8/EP8 与 D INT4 DP16/EP16，KV 均为 E4M3。固定来源为 GLM53 W4A8 镜像（20261027-1047，image ID a6cfca440e62）中的 SGLang ep_moe/layer.py、token_dispatcher/deepep.py 和 deepgemm 原生包。运行时不导入该镜像的 SGLang fork，仍使用官方 v0.5.22 源码与 plugin。

plugin 新增 layers/moe/deepgemm_int8.py 与 layers/quantization/deepep_int8.py，经量化聚合入口注册。显式选择 deep_gemm + deepep 后，SlimQuant W4A8 和官方 W8A8 INT8 方法使用原生 grouped GEMM：normal 为 contiguous，low-latency 为 masked。INT4 使用 pack_w4a8_moe_hipc_weight 和 HIPC 两个 GEMM，保留 checkpoint 转换后的 scale/16；INT8 使用原生统一权重重排与 i8 GEMM。标准 TP 的现有 Triton 路径不变。未选中该组合时 Hook 委托原有效方法。

DeepEP normal 使用每 token INT8 量化、256 对齐的专家段及 LightOp scatter/gather；low-latency 通过 DTK quant_type=1 分发 INT8 激活。两端均要求 deepep_dispatcher_output_dtype=int8。SwiGLU clamp 使用模型声明的 limit；routed_scaling_factor 经官方 top-k 融入路由权重，normal gather / low-latency combine 各应用一次。暂不支持输入侧路由权重、非 SiLU 激活或计算通信 overlap。adapter 的 SGLang DeepEP ABI 包装保留显式 DTK 原生调用，同时继续翻译官方 BF16/FP8 调用；vLLM 和 DeepInfer 无改动。

2026-10-30 在 gfx936 的原生小规模数值检查：INT4/INT8 × masked/contiguous 共 4 项 GEMM 通过，relative RMSE 约 0.00165。plugin 公共 quant_method.apply 的两层 MoE 检查覆盖 normal 空/非空、masked 空专家及不等 token 数，共 10 项通过；与逐专家参考结果的 relative RMSE 为 0–0.00751。上述为 ignored 运行诊断，未新增仓库单测。完整 P/D 首次启动被官方 CP/TP group sharing 对 disaggregation_mode 的限制挡在参数解析阶段；随后先分开验证单机 IFB CP8/EP8 INT8 与双机 D DP16/EP16 INT4；CP PD 接入与完整生成结果见 M07。Graph、MTP 和长上下文仍待验收；原始证据保存在 ignored artifacts/glm53-pd-20261030。

### M07 GLM CP8 Prefill 与异构线性状态传输（验证中）

2026-10-30 的 IFB INT8 CP8/EP8 双请求首次复验在 DSA 元数据边界失败：查询已经按 CP 分片，top-k 仍有 25 行，超过本地输出 8 行。进程内探针确认 DSA utils 绑定的是平台 seam 安装前的 is_hip，导致 is_dsa_enable_prefill_cp 返回 False；adapter 在 SGLang bridge 的既有 DSA 初始化处同步当前平台函数。修复后真实 RuntimeContext 的 CP/interleave 判定均为 True，其他框架注册链未变。

plugin 的 disaggregation_glm5.py 仅为 GLM Next + HCU + Mooncake 的 CP=TP、interleave、PP1、attention DP1、无 speculative/radix/unified-memory 的 Prefill 放行 CP/TP group sharing；其他配置沿原方法。复用现有 disagg 的 bootstrap 映射（CP rank 注册为状态 TP rank）及 state_tp_size 分片发送，不修改官方源码或在线共享 manager 坐标。Decode 当前限定 attention TP1：所有 8 个 rank 发送各自 KDA 状态分片并确认完成，只有 rank0 发送完整 MLA KV 与其他复制状态。新增 AROUND/AFTER 沿 disaggregation.register_hooks 聚合，包裹现有有效方法。

P INT8 CP8/EP8 的配置探针已通过，D INT4 DP16/EP16 已完成 warmup。真实配置的 CPU 传输边界诊断已通过：8 个 rank、卷积 [3,24576] 与 KDA [64,128,128]，80 个块覆盖 4,341,760 字节，无遗漏或重叠，拼接内容与各 rank 的确定性输入一致；诊断以 memmove 代替 RDMA，只验证现有公共发送方法的切片和新 Hook，不代表网络传输通过。真实 P INT8 CP8/EP8 → Mooncake → D INT4 DP16/EP16 + E4M3 KV 已通过三条串行请求（完整正文 4、Berlin、7319）及两请求并发（5、Paris），均 HTTP 200、正常 stop，正文和思考分离；日志确认 DP3、DP4 同时 Decode。另以 1,237-token 输入按 512+512+213 三块完成 Prefill，并通过公开 PD 接口定向到第二台 Decode 节点 DP8；完整返回 7319、HTTP 200、正常 stop。该用例命中新长度的 MQA kernel 首次编译约 25 秒，不能作为性能基准。完整服务 Graph、MTP、更长上下文和压力测试仍待验收。原始日志和诊断放 ignored artifacts/glm53-pd-20261030，未新增仓库单测。

### M08 GLM PD Graph 与原生 MTP（验证中）

2026-10-30 用户明确验收门槛：相同权重、数据集和生成参数下，各精度集相对原 SGLang 的下降不超过 2 个百分点；性能完全对齐原 SGLang 配方，以预热后重复测量的吞吐、TTFT、TPOT 及端到端延迟比较，不自行设置可接受退化比例。短确定性请求只算功能冒烟，不等于精度或性能达标。文档原配方的 32K Prefill chunk、LayerSplit/HiCache、ReplaySSM 与更大 Graph batch 必须单独对齐；当前 8K context / 512 chunk / batch 1、2 不能代替原负载验收。原始脚本及门槛归档在 ignored artifacts/glm53-pd-20261030/reference-recipes 与 acceptance-criteria.json。

MTP 首轮进入调度后发现既有 prepare_for_draft Hook 仍按两个返回值解包，而 v0.5.22 增加了 kv_loc_plan。Hook 改为只读取并修改第一个 ForwardBatch，原返回对象整体透传；draft_model_runner 的位置参数同步到第五项，verify Hook 同样保留原返回对象。另在 adapter 的 SGLang JIT 接口中修正 EH norm 的 HIP 线程上限；本轮启动不能计为 MTP 生成通过，需按修正后的组合重新验收。

修正后的三节点 MTP 已完成加载、Graph 捕获和 HTTP 请求，但输出正确性未通过：两请求并发的 Paris 请求复读至 512-token 上限，正文为空；同一请求串行定向 DP3、DP8 也出现杂词、复读及结束标记泄漏，不能解释为仅并发异常。原始结果归档为 pd-mtp-smoke.json 和 mtp-badcase-serial.json。下一项受控对照仅关闭 Decode Graph，保留两端 MTP、P INT8 / D INT4、E4M3 KV 和相同请求；根因仍待定位，完整精度与性能验收未完成。

eager 对照首轮在空 attention-DP rank 的 DSA 元数据初始化处退出，原方法对空 CPU 长度求 max。该路径本身缺少空 batch 支持，扩展归 plugin 的 dsa_empty_batch Hook：为零请求且零 token 的 TARGET_VERIFY 构造空 DSAMetadata，累计长度为 [0]，非空请求保持 original_fn。诊断阶段先以临时 adapter rebind 检查了同一构造逻辑，仓库交付已归位到 plugin；临时部署结果和正式 Hook 的服务结果分别记录。树接受/提交的 batch 1、2、17，六步卷积窗口及 KDA 的 batch 1、2、64 heads、128 维 FP32/BF16 状态探针均通过，仍不能排除服务中的索引或流同步问题；未新增仓库单测。

eager 第二轮通过空 DSA 元数据后，KDA target verify 在零请求分片的 `view(0, 6, -1)` 处退出。plugin 同一空分片模块补充 KDA Hook：仅 HCU 的空 TARGET_VERIFY 返回 `[1, 0, num_v_heads, head_v_dim]`，不访问或修改卷积/SSM 缓存；非空路径保持原实现。正式 plugin 公共加载路径的 DSA 空元数据与 KDA 空输出探针通过，KDA 使用无状态 backend 对象确认无需访问缓存。Decode 已按正式 Hook 重启；后续 SSH 超时，完整启动与生成结果尚未读取。该修复不作为此前 MTP 输出异常的根因结论。

网络恢复后，正式 plugin 空分片修复的 eager MTP 服务已就绪；固定 Paris 请求按 DP3/DP8/DP3/DP8 四次复验均正常，另三条串行（4/Berlin/7319）与两请求并发（5/Paris）均正常结束、无此前杂词。结果记录为 mtp-eager-badcase-serial.json、pd-mtp-eager-smoke.json；这仅将异常范围缩到 Graph 相关路径，尚非完整精度验收。下一项 ignored 诊断只保留 target Graph，关闭 draft decode/extend Graph，不作为正式性能规避。

仅保留 target Graph、禁用 draft decode/extend Graph 后，DP3/DP8 固定 Paris 请求仍出现杂词和结束标记泄漏，证明该失败不要求 draft Graph。独立诊断中 INT4/INT8 masked DeepGEMM runner 的变输入/变专家计数，以及 KDA 六步 B1/B2、FP32/BF16 状态变输入/变索引 Graph 重放，均与 eager 一致；这些结论不覆盖服务元数据与通信。下一项诊断比较同一 target-verify 的逐层输出，先恢复卷积状态再执行 eager，产物单独归档；精度/性能验收仍未通过。

随后按专家 ID 对齐真实路由权重，并保存 DeepEP 接收数据：有效请求的 MoE 输入、gate、共享专家一致，路由权重差约 1e-7；Graph 中十五个空闲 rank 各六行 padding 的八个 ID 均为 0、权重均为 0，专家 0 因而额外接收 720 条，超过每专家 256 行容量。eager 空分片没有这些路由。plugin 的 INT8 DeepEP 扩展在 normal 与 low-latency 的 dispatch 入口将零权重槽位 ID 规范为 -1；low-latency 在 dispatch_a 传入新的 TopKOutput，使 dispatch、runner 与 combine 使用一致的 ID，不只修正发送侧局部参数。该接口转换仅覆盖 plugin 已启用的 INT8 dispatcher，供 W4A8/W8A8 共用；不改其他框架或全局 TopK 行为。服务复验与完整精度、性能验收继续单独记录。

在 M07 的 eager 基线上继续启用 Decode full CUDA Graph，先覆盖每个 attention-DP rank 的 batch 1、2；Prefill CP 保持 eager。Graph 的捕获、变输入生成和同一 DP rank 双请求另存证据，不能沿用 eager 结论。

首轮 16 个 rank 均完成 Graph 捕获，真实算术请求完整返回 4 并正常 stop；重放约 5.8 秒/token，性能未通过。两步 trace 定位到 adapter mHC prenorm 的逐 split 小算子链，已在 SGLang 独立实现中批量化并完成算子级数值及变输入 Graph 诊断，服务 A/B 继续进行。启动探针另确认 adapter 自动配置会提前建立 GPU 0 上下文，设备日志报 12 个进程超过每轮调度上限 8；计划使用上游每进程单卡可见配置验证，不把此相关性直接写成已确认性能根因。原始 trace 与探针日志保存在同一 ignored artifact 目录。

mHC 批量化后，完整 Decode Graph 捕获约 23 秒（前轮约 96 秒），三串行正文 4/Berlin/7319、两并发正文 5/Paris 均 HTTP 200 和正常 stop；三串行含 Prefill 的耗时分别约 26/49/42 秒，输出 28/59/50 token，不作为独立 TPOT 基准。进一步将两条请求固定到第二台 Decode 节点的 DP8，正文 7/9274 均通过；日志确认同一 batch 的 running-req=2 且 cuda graph=True。3,637-token 分块输入与 MTP 继续验证。

长输入首次进入 ragged MQA 后，对照官方 torch 实现发现 adapter 的既有 Triton/torch 档均漏掉 per-head ReLU；旧长输入运行停止，不计通过。修正归 adapter 的 SGLang 路径，负点积反例、ragged、64/256 page 的 preshuffle 布局与变输入 Graph 诊断已通过；32-head 循环取消静态展开后 cold compile 约 2 秒。MQA catalog 状态降为 candidate，修正后的长输入与 MTP 服务结果单独验收。MTP 配方使用上游 SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1，并用 CUDA_VISIBLE_DEVICES 配合清除 HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES；单进程可见性探针已通过，完整通信待启动复验。

单卡配置首次完整启动暴露子进程继承全卡 HIP mask 的问题，PyNcclCommunicator 初始化失败。adapter 在 SGLang 独立 reindex 桥中同步 CUDA/HIP mask，退出上下文恢复父环境；物理卡 3,1 的双子进程反序分配和真实 PyNccl AllReduce 已通过。三节点 MTP 已按此修正重新启动，配置/小探针结论仍不替代实际生成验收。

真实权重头确认 INT4 checkpoint 的第 45 层 NextN 专家为 INT8，eh_proj/enorm/hnorm 为 BF16；复用 SlimQuant 的混合精度规则与官方 NextN mapper。官方 PD Prefill 只有 EAGLE 的 draft_input 才输出 draft hidden states，因此 MTP 需配置两端；plugin 的 CP PD 配置入口扩展到同一 checkpoint 的原生 EAGLE/NEXTN、top-k 1，其他 draft 架构和 top-k 仍拒绝。目标先按文档使用 5 steps、6 draft tokens；配置解析、加载、状态传输、接受率和生成分别验证，尚未据此宣称 MTP 完整可用。

## 5. 后续代码开发与记录规范

每次代码改动追加台账，写明完成模块/行为、源码或前置提交、提交 SHA/标题及文件范围、实际执行的验证和原始证据、未验证模式、已知限制与下一步。提交后的代码 SHA 可在后续记录提交中回填，避免用文档自身未产生的 SHA 伪装已提交。

当前旧版 `sync_deepseek_methods.py`、审计 profile 与 `reconstruct_sglang_das.py` 部分路径仍固定 `sglang_dcu_plugin`。本次不把新包直接宣称为这些旧工具的受支持输入；新命名空间的抽取/embedded 生成适配、完整 GPU/模型复验作为后续独立步骤。普通开发和严格加载已按实际命名空间选择，不能把旧抽取 profile 当成新布局。

修复后的完整 target verify、draft decode、draft extend Graph 已通过四次 DP3/DP8 原始坏例、三条串行、路由器两请求并发、固定 DP8 两请求并发，以及 3637 token 输入（512 token chunk）的跨 chunk 回忆请求；Decode 回答与 reasoning 正常且以 stop 结束。该轮为 8K context、Graph batch 1/2 的定向回归，不代表原文档大上下文配置或完整数据集精度/性能已达标。首次路由 smoke 因旧 router 熔断返回 503，重启自有 router 后重跑通过；没有把该 503 计为模型生成失败。原始请求及结果保存在本任务 ignored artifacts 的 `*padding-fixed*.json`。

新增候选 `dsa_lightop_mla`：注册官方 DSA `_forward_flashmla_kv` AROUND，沿原脚本 `SGLANG_DSA_HCU_USE_LIGHTOP_DECODE_GATHER` 显式启用（默认关闭）。仅处理 BF16 query、V32_NO_ROPE 528-byte E4M3FN/uint8 连续行、top-k 2048、最多 64 heads；其他情况调用当前 original。先按物理 token ID 做 LightOp gather/dequant，再用紧凑索引调用原生 BF16 FlashMLA；全空行使用安全占位并恢复零输出，不改变输入索引或缓存，Graph 内无 CPU 同步。原生数学探针 1/6/12 行、4/32 heads、每形状三次变输入回放通过；实际 AROUND 函数入口复验通过；完整服务注册及整模型验证仍待完成。不将该探针结果写成性能或精度验收。

### GLM-5.3 HiCache / LayerSplit 接入（2026-11-01，验证中）

GLM 混合注意力缓存通过 `_hybrid_full_attention_pool_class` 选择独立的 LayerSplit 子池；draft 保留原池。沿用官方 owner-broadcast 和 host pool builder，HiCache 声明将 KV/indexer 层集合与当前 CP rank 的所有权求交，不把远端空占位缓冲传给搬运内核。混合池导出的 KV 层 ID 只覆盖本地缓冲；indexer 在公共状态装配出口补齐对应全局层 ID，供 PD 按层映射，不依赖连续指针偏移。Mamba 保持已有按 TP 头切片传输；LayerSplit 时各 CP rank 发送其 KV/indexer，其他复制状态仍由 rank 0 发送。混合池透出分片属性，使官方 PD 只从末 rank 发送 draft KV。KV 的 host 布局保持原脚本 `layer_first`，Mamba host 单独使用其支持的 `page_first_direct`。

该扩展只进入 HCU GLM-5.3 的显式 LayerSplit 配置；不复用 vLLM/DeepInfer。关闭 LayerSplit 后保留当前有效构造路径。两仓组合 GPU 初始化、前缀命中/驱逐回载、MTP Graph、长输入、PD 状态、完整精度与性能均须逐项验证，不能由本地语法通过推出已支持。

本次恢复读取原版完整报告：HumanEval 164 题 97.56%、MATH-500 500 题 95.60%、GPQA Diamond 198 题 89.90%、HMMT-Nov25 30 题 80.00%，全部请求成功。以上是基线；候选分数和性能对齐仍待实测。

2026-11-01 接续验证：Prefill 等待 JIT 的原因是上次重启遗留 allocation lock；确认自有容器没有编译进程后归档该锁，KV/indexer/Mamba HiCache 完成初始化。首请求暴露 KPool 将 sparse 改派 FA3 后沿用 ragged 元数据、page_table_1 未赋值。Decode 则被 flashmla_kv 的 KPool guard 拒绝。plugin 的 LightOp 扩展现仅对已启用的 V32_NO_ROPE、top-k 2048、kpool 4、2051 列索引保留 sparse 路由或开放 KV 路由；实际算子将索引用 -1 补齐至 2112 列。sparse 路径不使用历史序列长度截掉末尾三列。adapter 独立扩展原生 sparse 契约的掩码补齐与 query head 补齐；未改官方源码。原生两路径及正式 Decode Hook 的尾部数值/Graph 探针已通过，正式 sparse 接口与完整模型复验继续进行。未新增仓库单测，所有诊断保存在 ignored artifacts。

2026-11-01 完整配置预热的下一处首错为 `_kpool_assemble_softmax_rotate_write_cache_kernel` 向 0x4000 写入；VMFault 的首个内核参数为空，非 owner 的持久 index buffer 为零行。GLM LayerSplit 子池现将非 owner 的写 buffer 定位到现有 remote scratch，owner 保留持久缓存；每次写前仍失效读缓存，读时重新广播 owner 数据，压缩 tail 状态仍按原流程维护。8-rank 原语复验通过：每 rank 九次跨层写入/读取与 owner 数值逐字节一致，非 owner 持久层保持零行；完整服务验证进行中。adapter 同时修复 MLA HiCache 地址表被当作普通张量索引的问题；独立双向字节拷贝、布局和 Graph 探针已通过，尚不代表真实 host 驱逐回载通过。

Mamba host 备份/回载诊断捕获 JIT 编译失败：`hicache.cuh` 的 page-unified 入口使用 runtime 地址解析，但缺少 `runtime.cuh` 显式 include。已补齐头文件依赖；不将仅编译成功视为回载精度验证，真实 `MambaPoolHost` 静态传输入口的 FP32/BF16、两种状态形状、三层备份与逐层回载四组对拍全部逐元素一致；完整服务驱逐回载仍待验证。

完整 CP8 预热在修复空缓冲后进一步暴露 ragged 行数不一致：query/top-k 补齐到八行，`topk_indices_offset` 仍为三行。LightOp sparse 扩展现仅在已支持的 KPool ragged 路由内，为单次 attention 调用补齐零偏移；原始 metadata 在返回或异常时恢复，indexer 后续仍见真实行数。公共 `forward_extend` 的未补齐/已补齐 top-k 两类共 16 组检查通过（真实行数 1/3/7/9，物理行数 8/16，heads 4/64），每组三次变输入 Graph 回放与独立单键参考精确一致，metadata 恢复身份一致；完整服务仍待复验。

真实 host allocator 补充验证使用与本轮服务一致的 `mmap + cudaHostRegister`，确认其张量及视图可被 `is_pinned()` 识别；12 组 MLA 双向布局搬运及每组三次变输入 Graph replay 均逐字节一致。独立尝试 `HSA_USERPTR_FOR_PAGED_MEM=0` 分支时，上游查找 `libamdhip64` 失败，该分支不属于本轮服务环境，不记通过。15:21 完整 CP8 P 通过 MTP/HiCache/LayerSplit 预热并就绪，KV 容量 4,878,848 tokens；首轮 P/D 请求因旧 router 不健康状态返回 503，保留记录后仅重启本任务 router。端到端、驱逐回载及完整对比继续验证，不能从启动成功推导验收通过。

本轮三串行、两并发 P/D MTP 请求均正常，但 4096-token 首次可缓存请求在 MLA host 写回暴露 draft 映射越界：device target 层数 11，压缩 host target 层数 2，draft 错用 host slot 11。GLM LayerSplit 扩展为 MLA draft 备份/回载建立单次浅拷贝调用视图，保留原有设备 draft tensor，将调用层号映射到 host target 层数加 draft depth；不修改共享 host 对象或目标层映射，继续使用已适配的原始传输实现。索引 host 池已有独立 draft 映射，无需重复修改。数值及完整服务复验待运行。

MTP 层映射修复的正式插件加载验证通过：同一 GPU 上构造 8 种 CP rank 层分配，每种分别 1/2 个 draft，共 16 组调用真实 `MLATokenToKVPoolHost.backup_from_device_all_layer` 和 `load_to_device_per_layer`，使用真实注册 host 内存，目标和 draft 搬运逐字节一致，非 owner 空层不参与搬运，原对象缓冲 tuple 身份未变。这是小容量对象的传输验证，非八卡完整模型或驱逐回载验收；P 随后重启继续复验。


4097-token 的固定 DP8 冷请求/命中/再次冷请求对照确认，命中 4096-token 前缀时首个错误已发生在 P 首 token，两个冷请求输出一致；因此暂不进入完整精度和性能评测。实际 adapter KDA fork 丢弃官方传入的 `track_state`/`track_chunk_idx`，而官方后端仍按 `supports_track_state_snapshot` 使用未写入的 FP32 缓冲。修复放在 adapter 原接口兼容层：参数贯穿 router/fork，递推在指定 chunk 开始处写 FP32 状态；不能用 BF16 中间 h 替代。快照调用避开尚无该输出接口的 AITER 分支。旧代码哨兵复现、独立前缀状态对拍及完整模型复验分别记录，尚不宣称缓存正确性通过。

KDA 受控对照完成：旧 adapter 输出/最终状态有限，但 FP32 快照 32,768 项仍全为预填 NaN；修复后公共 `TritonKDAKernel.extend` 的单序列、变长批次、跨层 stride 初态均通过。快照与独立截断前缀状态、从快照续算的输出及最终状态与完整序列均精确一致，负数跟踪索引对应行仍为 NaN。完整 Prefill 服务随后重启复验原始命中失败用例，内核通过不替代模型验收。


16:00 的 KDA 修复完整模型回归通过：固定 DP8 的 cold/hit/cold_again 首 token 均为 2097，D 的八个输出 token 一致；router 连续命中两次也一致。128 个并发压力请求后，原请求 device hit 为0、host hit为4096，回载输出与初始一致，MTP 正确草稿10/提出10。单 rank 计数 KV 回载4096、Mamba回载2；indexer 使用 KV 索引联动，不单独累加计数。以上为原失败场景和驱逐回载验证，完整数据集仍待跑。

同配置98304-token/32768-chunk profile 暴露性能差距：候选11.77秒，原版约4.23秒；rank0 GPU kernel累计11658.7ms对3966.7ms。主要候选热点为 ragged MQA 2838.1ms、INT8 scaledMM1822.1ms、广播1123.9ms。首轮候选短请求性能跑数在完整测量前停止，保留日志但不计验收。adapter MQA 矩阵分块候选已通过数值和变输入 Graph 探针，4096×8192×32×128代表形状28.2ms降至3.03ms，完整P仅更新该项重启采集独立收益。INT8 dense LightOp选配接口的数值/Graph/回退检查通过，尚未部署进完整服务。

LayerSplit 按请求页广播的候选保持物理地址不变：在 DSA metadata入口从CPU长度镜像和设备req-to-token构造固定页列表，打包 owner 的有效页，广播后写回 remote scratch 对应页。没有CPU长度镜像或页数达到整池时沿用原广播；跨批次先完成pending广播并失效旧remote层标记。索引和主KV使用相同物理页号，各自保留原布局；8-rank字节/重复页/跨批次检查及完整回载与性能验证待完成。

按页广播的8-rank诊断通过：三批页号变化、全部八个owner、KV与indexer、重复共享页均逐字节一致，未请求页保持哨兵值；进一步使用真实PyNcclCommunicator和原始prefetch/readout的异步stream路径也通过。首轮脚本误用每rank不同的collective root顺序，失败日志单独保留，该轮不计实现证据。候选尚未部署完整模型，后续仍需缓存回载和profile复验。

MQA 优化单独进入完整 P 后，固定 DP 缓存对照、router 前缀命中及 128 请求驱逐后的主机回载均通过；98304-token profile 为10.2027秒，MQA累计248.28ms（优化前2838.10ms）。随后 P 显式启用 adapter INT8 LightOp 候选，其固定 DP、命中和实际 host4096/device0回载也通过，长输入 profile 正在采集。四项完整候选精度及最终九次性能测量仍未完成。

MoE 路由热点由 adapter 复用官方 Triton 融合实现，候选仅覆盖 FP32、288专家、top8、单组、无融合共享专家、M≥32。公共包装、别名、重复安装、FP32 数值及变输入 Graph 检查通过，非支持输入原样调用已有 provider。BF16 原路由重复比较曾不一致，失败日志保留；候选不接管 BF16，未据此修改其现有行为。该候选尚未进入完整服务。

INT8 dense 单独完整 profile 为7.3414秒，GPU累计7254.92ms，原生dense624次累计566.33ms；其中torchTopK也从1340ms波动到241ms，不能把全部总耗时变化归因于dense。Decode128 profile识别到paged MQA先展开整个索引缓存再选择Triton的冗余工作；adapter将分发前移后，4种几何和每种3次变输入Graph与旧路径逐位一致，16384页代表形状0.709ms降至0.050ms，尚待完整D复验。

gfx936 INT8 grouped GEMM 的配套ASM权重布局与调用参数候选保持一致，INT4保留原路径。独立contiguous数值及Graph对照通过，代表gate/up形状0.331ms降至0.293ms、down形状0.188ms降至0.155ms；真实runner masked/public边界与完整P验证待完成。未直接切换计算标志而复用不同布局权重。

ASM runner 候选未通过原有 HIPC 路径的逐元素容差检查：动态专家计数129/3时，87/1572864元素超过rtol0.008/atol0.001。微基准的速度收益不足以替代完整FFN数值边界证据，本轮撤回该候选，保留原权重打包和HIPC执行；诊断源码及失败日志保存在ignored产物。前述contiguous单GEMM通过仅适用于其独立输入，不能外推至带激活重数量化的runner。

固定组合候选包含有效页广播、MoE官方融合路由和paged MQA提前分发。独立广播启动期间更新共享源码导致后续子进程读到新桥接/旧catalog，已终止该未完成实验并统一重启三节点；因此下一次profile属于组合对照，不能将总差异全归广播。部署核查60个改动运行时文件：59个字节相同，另一个仅模块docstring不同，可执行AST相同。源码固定后再执行完整回归和评测。

固定组合候选完整回归通过：fixed-DP冷/命中对照、router命中、128请求压力后host4096/device0回载均保持输出一致，MTP10/10；98304-token profile6.6961秒，GPU累计6446.42ms。trace中有效页广播标记实际出现，Broadcast66次累计38.42ms（上一轮922.74ms），MoE变为129次官方融合router累计7.09ms。剩余热点包括DeepEP notify/combine与grouped GEMM；端到端仍未对齐原版约4.23秒。16:58启动固定候选九次性能与四项全集评测，结果待完整收集，不把profile计时当作性能验收。

九次固定组合性能测量已完成，每轮16请求全部成功；三个场景吞吐中位数分别为原版的0.6472、0.6278、0.7177，未达到性能对齐要求。随后完整HumanEval在随机MTP验收路径首次触发adapter缺失的`flashinfer.sampling.top_p_renorm_probs`，D进程退出，后续请求503；已停止评测并保留失败日志，不将该轮计作准确率。adapter宿主桥补齐top-k/top-p两个导出，复用官方已有HIP Triton实现；公共入口数值、变输入Graph和后续tree sampling正在诊断，四项准确率需修复后重新完整执行。

随机MTP链路第二个缺口为tree sampling的FlashInfer头文件依赖，已由adapter的target-only Triton候选承接现有数学行为。小/奇数/163840词表、三组阈值、分支与零概率、每组3次变输入Graph，共9组与独立CPU参考的接受索引、计数、拒绝概率写回及最终采样一致；实际公共入口和完整接受链Graph也通过。初版FP32残差CDF在距边界约4e-8处选中相邻token，失败样本保留，最终CDF改用FP64累加后通过高精度参考；树验收保持FP32。原生镜像AOT树采样入口缺少导出，不能宣称对拍原生AOT通过；原生服务采用另一条target sampling路径。该候选需以相同四项全集评测确认整模型精度。两仓L0、修改Python语法及diff检查通过，没有新增单测。三节点按冻结修复版本重启，随后先执行3串行/16并发随机请求，再跑四项精度和九轮性能。

修复后的整模型随机回归于17:35通过：temperature1/top_p0.95，3串行及16并发共19请求均HTTP200、完整答案正确且正常stop。随后按原参数重启HumanEval164题，其他三项及修复版本九轮性能依次排队；目前仍未形成完整精度验收结论。

随机采样修复版本的 HumanEval 已完整完成：164 次请求成功、0 请求错误，163/164 正确，99.39%，较原版 97.56% 高 1.83 个百分点，满足该数据集下降不超过 2 点的要求。唯一未通过题为 HumanEval/116，生成触达 16384-token 上限且最终代码为空；未将该题归为请求错误。MATH-500 正在运行，GPQA Diamond、HMMT-Nov25 与九轮性能随后执行；四项精度和完整性能验收仍未完成。

评测冻结期间只读核对三节点 sysfs 拓扑，均为 GPU 到 NUMA 节点 0,3,2,1,4,7,6,5。独立 CPU 子进程诊断共 24 个，实际 CPU affinity 与每项内存映射 bind 策略均匹配；没有修改模型服务源码、配置或进程绑定。后续 NUMA 性能对照须在当前全集评测结束后单独启动，并再次检查真实 scheduler 进程。FFN 未初始化/NaN 缓冲区的 16-rank DeepEP combine 与 Graph 诊断仅已准备，尚未运行，不能据此移除清零。

随机采样修复版本的 MATH-500 于19:11完成：500次请求成功、0错误，474/500正确，94.80%，较原版95.60%下降0.80个百分点，满足该数据集下降不超过2点的要求。沿用原评分器与16384-token上限，未对格式误判候选手工改分。HumanEval与MATH两项已通过；GPQA Diamond开始，HMMT与当前版本九轮性能仍待完成，整体未验收。
### GLM5.3 masked FFN 初始化开销（2026-11-01）

随机采样修复版已完成 HumanEval 99.39% 与 MATH500 94.80%，分别较原版提高 1.83 点、下降 0.80 点。GPQA 在 102/198 时记录 5 次 7200 秒超时重试，已归档为未完成并停止；HMMT 和该版九轮性能尚未执行，整体未验收。

冻结代码下，16 并发真实 GPQA 输入、每请求 256 输出 token 的诊断 trace 中，所查 rank 的随机模式 BF16 清零累计 2.638 秒、DeepEP dispatch/combine 累计 5.930 秒；排序、扫描和归一化合计 0.147 秒，不能把本轮超时归因于随机采样算子。该短段 profiling 不替代正式基准。

`hcu_sglang/layers/moe/deepgemm_int8.py` 仅将 masked 路径的 gate/up 与 down 输出缓冲改为 `empty`，normal 路径保持原实现。真实 16-rank DeepEP→FFN→combine 对照覆盖 INT4/INT8、均衡/单活跃 rank/256 热点行/全空/补齐/129 热点行，共 192 条记录；零初始化、未初始化与 NaN 填充的有效专家行及最终 combine 逐位一致，变输入 Graph 与 eager 逐位一致。消费边界仅使用 `masked_m` 指定的有效行。证据位于 ignored `artifacts/glm53-pd-20261030/resume-20261101/masked-ffn-combine-*`，整模型 19/19 随机采样、固定 rank cold/hit/cold、128 压力请求后的 host 4096/device 0 回载均通过，输出一致；九轮性能正在执行，四项最终精度与性能对齐仍待验收。首次 smoke 曾遇路由器 503，重启自有路由器后复验通过，失败记录独立保留；没有新增单元测试。


masked-empty 九轮性能已完成，每轮 16 请求均成功。1024/128 单并发、1024/128 十六并发、32768/128 十六并发吞吐中位数依次为 28.197、189.562、47.923 tok/s，分别为原版的 69.77%、54.98%、65.16%，尚未对齐。随后只绑定 24 个 scheduler 的全部线程到各 GPU 对应 NUMA CPU，未改内存策略、线程数或源码；19/19 随机请求和固定 rank 缓存对照通过，九轮吞吐中位数为 28.180、231.427、51.751 tok/s，分别为原版的 69.73%、67.12%、70.36%。短输入十六并发吞吐较未绑定提高约 22%，但 TPOT 未同步改善，各场景仍不满足逐指标无退化要求。原始三轮波动全部保留，未调整容差或剔除样本。最终候选四项精度仍待执行，不能复用采样修复版的两项成绩充作完整验收。


### GLM5.3 normal dispatch 的共享专家重叠（2026-11-01）

现有 SBO 公共计算入口按三维 low-latency dispatch 解包，normal dispatch 使用二维激活；即使跳过该解包，INT8 runner 也会拒绝表示无 down-GEMM 重叠的空字典。plugin 的 batch-overlap Hook 对 HCU normal dispatch 返回 `(None, None, {})`，runner 接受该空元数据，实际 low-latency 重叠参数仍拒绝。其他 dispatch 调用当前有效原函数。

真实 8-rank INT8 normal DeepEP→共享专家 hook→FFN→combine 探针，通过公共 Hook 注册且预先导入消费者别名；原函数的二维解包错误可复现，修复后 33/129/3/0 行、不均匀空 rank、零权重补齐的 32 条记录与串行输出逐位一致且有限。非空重叠元数据继续抛错。首轮诊断保存的原函数引用被 Hook 别名传播更新，导致反例断言失败；改为容器内保存原引用后重跑通过，原始失败日志保留。该变化只接通 normal 通信期间的共享专家计算，尚未证明端到端提速；原验收配置的整模型回归、九轮性能及四项精度仍待执行。未修改官方源码、未新增单测。


DeepEP 版本隔离诊断：当前版本 16-rank、INT4/INT8 的 192 条 dispatch→FFN→combine 数值与变输入 Graph 记录通过。原版快照在当前镜像能导入，但 GPU `Buffer.sync` 初始化发生 SIGSEGV，未进入数值与计时，故未替换服务依赖。两次启动前还观察到共享 NFS 头文件读取阻塞；adapter 对 `jit_dpskmeta` 的内容未变写入加保护，临时文件重复执行保持 inode/mtime、注册仍完整，未宣称已消除所有 NFS 故障。修复后的原版诊断未再停在该读取阶段，而是上述 native 初始化失败。原始日志和未通过状态独立保留；当前服务继续使用原有 DeepEP。


SBO 候选在原 P INT8 CP8/EP8、D INT4 attention-DP16/EP16、E4M3 KV、MTP、Graph、HiCache/LayerSplit 配置下完成整模型回归：随机 3 串行加 16 并发共 19 请求通过，固定 rank cold/hit/cold 输出一致；128 压力请求后明确命中 host4096/device0，回载输出一致、MTP 接受 10/10。九轮原脚本性能已开始，四项最终精度仍待执行。该臂只在 P 开启 SBO，未保留上一臂 CPU affinity、未更换 DeepEP；启动 I/O 的内容未变保护不改变 kernel 转换结果。原版 DeepEP 崩溃栈进一步定位为 ROCSHMEM `GetClosestNicToGpu/select_nic`；隔离探针缺少原版配方的显式拓扑文件，故暂不能归因为二进制 ABI，不在当前性能臂混入该配置。


SBO 九轮性能全部完成，每轮 16 请求成功、0 错误。三个场景吞吐中位数为 33.615、288.352、52.449 tok/s，原版比例 83.18%、83.64%、71.31%；逐项 TTFT、TPOT、端到端延迟均仍未对齐。相同 98304-token Prefill profile 为 6.225 秒（上一组合 6.696 秒），所查 rank 的 combine 通知等待累计由 893.84 降至 490.49 ms，FFN 仍约 578.62 ms。真实 GPQA 16 请求×256 输出诊断，greedy profile 15.140 秒、随机 profile 16.156 秒；随机 rank0 的清零累计 228.06 ms、DeepEP 6940.17 ms、gatherTopK 1080.16 ms。两轮 model step 数不同，不能把累计耗时差直接当端到端收益；profiles 已完整导出，不替代四项全集准确率。后续只暂停 D 做数值/通信诊断，P 保持此版本。


小批次路由候选未接入：单卡实际 MoEGate logits 为 FP32，110 次变输入 Graph 的专家集合一致、权重最大差约 8.94e-8，6 行微基准由约 0.092 降至 0.010 ms；但真实 16-rank INT4 DeepEP→FFN→combine 在 rank14、1 行场景有 1/4096 元素超出既定 rtol0.008/atol0.001（绝对差0.00390625），因此不扩大 Triton 路由范围。原始失败日志与部分通过记录独立保留，不能以单算子对拍替代 FFN 消费边界。启动还观察到 dot_product JIT 头文件的无变化重复写入冲突；adapter 对该入口增加内容不变时跳过写入，保留原校验和登记语义，后续多进程启动独立复验。


DeepEP 拓扑/设备可见性对照已补齐。显式使用原版 topo.config 后，原版快照在单卡可见模式仍于 ROCSHMEM 选网卡处崩溃；改为全部8卡可见、rank选择实际GPU后，原版与当前版本均通过192条INT4/INT8变输入Graph/FFN/combine检查，跨版本192组最终输出逐位一致。两臂使用相同CPU/内存NUMA绑定与拓扑，包含三组FFN/combine的整图中位数：当前INT4 2.283ms、INT8 3.237ms；原版2.643ms、3.351ms。此为受控诊断整图时间，不是纯通信或整模型性能，未替换服务依赖。原版单卡可见崩溃不能再归为一般ABI不兼容。当前版本的单卡可见拓扑臂为1.942/2.395ms；两种可见性不可直接归因为通信版本收益。两个全部可见版本启动日志均无Traceback。此前dot_product短暂ENOENT后，三节点头文件均重新可读，sha256与预期补丁结果一致，未重新写入官方部署文件。

### GLM5.3 原生 mHC 候选（2026-11-01）

同一镜像的 AITER tilelang pre/post 经官方公共 `hc_pre/hc_post` 入口和 Hook 验证。覆盖1/6/12/24/32/128/512/4096行、3轮变输入Graph，输出有限且Graph/eager逐位一致；对当前plugin组合与独立Torch参考沿用原相对RMSE<0.015标准，所测最大约0.000427。pre明确返回 `norm_fused=False`，消费者执行输出RMSNorm后再比较。包括前后处理及归一化的微基准6行约0.074→0.041ms，4096行约2.565→1.128ms，不代表整模型提速。正式Hook以 `SGLANG_HCU_NATIVE_MHC` 默认关闭，非覆盖形状回到当前有效基线，不切换旧宽范围mHC环境项。最终注册入口加入随机输出归一化权重后，8种形状各3轮变输入Graph通过，最大相对RMSE约0.0003684；显式关闭保留有效original。2026-11-02复验中三节点24个rank均记录原生mHC命中，3串行与16并发随机请求全部正确且正常stop，固定rank的cold/hit/cold一致。128个压力请求后合成确定性缓存探针输出逐token一致，回载host4096/device0，MTP接受10/10；该缓存探针不作为自然语言精度成绩。九轮性能均16/16成功；短C1/短C16/长C16吞吐中位数31.860/283.761/56.931tok/s，为原版78.84%/82.30%/77.41%，TTFT、TPOT与E2E均未对齐。98,304-token profile耗时5.786s（上一SBO臂6.225s）；mHC三类内核合计166.1ms，原版155.6ms。真实GPQA16输入×256输出profile的greedy/stochastic为13.110/21.474s，随机轮rank0 DeepEP dispatch/combine合计12.773s，含通信等待，不能等同有效计算或单独归因为通信库。四项全集精度待最终候选完整执行，尚未达到完整验收。

仅移除mHC emu输出复制的候选在10种split/空行形状上逐位一致，6行约0.025→0.020ms；尚未部署，不与原生mHC臂混合。小批次路由的简化Torch eager与compile候选未保持原专家顺序，均未接入。原始失败和通过记录均保存在本任务ignored artifacts中。


### GLM5.3 NUMA 绑定对照（2026-11-02）

在原生mHC候选上仅加入按物理GPU的CPU与内存NUMA绑定，映射0/3/2/1/4/7/6/5，OMP保持4。三节点24个scheduler的CPU掩码均精确匹配，内存映射均有对应bind策略；vendor的prefer/local映射也保留，不宣称所有内存页均落在本地。随机请求19/19、固定rank缓存、128压力后的host4096/device0回载输出一致及MTP10/10均通过。九轮性能全部16/16成功；短C1/短C16/长C16吞吐中位数27.790/263.291/58.947tok/s，为原版68.76%/76.37%/80.15%。短输入相对未绑定原生mHC退化，长输入吞吐略好但TPOT更高，三个场景逐指标仍未对齐；该绑定未保留到后续RCCL元数据同步对照。四项全集精度仍待最终候选执行。


### GLM5.3 RCCL 元数据同步对照（2026-11-02）

相对未绑定原生mHC候选，仅启用上游 `SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1`；未启用skip-all-gather。24个scheduler均核实开关为1、CPU允许128核、无NUMA bind内存策略。19/19随机请求、固定rank缓存和压力回载host4096/device0输出一致、MTP10/10均通过。九轮性能全部16/16成功，短C1/短C16/长C16吞吐中位数29.936/293.360/56.141tok/s，为原版74.07%/85.09%/76.33%，三个场景逐指标仍未对齐。GPQA16输入×256输出profile的greedy/stochastic耗时13.367/18.153s；随机轮rank0不再出现Gloo all-gather，出现520次RCCL AllGather（合计1.118s），确认实际通信路径切换。DeepEP合计6.670s，含通信等待及不同调度轨迹，不作单独因果结论。该开关未保留到后续独立回传路径对照；四项全集精度仍待最终候选执行。


### GLM5.3 forward-stream D2H 对照（2026-11-02）

仅在D节点临时spawn bootstrap中选择上游HIP结果回传分支，未修改仓库运行时代码；16个scheduler实际run_batch均验证命中。19/19随机请求、固定rank缓存及128压力后host4096/device0回载输出一致、MTP10/10均通过。九轮性能全部16/16成功，短C1/短C16/长C16吞吐中位数24.398/230.774/55.880tok/s，为原版60.37%/66.93%/75.98%，三个场景逐指标均未对齐；相较正式native-mHC，短输入吞吐明显退化，不保留该候选。GPQA16输入×256输出profile的greedy/stochastic为14.846/17.983s；rank0主要DtoH拷贝分别126/150次落在forward kernel使用的stream221，确认路径实际改变，但profile不能替代完整性能或精度验收。两次早期bootstrap启动失败均未进入工作负载，失败日志独立保留。

诊断及profile均已结束，原始响应、summary和rank0 trace已归档。下一轮恢复正式native-mHC入口，清除临时D2H、NUMA及RCCL元数据开关，以独立arm重新跑整模型回归、九轮性能和HumanEval/MATH500/GPQA Diamond/HMMT-Nov25四项全集；不复用旧版两项精度作为当前验收，也不因性能尚未对齐而继续推迟全集精度。


### GLM5.3 prefill 归因与 normal INT8 缓冲候选（2026-11-02）

正式 native-mHC 配置四项全集已完成：HumanEval97.56%、MATH50095.40%、GPQA90.40%、HMMT80.00%，分别相对原版0/-0.20/+0.50/0个百分点。HE/MATH来自full-eval臂；GPQA为保留188条有效输出加10题续跑，已逐条确认原model_output不变，HMMT为30题完整运行。新router7500秒且无重试、客户端7200秒、EvalScope一次尝试后没有新超时；GPQA1题、HMMT6题达到65536输出上限，截断原样计分留存。此成绩只归属于候选修改前的冻结源码，不能自动继承到后续版本。

原始profile的完整hidden复制来自官方interleave CP在每个KDA前后恢复token顺序；原版跨层使用plain连续分片，KDA直接gather，仅MLA边界转布局。不能直接删除官方重排，否则会破坏token与SSM状态对齐。所谓215.6ms AllGather实际为embedding all-reduce的内部阶段；重复采集时早到rank等约529ms、最后到rank约4.94ms，第二次全部回到5.01–6.71ms，说明首次采集入场偏斜，不足以判断稳定网络带宽不足。第二次rank0填零约157.95ms、复制约256.12ms；这些为profile归因，非正式benchmark验收。

本次单变量仅把`HCUDeepGemmInt8Runner._normal`的INT8 gate_up/down预填零改为空分配。保留INT4 normal分支、总量为空的返回、最终输出和scatter输入/scale/padding初始化，以及原GEMM、激活、路由和masked路径。8-rank实际DeepEP dispatch→FFN→combine诊断覆盖33/129/3/0/257/128行、部分rank空和零权重路由；48条记录中空分配及NaN污染中间缓冲后，combine输入均与原zero分配逐位相同，参考combine输出有限。该诊断不等于全模型精度；当前P配置本来禁用Graph，未外推normal Graph支持。首次未按既有NUMA启动器运行的隔离诊断在权重分配阶段OOM，日志保留；恢复既有绑定后通过，不改变在服NUMA设置。

本地实现与CI等价lint待完成；部署、服务回归、候选完整九轮性能和四项全集精度仍待执行。改前对照使用同一原benchmark脚本、三场景各三轮、五项指标，原始失败与全部逐轮结果保留，不设置性能回归容差。

Resource incident correction: the NUMA-bound diagnostic passed 48 numerical records but competed with the resident model. At 11:27:51, kernel logs recorded a memory-policy OOM that killed a task-owned P scheduler and monitoring processes. The subsequent control benchmark failed during warmup and produced no valid performance measurement. The original passing source and launch configuration are being restored; the allocation candidate is local only. Future independent GPU probes must run in a separate window with owned model services stopped. A combined recovery driver was rejected by automatic review as a co-resident GPU risk. Static inspection confirmed the service clients use only standard-library HTTP; a safer single HTTP request after health recovery was approved.

Stable eight-rank trace: of 129 combine notifications, the last entering rank also finished GEMM last in 127 cases and had the largest expert token count in 120 cases. Rank0 spent 485.31ms of 488.50ms before the final rank entered. Median cross-rank max/min padded-row ratio was 2.71 and GEMM-duration ratio was 2.73. This locates most notification wait behind uneven FFN completion; no routing or communication-library change has been made.


### GLM5.3 短序列 KPool 索引与环境对齐（2026-11-04）

使用 hcu-plugin-dev 与推理精度排查流程，沿固定 token 前缀比较三题、每题两次、每次八个 prefill rank，共24对。此前迁移环境中，完整第0至2层边界及第3层投影/归一化/吸收Q一致；第一个差异是相同零 logits 的 torch.topk 索引顺序，索引集合相同。48次采集均命中 DTK sparse MLA；保持Q/KV不变、交换索引顺序后，24对均逐位复现各自attention输出，证明该输入上的顺序因果，不等于已解释GPQA总分和复读。

新增 dsa_kpool_short Hook，迁入原生 IndexerKPool 短序列有序 arange 分支，保留逻辑索引、PAGED/RAGGED映射、无效项和tail padding。官方模型源文件不改。远端冻结候选使用原部署命名 is_hcu，本仓使用 is_dcu；本地移除新增来源说明docstring，运行逻辑一致。候选先前L0和语法检查通过，三题修复后数值及无插桩badcase尚待验证；不新增单测。

进一步审计发现，既有原生基线与迁移使用不同容器镜像和算子包版本。按用户要求改在文档第117版镜像（0.5.18-ubuntu22.04-dtk2604-py3.10-sf_b075-20261019-1254-zhiputx）现有容器中运行上游源码、plugin和adapter；三节点镜像ID及FlashMLA、LightOp、DeepGEMM核心共享库哈希一致。原生包保留，代码通过独立路径加载。DTK26.10升级暂不执行。

首次切换未进入模型请求：文档容器默认工作目录为根目录，Python3.10的PYTHONPYCACHEPREFIX配合相对文件路径触发cache_from_source的IndexError，继而留下部分初始化的torch._dynamo模块。最小复现中根目录失败、工作目录成功；启动脚本显式切到工作目录后P/D导入探针通过，无需更换Torch或算子包。整模型启动和同前缀数值验收仍在进行。

仍待核实的源码分支包括长序列pooled top-k的AITER/LightOp dispatch、MQA logits的量化及实际算子、KPool缓存写入/尾部状态、head-gate精度与MTP共享索引。源码差异只作为审计线索；必须采集真实命中和同输入/同状态数值后归因。badcase保持九题并发9，固定前缀配对仍串行以保持批次条件。

### 2026-11-04 远端候选同步与交接分支

`feat/glm5.3-flash` 完整同步 short-topk-alignment 的 decode plugin 运行时与打包配置，避免遗漏 row_fusion 和量化接口依赖。保留 W8A8/SlimQuant 双路径；配套 adapter 同步 FP32 beta 候选。来源哈希、静态验证范围、脚本及未闭环问题见 [交接文档](glm5.3-flash-handoff.md)。此次未启动模型，不继承历史全集精度。

### 2026-11-04 主对话复核后更新交接

已归档远端 24/24 prefill 重复性配对及九题 6/9（错7/90/127，127仍复读），不归属统一提交组合。补独立 router.sh、无插桩官方包哈希与启动前源码检查；诊断树仅 model_runner.py 追加 observer 安装。网页配置按用户要求暂不接入。本次仅静态验证，未启动服务或新增单测。详见 [交接](glm5.3-flash-handoff.md)。

### 2026-11-06 P/D 性能路径静态复核

当前实验启动配置为 `--quantization w8a8_int8 --moe-a2a-backend deepep --moe-runner-backend deep_gemm`。`layers/quantization/deepep_int8.py` 的 `around_create_runner` 在 `requested()` 命中后构造 `DeepGemmInt8Runner`，`around_apply` 及 `around_run_moe_core` 调用该 runner。P 的 normal 路径为 `layers/moe/deepgemm_int8.py:_normal`，在 CPU counts 列表上做 256 对齐检查后复制至设备；D 的 low-latency 路径为同类的 `_masked`。源代码接线只说明该配置下的预期路径，真实服务命中和耗时仍需 profile 证明。

此前本地提交 `252f36e` 消除了 `lightop_deepep.py` 中的 counts 回读，但该函数以 `("deepep", "lightop")` 注册，不能作为当前 DeepGEMM 部署的优化。已撤回该候选及其专用测试，没有部署或性能收益结论。后续必须先证明实际 runner 命中，再比较同硬件、同参数、同负载下的算子与端到端耗时；保留 DeepEP、EAGLE、CP、HiCache 等原功能。

目前 SCNet 跳板机持续出现 SSH 连接超时或认证后命令无响应；不启动远端 shell 的端口转发同样未建立。未据此判断原 P/D 服务退出，也未重复加载模型。完整模型 profile、数值与精度回归、P/D 优化及 A/B 均为 pending。实验脚本与结果继续使用容器 `/home/work/agent/das-glm53-tx-pd/perfopt-20261105/`。本次只完成静态调用链复核，L0 和导览检查通过，不继承历史 GPU 验证为当前验收。

### 2026-11-06 冷启动架构检测与 GPU 上下文

按 hcu-plugin-dev 与配套 hcu-adapter-dev 规约，保留 `get_gcn_arch(device_id)` 原有按设备属性查询；仅架构布尔检测在 CUDA 未初始化且 adapter 可用时复用其架构元数据。adapter 缺失及已初始化路径仍查 Torch 属性。gfx936/gfx938/gfx942 分类与 native FP8 布尔语义不变，无新增配置。提交范围为 detection、离线回归及本记录。

配套 adapter 同时消除 import 期 GPU 分配并接管冷启动硬件查询。实际 P/D 环境的完整 bootstrap 探针无 CUDA 初始化尝试，新进程 KFD 显存全零；D 重启健康 200，8 个 scheduler 各占本卡，改前 CPU worker 与跨卡 scheduler 的 GPU 0 上下文已消失。原 trace 的通信等待与 GPU 0 runlist 超订阅是诊断证据，提速仍需同配置新 profile 和预热 A/B。不得继承历史精度。证据位于实验 `perfopt-20261105/cold-start/`，P 启动及 AIME25/HumanEval 尚待完成。

本轮离线测试 67 项中 66 项通过；唯一失败仍为既有 launcher 测试期望旧 bootstrap/官方模块转发，而当前入口转发 `hcu_sglang.launch_server`，已用改前 HEAD 复现，不属于本次 detection 修改。L0、导览重建与 freshness 检查通过。浏览器预览因本地 `file:` 协议被工具安全策略拒绝，未进行截图、交互及窄屏验收，也未绕过该限制。

### 2026-11-06 normal INT8 专家计数拷贝候选

P 新 trace 的 `hipMemcpyWithStream` 累计 3.697 秒/994 次，包含多种传输，不能全部归因于专家计数。实际 `DeepGemmInt8Runner._normal` 每次使用 `torch.tensor(counts, device=x.device)`；原生 `ep_moe/layer.py` 与官方 FP8 normal runner 的普通 offloader 分支使用 pinned CPU tensor 加异步设备拷贝。本候选在该调用点对齐，目标显式为输入设备。offloader 禁止 copy engine 时保留原有直接 tensor 路径，不在本候选扩展 offload 支持；量化、padding、scatter/GEMM/激活/gather 及 output 初始化保持原样，未改变 low-latency 路径。

真实公共 `run`→normal→combine-input 的 CPU 接缝回归保留计数、两个 GEMM 及 gather 出口，普通路径在改前失败、改后通过，并核实禁止 copy engine 时保持旧路径。数学和 vendor consumer 不由 mock 证明。首次设备探针发现 adapter 未提供官方 no-CE helper 所需的 `sgl_kernel.elementwise.copy_to_gpu_no_ce`，因此撤回新增 helper 调用；这不是本次在服 P 的执行路径，不能宣称 no-CE 支持。失败日志保留。候选文件与 baseline 独立保存于实验 `normal-counts-candidate/`，仅在已确认 GPU 空闲的 a12r4n13 测试；没有部署到在跑 P/D，不改变当前 AIME25/HumanEval 的代码。GPU 数值、预热算子时间与整模型 A/B 仍为 pending。

后续 `probe-v2.log` 与 `result.json`：GPU 0/3、真实 LightOp scatter→两次 DeepGEMM→激活→gather，BF16 输出与原路径逐位一致，共 16 项，覆盖 1/16/129/257 行、零计数专家、两种 offloader 配置（禁止 CE 时仅证明保留旧路径，不证明 no-CE 语义）。使用真实 4096 hidden / 2048 intermediate / 4 local experts；没有实际多 rank collective，不能代替 EP8 整模型验收。两卡每臂预热 5 次，三轮交替顺序、每轮 30 次，小样本候选约 0.38–0.39ms、原路径约 0.43–0.45ms。此为该 FFN 样本的主机加设备时间，不是整体 P 提速。P normal 原配置禁用 Graph，本轮不宣称 normal Graph 支持。整模型 A/B、profile 与数据集回归仍待候选部署后的独立验证。

设备环境为 gfx936、Torch 2.11.0、HIP 6.3.26113、驱动 6.3.31-V1.5.0a；结果附内核版本和 topology 架构字段。当前离线 68 项中 67 项通过，唯一失败仍是上述改前可复现的 launcher 用例；L0、diff 与导览 freshness 检查通过，浏览器验收仍受前述协议限制。
