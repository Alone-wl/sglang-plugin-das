# DAS 源码与 HCU 分支对齐记录（2026-11-02）

## 用途与基线

按 plugin-v3-fix 需求对齐当前 hcu_sglang 的逻辑及 is_dcu/is_hcu 分支。主执行 Skill 为 [能力开发](../skills/hcu-plugin-dev/SKILL.md)，审计使用 [迁移 Skill](../skills/sglang-hcu-plugin-migration/SKILL.md)。输入为固定三方源码与容器，交付为模块代码、本地提交和本轮验证记录。

插件起点为 `3a2d0e5bac78dc66468c46fc5da1e223e8c076c2`；DAS 为 `15c7204b7a3b6edaa222cff80018d8e5f2b2ad0e`；official 为 `baa9befa381511cd4d56f83980951e5a45fffc6d`。来源保持只读。验证在指定节点的 yangyn_sglang_plugin_1022 容器中使用 Python 3.10.12，显式将固定 official/python 与插件加入 PYTHONPATH，避免容器默认旧版本接口混用。

## 代码与分支处置

| 模块 | 处置与真实消费者 |
|---|---|
| 公共运行时 | 按 Source 的 HIP 检测、FP8 架构与 provider 优先级修正；补齐环境项、冻结 ServerArgs 派生配置、Req/ScheduleBatch/ForwardBatch 生命周期、CP 策略和 DSA 元数据消费者。 |
| 计算与量化 | 补齐 MegaMoE 控制与权重布局、DeepEP INT8 生产消费、DP padding/collective、Marlin 注册与 SlimQuant BF16/clamp；保留正确 vendor ABI 名称。 |
| Attention 与模型 | 迁入完整 MLA/FLA/GDN/相关模型方法，接入 Qwen、Bailing、GLM 和多模态分支；消除重复通信与重复 KV 写入，已有上游行为由上游接管。 |
| KV 与拓扑 | 修正容量校验、分组释放、SWA/DSA/DSV4 分支、MLA 容量、CacheInitParams/retention、CP/DP group 所有权；补齐 prefix 协议和 LayerSplit 计划及实际 read/write。 |
| PD 与投机 | 同一 hidden state 表贯通 prefill/decode/scheduler/ACK/cleanup；迁入描述符传输、drain/ACK、DSpark worker/model/injector 和 HiCache 消费者。保留 descriptor-aware 校验与 READY 索引修复。 |
| KDA | 成对落地配置生产、Mamba pool、backend、卷积/状态 kernel、verify/accepted commit 与 rollback；保留原生类型身份，默认 HCU KDA 分支遵循 Source。 |

本轮没有按字符串将所有外部 dcu 名称替换。`megamoe.dcu_megamoe_opt` 等真实依赖接口必须保留。来源方法内的模块别名、f-string、descriptor 及预导入消费者均参与核对，不能用语法通过代替分支执行。

## 执行与证据

本轮 ignored 原始证据位于仓库 `artifacts/plugin-v3-fix/`，按 core、layers-moe、attention-models、cache-communication、pd-transport、kda-config 分组；其原始日志不随源码提交。各模块包含来源映射、哈希和针对性消费者测试。当前注册合同保存在 [tests/contracts](../tests/contracts/hcu_source_alignment_targets.json)，该合同只表示固定 HCU 默认配置的注册集合。

已执行真实 CPU 数据流、GPU 数值/状态和多 rank Gloo 小范围验证：计算模块 29 个 runtime 脚本的 197 项测试、W4A8 GPU 3 项；Attention CPU 15 项与 FLA GPU 4 项；HiCache 5 项、prefix 6 项、DSpark 6 项；KDA backend GPU 5 项及配置 7 项；PD transport 8 项；cache allocator 11 项、topology 10 项、PD frontend 7 项、Mamba CPU 4 项及 HIPGraph replay。具体夹具中的 stub 范围以各测试和模块证据为准，Gloo 是真实进程组通信。

285 份 JIT overlay 全量核查：281 份与 Source 字节相同，另外四份为包入口和既有 launch bounds、空 grid、HIP BF16/warp 适配。Rust 两条原生构建/运行路径通过；原 ignored Cargo cache 有损坏文件，本轮使用已有工具链与独立 cache 离线重解包，没有修改原 cache，也不对损坏来源作推断。

## 最终集成检查

- 仓库 unittest：117 项全部通过，显式启用 layer/GPU 测试开关，无跳过。
- 固定 official 源码加载：597/597 Hook 全部应用、4 个具名 resolution seam，错误为零。目标集合已独立保存；从仓库外加载独立 wheel 安装目录并与合同核对，同样为 597/597、4 seam、0 errors。
- L0：556 个 Python、8 个 Skill 通过；226 处 Markdown 锚点通过。
- guide、迁移计划及执行台账快照检查通过。计划检查只核对保存快照，不表示重新采集全部来源。
- 本地候选 wheel：721 个包文件与最终源码逐字节相同，包含 KDA HIP 源资产；SHA256 `30ba576f8e66fa15ee197c24b0b93e44d9557f8d20d77ab36e46b613f5e9345d`。没有覆盖系统安装，使用独立临时 target 验证。

复现基础检查：

```bash
export PYTHONPATH="$OFFICIAL_ROOT/python:$PLUGIN_ROOT"
export HCU_PLUGIN_RUN_LAYER_FIX_TESTS=1 HCU_PLUGIN_RUN_GPU_TESTS=1
python3 -m unittest discover -s tests -v
python3 ci/l0_gate.py
python3 tools/build_guide.py --check
python3 tools/build_migration_plan.py --check
python3 tools/build_migration_execution.py --check
```

严格加载前先调用 `hcu_sglang.compat.aiter.bootstrap_aiter_compat()`，随后运行 `tools/validate_plugin_runtime.py`，指定固定 official、当前插件、`--plugin-package hcu_sglang` 及 `--expected-targets tests/contracts/hcu_source_alignment_targets.json`。与空 target、旧安装或错误路径混用时应判失败。

最终空白检查去除了少量新文件的 EOF 空行；方法逻辑与之前模块测试一致，最终 wheel 对照使用清理后的字节。代理早期方法哈希记录对应其冻结时刻，最终包字节以 wheel-members-final.json 为准。

## 验证边界

以上证明具体方法、注册、状态与消费者，不证明全部 fork 行为、全部模型、shape 或性能等价。没有执行新的完整模型服务、完整多进程 PD 服务或八卡数据集验收。容器缺少 deepgemm.mega 及部分 AITER API，这些 provider 的完整数值与 Graph lane 保持未验证；没有用其他实现替代来源算法来制造通过。

没有发布或部署产物，没有修改来源仓库或安装升级系统依赖。当前合同与 wheel 只用于本轮固定环境复验。历史 plans 快照保持原来的审核边界，不能据本轮局部测试自动关闭全部历史 pending。

界面控制权限不可用，因此网页截图、交互及窄屏视觉验收未执行；生成快照和链接检查不能替代视觉验收。
