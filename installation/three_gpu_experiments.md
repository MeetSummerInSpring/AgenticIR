# 三卡自主系统验证

使用 `eval.local_service` 将原生 DepictQA severity/compare 各绑定一个实际获配 GPU UUID；运行器和工具继承第三张 GPU 的 `CUDA_VISIBLE_DEVICES`，各进程内均使用逻辑 cuda:0。服务仅监听 localhost。先查看 `nvidia-smi`，不得停止其他任务；服务停止命令只处理自己的 PID/启动时间记录。

```bash
python -m eval.local_service start --mode severity --out experiments/RUN/services --gpu GPU-SEVERITY
python -m eval.local_service start --mode compare --out experiments/RUN/services --gpu GPU-COMPARE
CUDA_VISIBLE_DEVICES=GPU-TOOLS python -m eval.run_manifest \
  --manifest experiments/RUN/dev_manifest.csv --out experiments/RUN/dev_B1 \
  --method B1_current --mode auto --limit 8 \
  --catalog experiments/RUN/catalog.json \
  --tool-profile experiments/RUN/profile_k1.json \
  --memory-snapshot experiments/RUN/calibration.sqlite3 \
  --max-tool-calls 12 --max-llm-calls 12 --timeout 1800
```

`--catalog` 现在可独立于单卡 staging 使用；必须覆盖所有注册任务，并仅列出已准备好权重和环境的工具。工具顺序由文件明确指定，B0/B1共用同一清单。`--tool-profile` 在实验目录复制并进入续跑身份，未设置则沿用生产配置。K=1需要显式命名为实验配置，不能写成默认K=3；证据不足仍保留全部候选。冻结记忆与运行日志分库，正式测试不在线更新统计。

主对照 B0 使用 `B0_registry_no_retrieval`（注册顺序、无调度检索），并非原始随机基线。选择消融使用 B1 的工具选择器配合 `--schedule off`；与 B1 的调度开关差异单列。报告实际调用、Top-K触发分母、命中规则ID、在线比较、回滚及计划调整。耗时包括每图推理和工具子进程启动；常驻服务的冷启动另计，不能直接把单卡阶段装卸时间写成架构收益。

原始图和参考图均留本机。`ExperimentAgent` 强制视觉任务使用本地 DepictQA，文本运输校验器只允许既有模型/端点和文本内容；配额拒绝停止，不自动更换模型。权限审批如有拒绝，先审计实际数据流与已有授权，再依正常审批流程处理。

## 冻结训练比较器的独立系统实验

停止本轮自己拥有的原生比较服务后，可在同一比较卡启动：

```bash
python -m eval.local_service start --mode compare --out experiments/RUN/trained_service \
  --gpu GPU-COMPARE --adapter experiments/TRAIN/best_adapter.pt
```

此服务只覆盖 LoRA 与视觉投影参数，严格检查完整键集和形状；原始 delta、提示与生成配置不变，无回答缓存。记录 adapter/config/delta SHA256，默认最大生成长度沿用原作者配置。它是独立实验，不会改写生产权重文件。两模型必须在相同输入、工具、策略与预算下重跑，不能把固定候选回放混称在线系统效果，也不能把VLM本身的评价当独立质量真值。实验结束用对应 `--out` 的 `stop` 命令释放服务。
