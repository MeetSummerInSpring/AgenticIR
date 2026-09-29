# 单卡 DepictQA 参数训练与独立验证

本入口用于研究比较器的内容保持、双图偏好、同图平局及不对应场景拒答。它不更改生产权重、恢复工具或在线接受规则。2026-09-29 的实际资料、检查点及运行日志保存在服务器 `experiments/midterm_v5/`；业务图像与权重不入 Git。

## 已验证环境与模型

已有 DepictQA 作者实现固定在 `22f5795f297af59a996c1ee67fd9881db653c5e9`，使用项目的 `config_comp.yaml`、Vicuna-7B-v1.5、CLIP ViT-L/14 和 DQ495K_Abstractor delta。不要替换成退化等级 delta。现有 Python 3.10、PyTorch 1.13.1+cu117、Transformers 4.35、PEFT 0.3 环境通过 `venv --system-site-packages` 建立研究入口，继承已安装依赖且不升级原环境。

冻结主干、CLIP 和 abstractor；更新 rank-16 q/k/v/o LoRA 与视觉投影，共 20,975,616 参数。主干 FP16、可训权重 FP32、AMP/GradScaler、梯度累积 4、梯度检查点。仅缓存确定性的冻结视觉特征，缓存键含原图、视觉权重、delta 和预处理指纹；投影始终参与反向传播。

旧作者 Llama 的梯度检查点接口与新 HF 签名不一致，因此设置其已存在的布尔开关。单样本作者分词把提示内 EOS 当作 padding，而其原生生成将掩码重设为全 1；研究封装只对无填充的单样本对齐注意力。拒绝多样本输入，避免把真实 padding 放开。作者仓库未修改。

## 执行

先释放本项目占用的服务，确认没有其他 GPU 任务。命令均在项目根目录运行，输出目录必须是新的。

```bash
/root/autodl-tmp/conda/envs/depictqa/bin/python -m venv --system-site-packages experiments/midterm_v5/train_env
experiments/midterm_v5/train_env/bin/python -m unittest eval.test_vlm_training

# 使用已冻结的本机数据清单；重新构造时应使用含现代 Pillow 的 agenticir 环境。
experiments/midterm_v5/train_env/bin/python -m eval.vlm_training train \
  --data experiments/midterm_v5/data/train.json \
  --val experiments/midterm_v5/data/val.json \
  --out experiments/midterm_v5/new_run \
  --cache experiments/midterm_v5/feature_cache \
  --epochs 3 --accumulation 4 --lr 0.00001 --seed 929

# 两个模型都已在验证集选定后再运行；不会按迁移结果重新选模。
experiments/midterm_v5/train_env/bin/python -m eval.vlm_validation_suite \
  --protocol experiments/midterm_v5/postselection_protocol.json \
  --mixed experiments/midterm_v5/train_aligned_mixed/best_adapter.pt \
  --controlled experiments/midterm_v5/train_aligned_controlled/best_adapter.pt \
  --cache experiments/midterm_v5/feature_cache \
  --out experiments/midterm_v5/new_postselection
```

`eval.prepare_vlm_data` 从 pack manifest 和预先确定的 train/val/test 排除组生成受控干扰、同图和非对应场景问答。`verify_rows` 在训练前检查来源组、配对第二来源组、像素哈希和跨集完全重复。当前文件路径以冻结服务器清单为准，迁移到另一台机器需保持相同图像哈希并更新清单路径。

## 统计与解释边界

旧开发资料 28 张来源图分为训练 20 张/5 组、验证 8 张/3 组；320/128 是双顺序问答数，不是独立图像数。来源组分离但有一个点位重叠。320 问答含 A/B 各120、Tie/Uncertain 各40。主要配方是“真实道路底图的受控偏好 + 真实同图/场景对应控制”，没有独立人工恢复偏好，不能称为已验证的真实偏好混合训练。

对照只使用训练组内 7 张相对清晰基图/3 组，112 条独立问答重复为320次曝光/轮；同为3轮240次更新。它同时改变数据量与多样性，因此不能隔离真实/合成比例的因果效应，也不把相对清晰图当作像素级完美真值。

选模规则固定为128问答验证集的四类宏平均正确率，同分保留较早轮。原模型也保留。新格式自由生成的不可解析回答单列；四个选项的平均条件对数似然作为免自由文本解析的补充，不是校准概率，且仍受答案词汇、长度先验及指令适配影响，不能单独作为视觉能力指标。原生提示的双图结果另列，不能与四选一结果混为一个指标。

训练后使用12张项目既有 HQ 库图及4张天脸开发图做受控迁移诊断，分别96/32问答；这部分不是新最终测试、无人类 MOS，不能证明完整通用能力或真实去雨提高。所有天脸图禁止参与参数更新和检查点选择。人工候选图保留待复核状态，不自动填入伪标签。

固定候选回放复用旧自动轨迹：两次顺序都偏好候选才接受，输入、候选、提示、预算和规则对各模型相同。它只诊断比较器及固定规则，未运行在线重规划；旧道路轨迹来源组与旧先导训练重叠，与新来源训练不重叠；它已被反复查看，仍不能充当新的最终泛化证据。

## 研究依据

优先复用现有可训练模型而不盲目换模型。作者的 [DepictQA/EDQA 实现](https://github.com/XPixelGroup/DepictQA) 提供相应比较与轻量参数训练路线，并提示部分数据混合会损害比较能力；因此保留未训练模型和基图受控对照。[DeQA-Score（CVPR 2025）](https://github.com/zhiyuanyou/DeQA-Score) 是质量评分分布路线的候选，作者两张3090的配方不直接视为本项目单卡验证结果。[LoRA](https://arxiv.org/abs/2106.09685) 支持冻结主干进行低秩适配；本轮实际单卡可训，未引入额外量化。[Syn2Real](https://arxiv.org/abs/2006.05580) 支持研究真实域差距的动机，本轮未复现其高斯过程方法。

接下来应优先补齐新的道路来源与少量独立人工偏好，再决定是否训练描述/置信度多任务或改用分布式评分目标。优化目标分开报告细节保护、过曝/伪影风险、平局和拒答、跨域保持、成本及最终恢复；不以单个 NIQE 或“雨等级降低”替代应用有效性。


## 街道003新来源增补

运行中已收到 `REQ-20260929-street-003`：20个新来源记录组、74张来源图，提供 train45/val16/test13。81个交付清单文件均通过哈希核验。固定每训练组最多首尾两张的稀疏规则，实际使用23张/12组（另22张保留），验证使用全部16张/4组。生成368/256问答，3轮276次更新，仍从作者delta重新开始；不从旧实验继续训练。模型只由新的验证集选定。

`eval.prepare_vlm_increment` 按请求排除旧组和已知同日关联，保存分组、实际PTS及来源标签；共享摄像头及未知相机编码单列，不声称跨摄像头独立。图像已有AI辅助现象标签，仍无人类恢复偏好。

`eval.prepare_vlm_final_test` 必须看到新来源训练完成、选定检查点与数据哈希一致，才创建启封记录并解码13张最终测试图。所有构造后的测试条目标为 `final_test`，训练入口拒绝使用。测试上的扰动与任务格式提前固定，不按测试结果改标签、参数、裁剪或模型。

```bash
# 先固定新来源训练，再选模；输出目录按实际新实验命名。
experiments/midterm_v5/train_env/bin/python -m eval.vlm_training train   --data experiments/midterm_v5/new_sources/data/train.json   --val experiments/midterm_v5/new_sources/data/val.json   --out experiments/midterm_v5/new_source_reproduction   --cache experiments/midterm_v5/feature_cache --epochs 3 --accumulation 4   --lr 0.00001 --seed 929 --study-scope new_record_group_study
```

本轮完整复核命令是在上述 suite 命令中增加 `--fresh experiments/midterm_v5/train_new_sources/best_adapter.pt --fresh-test experiments/midterm_v5/new_final_test/test.json`。此时旧主配方和受控对照只检查旧验证集；未训练与新来源模型检查新最终测试、HQ、天脸和固定候选。旧对照不能作为新来源训练真实/合成比例的因果对照。
