# Offline EMB AutoDL

## Fork and scope

- Fork: <https://github.com/cozylove-girl/wesep-real-tse>
- Upstream: <https://github.com/REAL-TSE/wesep-real-tse>
- Source commit: `2a540977a348fbaa92e623210505430e2cec608d`.
- Main entry: **`EMB_AutoDL_Full_Pipeline.ipynb`**.

这是离线 **BSRNN_EMB** 的云端训练流程，正式训练150轮。第一段 MODE 选择显存测试、正式训练或断点恢复；默认单卡4090先测试batch=4。分离器随机初始化，ECAPA声纹编码器加载预训练权重并冻结参数。

官方训练代码仍标为 WIP。本流程复现其公开 EMB 架构和主要训练配方，并明确记录数据登记、运行配置及修补。它不保证与已发布的 `spk_emb_100` 权重逐位一致或精确复得历史指标。

## Upload and run

1. AutoDL 镜像选含 Miniconda 的 **PyTorch 2.8.0 + CUDA 12.8**，Ubuntu 20.04/22.04 或更新，Python 3.10～3.12。默认 `ENVIRONMENT_MODE="image"`，复用 `/root/miniconda3/bin/python` 里的现有 PyTorch 和 GPU 库，缺 torchaudio 时只补匹配的 2.8.0。
2. 上传 `EMB_Code_Upload.zip`，解压到 `/root/autodl-tmp`，得到 `/root/autodl-tmp/wesep-real-tse`；也可上传完整仓库目录。
3. 打开 `cloud_upload/autodl_emb/EMB_AutoDL_Full_Pipeline.ipynb`，检查第一段参数，Run All。
4. 第一次设 `MODE="benchmark"`，按顺序运行；确认batch后改 `MODE="train"` 再按顺序运行。中断恢复选 `MODE="resume"`，沿用原 WORK_ROOT。

**所有模型训练/计时仅在云端执行。本机没有启动训练。** 本地检查只使用文件、JSON/YAML/AST 和算术。MODE=benchmark会训练并丢弃临时模型，不能称为纯推理。RUN_TRAINING/SETTINGS由代码同步。

### Reuse the installed PyTorch image

2026-10-10：针对用户的单卡 RTX 4090 镜像（Python `/root/miniconda3/bin/python`、torch 2.8.0+cu128、torchvision 0.23.0+cu128、缺 torchaudio），默认改为复用镜像。旧版本无条件建立 Python 3.10 环境并下载 torch 2.7.1；请中断旧安装、关闭旧 Notebook、上传更新后的代码 ZIP 并覆盖解压，重新打开 Notebook 和重启内核。

- 自动检查 torch/CUDA 和配套库，缺少/不匹配的 torchaudio、torchvision 用 `--no-deps` 安装；不会顺带再下载 torch。
- 固定已安装 torch 和 NVIDIA 库的版本约束，约束也传给评估安装步骤。
- 训练和官方评估均使用选定镜像的 Conda Python 前缀；不再固定为数据盘 `env/bin/python`。
- 其他训练依赖按 requirements 安装，可能调整镜像里的 numpy 等包；仍需通过依赖检查和真实 CUDA 预检。
- `environment_selection.json` 记录实际 Python、torch、CUDA 和 GPU 库版本；环境改变后不能混入已有正式训练实验。
- `ENVIRONMENT_MODE="isolated"` 是显式备选，会安装 Python 3.10、torch 2.7.1 和整套 CUDA 依赖。默认镜像模式失败会停止，不会自动回退并产生大文件下载。
- 不删除旧环境或下载缓存，也不复制不同 Python 版本的 site-packages。

镜像版本配对依据：[PyTorch 官方历史版本](https://pytorch.org/get-started/previous-versions/)。这次更新只在本地用模拟安装命令/静态检查验证，尚未远程访问用户实例执行安装或训练。

### Compact progress

Notebook 安装/下载进度使用同一状态框刷新，保留最近最多 8 行、约 1800 字符，不连续追加完整 pip/tqdm 日志。完整输出仍写入 `WORK_ROOT/logs/`，失败时显示更多错误上下文。终端模式最多每 30 秒报告一次状态，pip/HF 进度条关闭。

### Source speed and academic acceleration

LibriSpeech 下载默认 `LIBRISPEECH_SOURCE="auto"`：在云端对每份压缩包从最多 6 个源各读最多 1MiB，确认精确 Range 和官方长度，选择该次采样较快的源。来源包括 OpenSLR 官网、官网列出的 CN/EU 镜像、CN 实际指向的 k2-fsa/LibriSpeech，以及 HF 镜像。HF 固定版本为 `2dd4206a992cd7a475e1ce2e82b708b014319152`；完整文件仍校验 OpenSLR 官方 MD5。

- `ENABLE_ACADEMIC_ACCELERATION=True`：在 Notebook 内核加载 `/etc/network_turbo` 的代理设置，传给后续下载进程，代理值不打印到日志。只在另一个终端 source 不会改变已有内核。
- `LIBRISPEECH_SOURCE="huggingface"`：只测固定 HF 源，适合已开启学术加速；auto 会一起比较其他源。
- `MINIMUM_DOWNLOAD_MBPS=0`：默认不按速度强制中止。只有主动设正数才启用采样/120秒下载门槛；三个包共享带宽时不适合按单包速度贸然中止。
- 更换镜像复用旧 `.parts` 与偏移；速度单位为十进制 MB/s。每个包记录 `*.source_probe.json`，不保存带签名的跳转 URL。
- 拼接完成并通过官方 MD5 后自动删除已重复的下载分片和 `.part`，只清理固定下载目录内的生成文件；中断/校验失败时保留分片。旧版可能残留的完整分片在下次验证同一 tar 包时清理。测速 JSON 保留作为来源记录。
- 采样速度不是全程速度保证；无法从本机测速推断 AutoDL 实例线路，也没有在用户实例实测。

用户云端诊断显示：加速开关 False、内核无代理，HF 直连 ConnectionError；HF 镜像旧测速约 0.18～0.21MB/s。新版默认打开加速；重新执行 prepare 时也加载代理。测速将连接/跳转和数据传输耗时分开，避免 1MiB 小样本被握手时间误判。2026-10-11 用户报告 test-clean 因0.49MB/s低于0.50MB/s中止，故默认关闭强制速度门槛；已下载/解压的train、dev在重试时复用，仅补齐test。并行任务使用 as_completed 即时报告失败和解压成功项，保留成功文件，避免等最长下载完成后才看到早先失败。

针对该实例 HF 学术代理已出现 SSLError、镜像实际可下载的情况，Notebook 默认 HF_ENDPOINT 改为 hf-mirror.com。通用权重下载与本流程 HF Hub 下载使用直连 Session，镜像跳转到 CDN 时也不继承学术代理。TLS验证及固定权重 SHA256/长度校验保留；该实例实际初始权重/评估模型下载仍待执行确认。

正常停止/重新打开同一 AutoDL 实例时，磁盘环境和数据可复用。关闭/重启 Jupyter 内核只清空内存状态，需要重跑参数和代码定位。实例释放、系统重置/迁移的保留规则不同，按平台页面操作；最终权重和报告及时下载备份。

单卡4090先设 MODE=benchmark、BATCH_SIZE=4；代码完成数据准备和真实benchmark，显示峰值显存及训练+验证时间。未保存正式权重时可改BATCH_SIZE为8/15重跑测试，确定后选MODE=train。

### Batch test operations

操作已经写进Notebook的可执行代码，不需要另建单元格手工调用函数。

```python
# 先测显存：第一段这两行设为下列值，再Run All
MODE = "benchmark"
BATCH_SIZE = 4
```

```python
# 测更大batch：第一段仍为benchmark，只改batch，再Run All
MODE = "benchmark"
BATCH_SIZE = 8
```

```python
# 测试通过后：第一段改成train，batch保留通过的值，再Run All
MODE = "train"
BATCH_SIZE = 8
```

```python
# 训练中断后：第一段改成resume，WORK_ROOT保持原目录，再Run All
MODE = "resume"
# batch/轮数/其他训练参数自动读取原pipeline_settings.json，不用手动重填。
```

- benchmark模式跳过评估依赖、正式训练、权重平均和最终评估；第4节显示真实测速报告。测试时可用 `watch -n 1 nvidia-smi` 查看总显存，留约2～4GB余量。
- train模式复用相同配置的成功测速记录，第5节准备评估依赖，第6节训练150轮，其后自动平均、评估和打包。
- resume模式要求已有正式检查点；从最近保存的完整epoch恢复，不承诺逐batch恢复。已有检查点时benchmark/train模式会提示改为resume或新实验目录，避免误改batch混用实验。
- 显存测试默认batch=4是工程起点，公开配方batch=15；短试训通过仍不保证所有后续样本的峰值。

### Evaluation package availability

2026-10-11：用户 prepare_eval 因固定 `onnxruntime-gpu==1.20.1` 不可用而中止。已核对 PyPI 官方元数据：1.20.1 当前返回404，1.20.2提供 Python3.12/Linux x86_64 wheel（291,510,804字节），故评估依赖改为1.20.2。该1.20.x GPU系列使用CUDA12/cuDNN9，与当前PyTorch2.8.0+cu128的主要运行库版本匹配；实际官方评估仍需云端验证。

修复只更新评估 requirements，训练数据和原训练配方无需重做。继续执行 prepare_eval 可复用已下载DEV、已拉取评估仓库和子模块。新版Notebook第4节是显存测试，第5节是评估准备；benchmark模式不执行评估准备。

来源：[PyPI 1.20.2文件列表](https://pypi.org/project/onnxruntime-gpu/1.20.2/)、[ONNX Runtime CUDA/cuDNN版本表](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。

### Evaluation model TLS/download fix

2026-10-11 用户prepare_eval在官方pre.sh的urllib GitHub Zipformer归档下载处遇到自签名证书链错误。该路径不使用HF_ENDPOINT。现由本流程先下载并验证相同发布名称的FP32文件，再将pre.sh的模型下载开关设为0，保留其环境/映射准备。没有关闭TLS证书校验。

- Zipformer-EN：k2-fsa/sherpa-onnx-zipformer-gigaspeech-2023-12-12，固定revision859931db4c4c7bede01b959389dc8a4971a39ec5，encoder/decoder/joiner epoch30 avg1。
- Zipformer-ZH：zrjin/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2，固定revision49ba8f846b47d95596d4dc9751353a51b375ed78，encoder/decoder/joiner epoch20 avg1。
- FireRedVAD：FireRedTeam/FireRedVAD，固定revision7990aaccc6b7aec1e527743bd30201f2c4a03b8c，VAD/model.pth.tar和cmvn.ark。
- 文件清单、大小、SHA256见assets/evaluation_core_models.json，总计约538.4MB。不下载额外INT8副本，评估器仍按原默认FP32选择；既有正确文件复用，未完成文件续传。
- SHA256来自发布者LFS元数据；小型tokens/CMVN文本还核验Git blob ID。GitHub tar归档未做逐字节比较，模型名称、FP32文件名和来源与官方推理文档对应。
- 只修下载和准备方式，TER/SpkSim/DNSMOS-P808/F1及官方评估代码保持原配置。实际模型加载/指标仍需AutoDL执行确认。
- FireRed包初始化中的ASR系统改为惰性导入，VAD入口不再连带导入未使用的ASR/LLM/标点模块。VAD模型和算法未改。准备阶段在云端加载两个Zipformer、FireRedVAD以及说话人/DNSMOS ONNX模型，提前检查缺包/模型损坏；不执行训练或完整指标推理。结果写入evaluation_model_check.json。

参考：[官方Zipformer模型说明](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html)、[FireRedVAD发布者](https://huggingface.co/FireRedTeam/FireRedVAD)。

### Training import fix and GPU memory display

2026-10-11 用户benchmark原始错误为 `ModuleNotFoundError: silero_vad`，发生在导入 `wesep.bin.train` 时，尚未进入模型训练。WeSep根包原来提前导入推理Extractor及其Silero VAD。运行副本现在惰性导出load_model/load_model_local：训练导入不会加载推理VAD，两个公共推理API仍在显式调用时加载。原fork和模型网络/损失保持原实现，修补和SHA记录在runtime_manifest。更新后重新执行prepare可刷新运行副本，完整数据会复用。

- Notebook第2.1节为可执行的显存查询代码，无需Terminal。
- benchmark/正式训练在独立进程中约每秒读取一次nvidia-smi，每10秒在Notebook同一状态框显示`VRAM GPU 0`当前使用/总量、采样峰值、利用率。
- benchmark显示PyTorch分配峰值和nvidia-smi采样峰值；后者可能漏掉瞬间峰值，并包含同卡其他进程。记录在benchmark/nvidia_memory.json和nvidia_memory_training.json。
- 想开独立实时窗口：JupyterLab左上角“＋” → Other/其他 → Terminal/终端，输入`watch -n 1 nvidia-smi`回车。Memory-Usage已用/总量就是显存；Ctrl+C只停止监视。
- Notebook第4.1节读取原始错误，不重跑训练；日志函数优先显示第一条子进程traceback。torchrun入口使用record记录根因，不再只给ChildFailedError包装。
- 本次本地只做包导入/日志/采样的模拟验证，不导入torch或运行模型；显存监控和完整训练仍须AutoDL执行。

查询格式参考：[NVIDIA nvidia-smi文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)。

参考：[OpenSLR 镜像列表](https://www.openslr.org/12)、[镜像所指向的原始压缩包](https://huggingface.co/datasets/k2-fsa/LibriSpeech/tree/main)、[AutoDL 学术加速](https://www.autodl.com/docs/network_turbo/)。


## Inputs downloaded/generated on cloud

| 内容 | 作用 |
|---|---|
| LibriSpeech train-clean-100、dev-clean、test-clean | 固定 OpenSLR 官方 MD5，压缩约 7.07GB |
| 官方 LibriMix 配对和源增益 | 版本 `b898ca375c03e4039cdb3dbe3dd485aad58ec7d1`，已随代码包携带元数据 |
| Libri2Mix-100 16k/min/mix_clean | 云端生成约 17.96GB，train 13,900、dev/test 各 3,000 对 |
| 原始 FLAC | 约 7.35GB，生成后保留；压缩副本可自动删除 |
| ECAPA512 预训练权重 | 约 38.7MB，固定 HF revision/SHA256 |
| REAL-T DEV 与评估模型 | 开启最终四指标评估时准备；不参与训练 |

不用上传本地数据、权重、虚拟环境或旧 outputs。必须保留 `assets/` 中的下载清单、LibriMix 元数据和 ECAPA 最小源码，它们不是音频。

已有云端 data 根目录可填 `EXISTING_DATA_ROOT`，应包含 `simulated/Libri2Mix/wav16k/min/{train-100,dev,test}/{mix_clean,s1,s2}`。脚本重建云端 JSONL 路径，不使用本机 Windows 绝对路径。

## Training parameters

- 16k、3 秒片段、非因果 6 层 BSRNN、ECAPA c512/192 维全局嵌入。
- SI-SDR loss、FP32、Adam、150 轮，指数学习率 0.001→2.5e-5。
- Notebook默认测试 **batch=4 个混音/卡**；collate 后是 **8 个目标样本/卡**。公开模板batch=15，对应30个目标样本。
- ECAPA 冻结参数，BN 统计策略沿用公开模型实现。
- 同说话人注册语音必须来自不同 utterance，训练随机选择，dev/test 固定选择。
- raw 数据入口，不复制生成 tar shards。
- `SHUFFLE_BUFFER=250`（原模板为 2500）、worker=4、prefetch=2，降低解码后音频缓存的 CPU 内存开销。这会改变数据顺序/吞吐，属于明确的运行配置差异。
- `INITIAL_TSE_CHECKPOINT` 留空表示分离器从头训练；可指定同架构、含 `models[0]` 的兼容 WeSep 权重进行新实验。不能直接把 PS4/旧 legacy 模型权重塞入当前结构。

数据 train/dev/test 的 speaker ID 集合不相交，注册语音不会来自其他 split，也不会复用当前目标 utterance。REAL-PS4 没有干净目标，不能直接加入本 SI-SDR 配方。

## GPU and memory

完整数据准备建议数据盘 **100GB 以上，初次空余至少 80GB**。CPU 内存至少 32GB，64GB 优先。

| GPU | 建议试训 batch（混音/卡） |
|---|---:|
| RTX 3080 10/12GB | 1–2 |
| RTX 5080 16GB | 2–4 |
| RTX 4090 24GB | 4 起步，再试 8 / 15 |
| RTX 5090 32GB | 4–8，资源允许再试 15 |
| RTX 6000 Ada 48GB / A100 40GB | 8–15 |
| A100 80GB / RTX PRO 6000 96GB | 15 |

这些是起步建议，不是已测出的最低配置。显存不足时 benchmark 会失败，不会自动改变配方。正式 checkpoint 尚未保存时，可以改 batch 并复用数据；已有正式 checkpoint 后改变训练配置应使用新 WORK_ROOT，可复用旧数据根目录。

## Time estimate

原 trainer 按混音记录计算：

```text
train_steps_per_epoch = 13900 // world_size // mixtures_per_batch_per_gpu
val_steps_per_epoch   = 3000  // world_size // mixtures_per_batch_per_gpu
train_time = train_steps_per_epoch * epochs * measured_train_seconds_per_step
val_time   = val_steps_per_epoch * epochs * measured_val_seconds_per_step
```

双目标展开增加单步计算量，不将训练步数再乘二。按公开配方单卡batch=15计算：每轮926步，150轮138,900次更新；每轮200次验证。Notebook先测batch=4时每轮3475步、验证750步，时间必须按实际batch测速计算。

以下仅为纯训练的条件计算，不是 GPU 测速结果：

| 平均每步 | 150 轮纯训练 |
|---:|---:|
| 0.1 秒 | 3.86 小时 |
| 0.25 秒 | 9.65 小时 |
| 0.5 秒 | 19.29 小时 |
| 1 秒 | 38.58 小时 |

云端先执行 30 步预热、200 步真实训练计时及 30 步验证计时。多卡使用最慢 rank 的平均步时估算；不能按 GPU 数直接保证线性加速。

- `time_estimate.json`：实测训练/验证平均步时、峰值显存、展开后的 batch、预计小时数。
- `live_time_estimate.json`：正式训练后按最近完整 epoch 更新剩余时间，包含 loader 启动成本。
- 估算不包含安装、下载/数据生成、保存权重、平均模型、最终四指标评估；GPU 首次预热、动态注册语音长度也会影响外推。
- `MAX_ESTIMATED_TRAIN_HOURS=0` 默认不设上限；填正数时，超出估算上限会在正式训练前停止。

benchmark 使用临时模型，其优化结果不会成为正式训练的初始权重。

## Resume, averaging and evaluation

上游每 10 轮及最后 10 轮保存 checkpoint。恢复从最近完成保存的 epoch 开始；未保存轮次重跑，不承诺逐 batch/RNG 精确续训。运行副本使用原子 checkpoint 写入，避免中断留下被误识别为完成的权重。

根据保存权重对应的 LibriMix dev 验证 loss，选择最多 10 份最优 checkpoint 平均。多卡验证 loss 做 rank 平均，来源和 epoch 记录在 `averaging_report.json`，不沿用公开 run.sh 的过时固定 epoch 参数。

LibriMix test 的 SI-SDR 是辅助测试；REAL-T DEV 通过固定版本官方 `run_eval.sh` 计算 **TER、SpkSim、DNSMOS-P808、F1**，检查所有 1,991 条记录完整。未下载/使用 EVAL1/EVAL2 做训练选择。

最终结果 `EMB_Training_Results.zip` 包含平均权重、所需模型代码、`infer_emb.py`、配置、耗时/来源记录与指标摘要，不含数据或完整环境。只评价自己重训的模型；与官方发布权重比较时应另外用同一工具重新评估。

## Verification boundary

本地只做静态及数据规则验证，没有跑模型训练或 benchmark。GPU、原 trainer 的实际运行、显存、正式评估都须以 AutoDL 的执行结果为准。没有作者公开的 GPU/耗时资料可用于保证某张卡一定训练成功或精确预测小时数。

原始 fork 源码保留。运行副本在 `WORK_ROOT/runtime`；修补清单包括 ECAPA 严格加载、注册语音独立性、严格续训及原子保存，记录在 runtime_manifest.json。ECAPA/池化代码来自作者指定的 WeSpeaker commit，加载器仅启用 ECAPA，不安装未使用的 Whisper/WavLM 分支。

参考：[WeSep](https://github.com/REAL-TSE/wesep-real-tse)、[公开训练配置](https://github.com/REAL-TSE/wesep-real-tse/blob/main/examples/audio/librimix/confs/tse_bsrnn_spk.yaml)、[LibriMix](https://github.com/JorisCos/LibriMix)、[OpenSLR](https://www.openslr.org/12)、[官方评估](https://github.com/REAL-TSE/REAL-TSE-Challenge)。
