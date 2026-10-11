"""Generate an ASCII-named UTF-8 notebook. No model imports or training."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
cells = []

def add(kind, source):
    cell = {"cell_type": kind, "metadata": {}, "source": source.strip().splitlines(True), "id": f"emb-{len(cells):02d}"}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)

add("markdown", """
# Offline EMB: AutoDL Training and Time Estimate

主模型：**离线 BSRNN + ECAPA 全局声纹嵌入（EMB）**。只上传整个 `wesep-real-tse/` 代码目录，数据和初始声纹权重在云端下载。

第一段代码选择 `MODE`：`benchmark` 测显存与时间，`train` 正式训练150轮并评估，`resume` 读取原参数恢复已保存的训练。默认单卡4090先测 batch=4；所有操作都已写成代码单元格，可以按顺序运行。

这里重新训练公开 EMB 配方：**BSRNN 分离器随机初始化，ECAPA 使用预训练权重并冻结参数**。不加载 PS4，不使用 Whisper 四损失，不将 REAL-T/REAL-PS4 用于这个 SI-SDR 训练。

作者的训练例程仍标为 WIP。本 Notebook 自动化公开模板，并明确记录数据登记策略、运行修补和云端参数；不能保证与官方发布权重逐位相同或精确复得公开分数。

**本机不训练。** 下面的训练入口只允许 AutoDL Linux CUDA；本地仅创建/检查文件。
""")
add("markdown", """
## 0. Parameters and Hardware

推荐镜像：**PyTorch 2.8.0 + CUDA 12.8、Ubuntu 20.04/22.04 或更新、Python 3.10～3.12**。默认复用 `/root/miniconda3/bin/python` 中已有 PyTorch/torchvision，只补所需依赖。若缺 torchaudio，安装匹配的 2.8.0 wheel（不下载另一套 torch/CUDA 库）。其他训练依赖按本流程的固定版本安装，可能调整镜像中的小包版本。

`ENVIRONMENT_MODE="image"` 为省下载默认值；只有明确要独立固定环境时才设 `"isolated"`，它会重新安装 Python 3.10、PyTorch 2.7.1 和 CUDA 依赖。镜像模式检查失败会停止，不会偷偷回退重装。

完整数据准备及训练建议数据盘 **100GB 以上（初次至少空余 80GB）**、CPU 内存 32GB 以上，64GB 优先。

`BATCH_SIZE` 是混音条数。双说话人 collate 会把它展开成 **2 × BATCH_SIZE 个目标样本**。4090默认从4测试，公开模板为15。

运行模式、batch同步、显存测速、正式启动和断点恢复均由下方代码处理。

| GPU | 可尝试的起步 batch（混音条数/卡） | 说明 |
|---|---:|---|
| RTX 3080 10/12GB | 1–2 | 仅作为试训起点，未实测最低显存 |
| RTX 5080 16GB | 2–4 | 先通过真实训练计时与显存检查 |
| RTX 4090 24GB | 4 起步，再试 8 / 15 | 尚未实测显存和时间 |
| RTX 5090 32GB | 4–8 | 也可尝试 15；不足时降低 batch 并重新计时 |
| RTX 6000 Ada 48GB / A100 40GB | 8–15 | 需要实际检查 |
| A100 80GB / RTX PRO 6000 Blackwell 96GB | 15 | 保守起步选择，仍需检查 |

以上 batch 是工程建议，不是硬件实测保证。更改 batch 会改变更新次数及训练轨迹，Notebook 会重新估算时间。两卡普通 DDP 每卡各存一份模型，显存不能合并。
""")
add("code", '''
from pathlib import Path
import json
import sys

CODE_ROOT = "/root/autodl-tmp/wesep-real-tse"
WORK_ROOT = "/root/autodl-tmp/emb_reproduction"

# ===== 主要操作：只改 MODE 和 BATCH_SIZE，然后按顺序执行 / Run All =====
MODE = "benchmark"               # benchmark=测试显存；train=正式训练；resume=恢复训练
BATCH_SIZE = 4                   # 4090先测4，再改为8/15重测；混音/卡，实际目标数为2倍
# 测试通过：MODE="train"，保留已通过的BATCH_SIZE，再Run All。
# 训练中断：MODE="resume"，原WORK_ROOT不变，再Run All；batch等参数自动读取旧记录。
# MODE="resume" 时这里写的BATCH_SIZE不生效，沿用原训练batch。

assert MODE in ("benchmark", "train", "resume"), "MODE 必须为 benchmark / train / resume"
RUN_TRAINING = MODE in ("train", "resume")  # 自动同步；不用再手动改SETTINGS
ENVIRONMENT_MODE = "image"        # 默认复用镜像；isolated 才重新下载完整 PyTorch/CUDA
IMAGE_PYTHON = "/root/miniconda3/bin/python"
GPU_IDS = "0"                     # 双卡可设 "0,1"
TORCH_CUDA = "cu128"              # 仅 isolated 模式选安装版本；image 使用镜像实际 CUDA
EPOCHS = 150
NUM_WORKERS = 4
SHUFFLE_BUFFER = 250              # 公开模板是 2500；降低 decoded-WAV 缓冲以减少 CPU 内存
SEED = 42
MINIMUM_FREE_GB = 80
HF_ENDPOINT = "https://hf-mirror.com"  # 当前HF代理报SSL错误，使用镜像；可按线路改回官方源
DOWNLOAD_WORKERS = 8
ENABLE_ACADEMIC_ACCELERATION = True  # 在 Notebook 内核加载 AutoDL HF/GitHub 学术代理
LIBRISPEECH_SOURCE = "auto"       # 云端小量测速选源；可指定 huggingface / hf-mirror / cn / eu-elda / eu-trmal / openslr
MINIMUM_DOWNLOAD_MBPS = 0         # 默认不按速度强制停下载；正数为主动选择的门槛
EXISTING_DATA_ROOT = ""           # 已有云端 data 根目录可填；需有 simulated/Libri2Mix/...
INITIAL_TSE_CHECKPOINT = ""       # 默认分离器从头训练；仅支持同结构、含 models[0] 的兼容 WeSep 权重
REMOVE_SOURCE_ARCHIVES = True     # 数据生成成功后删除三个压缩副本，保留原始 FLAC 和生成的 WAV

BENCHMARK_WARMUP = 30
BENCHMARK_STEPS = 200
BENCHMARK_VAL_STEPS = 30
MAX_ESTIMATED_TRAIN_HOURS = 0      # 0 不设上限；正数时超过估算上限会在正式训练前停止
EVALUATE_LIBRI_TEST = True
EVALUATE_REAL_T = True

SETTINGS = {
    "operation": MODE,
    "work_root": WORK_ROOT, "run_training": RUN_TRAINING, "gpu_ids": GPU_IDS,
    "environment_mode": ENVIRONMENT_MODE, "image_python": IMAGE_PYTHON,
    "torch_cuda": TORCH_CUDA, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS, "shuffle_buffer": SHUFFLE_BUFFER, "seed": SEED,
    "minimum_free_gb": MINIMUM_FREE_GB, "hf_endpoint": HF_ENDPOINT,
    "download_workers": DOWNLOAD_WORKERS, "existing_data_root": EXISTING_DATA_ROOT,
    "enable_academic_acceleration": ENABLE_ACADEMIC_ACCELERATION,
    "librispeech_source": LIBRISPEECH_SOURCE, "minimum_download_mbps": MINIMUM_DOWNLOAD_MBPS,
    "initial_tse_checkpoint": INITIAL_TSE_CHECKPOINT, "remove_source_archives": REMOVE_SOURCE_ARCHIVES,
    "benchmark_warmup": BENCHMARK_WARMUP, "benchmark_steps": BENCHMARK_STEPS,
    "benchmark_val_steps": BENCHMARK_VAL_STEPS,
    "max_estimated_train_hours": MAX_ESTIMATED_TRAIN_HOURS,
    "evaluate_libri_test": EVALUATE_LIBRI_TEST, "evaluate_real_t": EVALUATE_REAL_T,
}

checkpoints = sorted(
    (Path(WORK_ROOT)/"exp/emb/models").glob("checkpoint_[0-9]*.pt"),
    key=lambda p: int(p.stem.split("_")[-1]),
)
if MODE == "resume":
    saved_settings = Path(WORK_ROOT)/"pipeline_settings.json"
    if not checkpoints or not saved_settings.is_file():
        raise RuntimeError("没有已保存的正式训练检查点/参数记录；首次训练请选择 MODE='train'。")
    previous = json.loads(saved_settings.read_text(encoding="utf-8"))
    if Path(previous["work_root"]).resolve() != Path(WORK_ROOT).resolve():
        raise RuntimeError("旧参数的工作目录不匹配；检查 WORK_ROOT。")
    SETTINGS = previous
    SETTINGS.update({"operation": "resume", "run_training": True})
    # 恢复原配方，避免新Notebook默认值改变正在恢复的实验。
    BATCH_SIZE = int(SETTINGS["batch_size"])
    EPOCHS = int(SETTINGS["epochs"])
    GPU_IDS = SETTINGS["gpu_ids"]
    NUM_WORKERS = int(SETTINGS["num_workers"])
    RUN_TRAINING = True
    print("Resume checkpoint:", checkpoints[-1])
elif checkpoints:
    raise RuntimeError("此目录已有正式训练检查点：续训选 MODE='resume'；新实验请改 WORK_ROOT 并复用已有数据。")

assert EPOCHS > 0 and BATCH_SIZE > 0 and NUM_WORKERS >= 0
assert BENCHMARK_STEPS > 0 and BENCHMARK_WARMUP >= 0 and BENCHMARK_VAL_STEPS > 0
assert all(x.isdigit() for x in GPU_IDS.split(",")) and len(set(GPU_IDS.split(","))) == len(GPU_IDS.split(","))
print("Mode:", MODE, "| mixture batch/GPU:", BATCH_SIZE, "| formal epochs:", EPOCHS)
print("WORK_ROOT:", WORK_ROOT, "| formal training:", RUN_TRAINING)
''')
add("markdown", "## 1. Locate Code\n\n完整代码包必须包含 `wesep/`、`examples/` 和本 Notebook 旁的所有脚本及 `assets/`。")
add("code", '''
candidates = [Path(CODE_ROOT), Path.cwd(), *Path.cwd().parents, Path("/root/wesep-real-tse")]
repo = next((p.resolve() for p in candidates if (p/"wesep/bin/train.py").is_file()
             and (p/"cloud_upload/autodl_emb/cloud_common.py").is_file()), None)
if repo is None:
    raise FileNotFoundError("请解压完整 wesep-real-tse 代码包并设置 CODE_ROOT")
sys.path.insert(0, str(repo/"cloud_upload/autodl_emb"))
from cloud_common import bootstrap, call_worker, sha256
print("Code:", repo)
''')
add("markdown", """
## 2. Check Image Environment and Install Missing Dependencies

默认先检查镜像 Python/PyTorch/CUDA，复用已有 GPU 库，补匹配 torchaudio/torchvision 和其他训练依赖。pip 使用保护版本约束，避免其他依赖顺带升级/降级 torch 或 NVIDIA 库。随后检查 Linux/glibc、GPU、实际 STFT/LSTM CUDA 反向算子。不直接安装上游混入多个未用大模型的旧 requirements。
Notebook 的下载/安装日志使用同一个状态框更新，只显示最近几行；完整输出保存在 `WORK_ROOT/logs/`。失败时展开错误末尾。终端运行时每 30 秒最多输出一次状态。
ECAPA 依赖仅保留作者指定 WeSpeaker commit 的原始 ECAPA/池化代码，避免为了 EMB 导入未使用的 Whisper/WavLM。具体来源与 SHA256 随代码包保留。
""")
add("code", "cloud_python = bootstrap(SETTINGS)\nprint('Cloud Python:', cloud_python)")
add("markdown", """
### 2.1 View GPU Memory — No Terminal Required

运行下方代码会查询一次当前显存。执行benchmark或正式训练时，日志会自动每10秒更新 `VRAM GPU 0`：当前占用/总显存、采样峰值、GPU利用率；不用自己开终端。GPU内存采样约每秒一次，可能漏掉瞬间峰值，需结合PyTorch峰值与OOM报错。

需要单独实时窗口时：JupyterLab左上角点“＋”打开Launcher，在“Other/其他”中点“Terminal/终端”（也可菜单 File → New → Terminal），输入 `watch -n 1 nvidia-smi` 后回车。查看 Memory-Usage 左边的已用值/右边总量；RTX4090约24564MiB。终端Ctrl+C只停止这个查看命令，不停止另一个Notebook里的训练。训练中同一内核不能同时执行另一单元格，独立Terminal可持续查看。
""")
add("code", '''
from gpu_monitor import gpu_snapshot

# 只查询显存，不执行模型；这时可能尚未开始训练。
for device in gpu_snapshot(SETTINGS["gpu_ids"]):
    print("GPU", device["index"], device["name"])
    print("当前显存:", round(device["used_mib"]/1024, 2), "/",
          round(device["total_mib"]/1024, 2), "GiB")
    print("当前GPU利用率:", device["utilization_percent"], "%")
print("进入测试/训练后，Notebook日志会自动显示 VRAM GPU 占用。")
''')
add("markdown", """
## 3. Prepare Data and ECAPA Initial Weights

云端从 OpenSLR 下载 **train-clean-100、dev-clean、test-clean**（压缩合计约 7.07GB），验证官方 MD5。使用固定版本 JorisCos/LibriMix 的配对与增益生成 **Libri2Mix-100、16k/min/mix_clean**，约 17.96GB；原始 FLAC 约 7.35GB。

每份压缩包先从 OpenSLR 官网/列出的镜像、官方 CN 镜像所指向的 k2-fsa/LibriSpeech 固定版本及 HF 镜像下载最多 1MiB 测速，再选可用较快源。所有源仍须通过原始大小、精确 Range 和官方 MD5 检查。默认不按速度强制停止，避免三个包共享带宽时误停。只有用户主动设置正数门槛时，测速及连续 120 秒下载过慢才会停止并保留 `.parts`。某个包失败会即时显示 SOURCE_FAILED，其他下载完成后保留；重新运行复用已解压语料并续传其余分片。启用学术加速可将 `ENABLE_ACADEMIC_ACCELERATION=True`，而只在终端 source 代理不会自动改变当前 Notebook 内核。

生成 train-100 13,900 对、dev 3,000 对、test 3,000 对。每条混音提供两个训练目标。注册语音来自同说话人的**不同 utterance**，训练随机选、验证/测试固定选；校验 split 的 speaker 集合不相交。

下载约 38.7MB 的预训练 ECAPA512，固定 revision 和 SHA256。分离器默认不加载 TSE 权重，正式训练从随机初始化开始。

该阶段会生成云端绝对路径的 JSONL、cue 资源和 YAML。原始数据不会混进 REAL-T 评估数据；训练 loss 需要干净 s1/s2，不使用缺少干净目标的 REAL-PS4。

这里使用 raw 数据入口，避免额外生成重复的 tar shards。注册语音策略、较小 shuffle 缓冲与原模板的差异会写入说明，不能称为官方权重的逐位复刻。
""")
add("code", '''
call_worker(SETTINGS, "prepare")
print((Path(WORK_ROOT)/"prepared/data_report.json").read_text(encoding="utf-8"))
''')
add("markdown", """
## 4. Batch Selection, VRAM Test and Time Estimate

用真实数据、原模型、SI-SDR、优化器和当前 batch 在云端执行临时训练：排除前 30 步预热，测后 200 步；另外单独测验证步骤。检查有限 loss、分离器梯度、冻结 ECAPA、实际目标 batch 和峰值显存。

**临时模型与权重丢弃，不用于正式初始化。** 显存不足会停止；降低 `BATCH_SIZE` 后重新运行即可（没有正式 checkpoint 时允许调整配置并复用数据）。

原 trainer 的每轮步数公式是 `训练混音条数 // GPU数 // 每卡混音batch`。单卡 batch=15 时，每轮 926 步、150 轮共 138,900 步；验证每轮 200 步。

仅作算术示例，未包含验证/下载/评估：每步 0.1 / 0.25 / 0.5 / 1 秒分别对应约 **3.86 / 9.65 / 19.29 / 38.58 小时纯训练**。这些不是已测得的 GPU 速度。

输出 `time_estimate.json` 使用本次云端实际测速，同时给出验证耗时；正式训练后再按完整 epoch（包含 loader 启动）更新 `live_time_estimate.json`。显卡型号不能直接换算训练天数。

`MODE="benchmark"` 时这里每次重新测速；修改第一段 BATCH_SIZE 再按顺序运行即可。`MODE="train"/"resume"` 时复用相同配置的已通过测速记录，记录不存在或配置变化才重新测。另开 AutoDL 终端运行 `watch -n 1 nvidia-smi` 查看整张卡实际占用；下方峰值仅为 PyTorch 已分配显存。建议实际总占用留约2～4GB余量，短试训通过仍不保证后续所有样本都没有更高峰值。
""")
add("code", '''
def show_batch_report():
    report = json.loads((Path(WORK_ROOT)/"time_estimate.json").read_text(encoding="utf-8"))
    print("Mixture batch/GPU:", report["mixtures_per_batch_per_gpu"])
    print("Target samples/GPU:", report["target_samples_per_batch_per_gpu"])
    for item in report["gpu_reports"]:
        print("GPU:", item["gpu"], "| peak allocated:",
              round(item["train"]["peak_allocated_gib"], 2), "GiB")
    for item in report.get("nvidia_memory_monitor", {}).get("devices", []):
        print("nvidia-smi采样峰值:", round(item["observed_peak_used_mib"]/1024, 2), "/",
              round(item["total_mib"]/1024, 2), "GiB；可能漏掉瞬间峰值")
    print("Estimated train + validation hours:", round(report["train_and_validation_hours"], 2))
    print("Excluded:", report["excluded"])
    return report

benchmark_record = Path(WORK_ROOT)/"time_estimate.json"
prepared_config = Path(WORK_ROOT)/"prepared/train.yaml"
same_config = (
    benchmark_record.is_file()
    and prepared_config.is_file()
    and json.loads(benchmark_record.read_text(encoding="utf-8")).get("config_sha256") == sha256(prepared_config)
)
same_hardware = False
device_record = Path(WORK_ROOT)/"gpu_preflight.json"
if benchmark_record.is_file() and device_record.is_file():
    previous_test = json.loads(benchmark_record.read_text(encoding="utf-8"))
    current_device = json.loads(device_record.read_text(encoding="utf-8"))
    tested_gpus = previous_test.get("gpu_reports", [])
    current_gpus = current_device.get("devices", [])
    same_hardware = (
        len(tested_gpus) == len(current_gpus) == len(SETTINGS["gpu_ids"].split(","))
        and all(old.get("gpu") == now.get("name")
                and old.get("torch") == current_device.get("torch")
                and old.get("cuda") == current_device.get("cuda_runtime")
                for old, now in zip(tested_gpus, current_gpus))
    )
if MODE == "benchmark" or not (same_config and same_hardware):
    print("执行真实训练显存测试，混音batch/GPU =", SETTINGS["batch_size"])
    call_worker(SETTINGS, "benchmark")
else:
    print("复用相同配置已通过的测速记录；以下峰值来自该次测速。")
estimate = show_batch_report()
print("Train steps/epoch:", estimate["train_steps_per_epoch_per_rank"])
print("Measured train seconds/step:", estimate["train_seconds_per_step"])
budget = SETTINGS.get("max_estimated_train_hours", 0)
if MODE != "resume" and budget > 0 and estimate["train_and_validation_hours"] > budget:
    raise RuntimeError("预计训练+验证时间超过 MAX_ESTIMATED_TRAIN_HOURS；未启动正式训练。")
''')
add("markdown", "### 4.1 Read Original Error If Benchmark Failed\n\n测试失败时可单独运行下方代码，不重新训练。ModuleNotFoundError表示缺依赖，CUDA out of memory/OutOfMemoryError才表示显存分配失败；ChildFailedError只是torchrun的外层汇总。")
add("code", '''
from cloud_common import error_context

log = Path(WORK_ROOT)/"logs/benchmark.log"
state_file = log.with_suffix(".process.json")
state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.is_file() else {}
if state.get("status") == "failed" and log.is_file():
    print(error_context(log.read_text(encoding="utf-8", errors="replace")))
else:
    print("当前没有已记录的benchmark失败；无需重新训练来读取日志。")
''')
add("markdown", """
## 5. Prepare Evaluation Before Formal Training

benchmark模式跳过评估模型下载，先完成显存测试。train/resume模式在正式训练前补齐官方评估依赖，已下载的DEV、仓库和模型复用。两个Zipformer的FP32文件和FireRedVAD从发布者HF仓库经HF_ENDPOINT下载，固定revision/大小/SHA256，避免pre.sh的GitHub urllib证书链错误；不关闭TLS验证。随后在云端加载检查评估模型，成功后写evaluation_model_check.json。
""")
add("code", '''
if MODE in ("train", "resume"):
    call_worker(SETTINGS, "prepare_eval")
else:
    print("显存测试模式：跳过评估依赖。改 MODE='train' 后运行会自动准备。")
''')
add("markdown", """
## 6. Train Offline EMB

train模式启动150轮；resume模式从最近保存的正式检查点恢复。benchmark模式在下面代码中跳过训练。沿用16k、3秒、6层BSRNN、全局192维ECAPA声纹、SI-SDR、FP32、Adam和指数学习率0.001→2.5e-5。

每轮用 LibriMix dev 计算验证损失。默认每 10 轮及最后 10 轮保存 checkpoint；中断后从最近**保存完成的 epoch**恢复，未保存轮次重跑。没有逐 batch 游标/RNG 的精确恢复。

Notebook Interrupt 会终止整个训练进程组。更改正式训练配置需新实验目录；可通过 `EXISTING_DATA_ROOT` 复用已生成数据。

每轮会打印 `[ETA]`，并更新 `live_time_estimate.json`。不要将这里的估算与前面 PS4 的假设耗时混用。
""")
add("code", '''
if MODE == "benchmark":
    print("显存测试已完成，未启动150轮。")
    print("下一步：第一段 MODE='train'，BATCH_SIZE保留已通过的值，再Run All。")
else:
    if SETTINGS.get("evaluate_real_t"):
        environment_ready = Path(WORK_ROOT)/"evaluation_environment.json"
        model_ready = Path(WORK_ROOT)/"evaluation_model_check.json"
        if (not environment_ready.is_file() or not model_ready.is_file()
                or json.loads(model_ready.read_text(encoding="utf-8")).get("status") != "passed"):
            raise RuntimeError("请先成功执行上一节评估环境及模型加载检查，再开始正式训练。")
    SETTINGS["run_training"] = True
    RUN_TRAINING = True
    print("Resume training" if MODE == "resume" else "Start formal training",
          "| batch/GPU:", SETTINGS["batch_size"], "| epochs:", SETTINGS["epochs"])
    call_worker(SETTINGS, "train")
    path = Path(WORK_ROOT)/"live_time_estimate.json"
    if path.exists():
        print(path.read_text(encoding="utf-8"))
''')
add("markdown", "### 6.1 Read Training Memory Samples\n\n训练结束或中断后，可运行下面代码查看采样记录。训练正在运行时，显存在前面的日志中自动显示；独立Terminal也可用watch查看。")
add("code", '''
memory_file = Path(WORK_ROOT)/"nvidia_memory_training.json"
if memory_file.is_file():
    memory_report = json.loads(memory_file.read_text(encoding="utf-8"))
    print("Monitor status:", memory_report["status"], "| samples:", memory_report["samples"])
    for device in memory_report["devices"]:
        print("GPU", device["index"], "最后采样:", round(device["used_mib"]/1024, 2),
              "GiB | 采样峰值:", round(device["observed_peak_used_mib"]/1024, 2),
              "GiB | 总量:", round(device["total_mib"]/1024, 2), "GiB")
else:
    print("尚无正式训练采样记录；benchmark记录保存在 benchmark/nvidia_memory.json。")
''')
add("markdown", """
## 7. Average Checkpoints

按已保存 checkpoint 的验证 SI-SDR loss 选择最好的最多 10 份，平均模型浮点权重（整数 buffer 使用选定 checkpoint 值）。生成 `avg_model.pt`。
公开 run.sh 中的固定 epoch 参数不适合自动续训，这里依据实际验证记录选择并保存 `averaging_report.json`。
""")
add("code", "if MODE != 'benchmark':\n    call_worker(SETTINGS, 'average')\nelse:\n    print('显存测试模式：跳过权重平均。')")
add("markdown", """
## 8. Synthetic Test and REAL-T DEV Metrics

LibriMix test 的整段音频 SI-SDR 是辅助测试，不替代老师要求的四指标。
REAL-T DEV 用官方 `run_eval.sh` 输出 **TER↓、SpkSim↑、DNSMOS-P808↑、F1↑**。严格检查 1,991 条结果完整、数值有限。

只评价自己训练的 EMB。若要与官方发布权重对照，应在同一工具配置下另外跑该权重；不把不同 ASR/数据版本的历史数值直接作为自己的实测基线。
""")
add("code", '''
if MODE != "benchmark":
    call_worker(SETTINGS, "evaluate")
    for filename in ["libri_test_summary.json", "real_t_summary.json"]:
        path = Path(WORK_ROOT)/"evaluation"/filename
        if path.exists(): print(path.read_text(encoding="utf-8"))
else:
    print("显存测试模式：跳过正式模型评估。")
''')
add("markdown", "## 9. Package Results\n\n打包平均权重、推理脚本和所需模型代码、配置、来源/修补记录、耗时和指标摘要。数据和安装环境不进入结果包。")
add("code", '''
if MODE != "benchmark":
    call_worker(SETTINGS, "deliver")
    print("Download:", Path(WORK_ROOT)/"EMB_Training_Results.zip")
    print("在 JupyterLab 左侧文件栏找到 ZIP，右键 Download。")
else:
    print("完成：显存/时间测试。下一次第一段设 MODE='train' 并保留测试通过的batch。")
''')
add("markdown", """
## Verification Boundaries and Sources

本地仅检查文件、语法、下载来源、注册语音规则及估时公式，**未在本机训练，也未在 AutoDL 上实际执行本 Notebook**。云端 benchmark 与正式训练成功后才会产生相应完成记录。没有作者公开的硬件/耗时数据可用来保证某张显卡一定跑完。

原始 fork 的 `wesep/` 不修改；运行副本与必要修补位于 `WORK_ROOT/runtime/`。内容包括严格 ECAPA 权重/续训检查、原子 checkpoint 写入、排除当前 utterance 的注册音频、多卡验证 loss 平均以及训练计时。记录见 runtime_manifest.json 和 Notebook 辅助脚本。

[WeSep source](https://github.com/REAL-TSE/wesep-real-tse)
[LibriMix](https://github.com/JorisCos/LibriMix)
[LibriSpeech/OpenSLR](https://www.openslr.org/12)
[WeSpeaker](https://github.com/wenet-e2e/wespeaker)
[Official REAL-T evaluation](https://github.com/REAL-TSE/REAL-TSE-Challenge)
[PyTorch CUDA 12.8/Blackwell](https://pytorch.org/blog/pytorch-2-7/)
""")

notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.12"}}}
path = HERE/"EMB_AutoDL_Full_Pipeline.ipynb"
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(path)
