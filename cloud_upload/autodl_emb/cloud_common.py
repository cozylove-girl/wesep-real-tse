"""Standard-library Notebook orchestration; model execution happens on Linux CUDA."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from urllib.parse import urlsplit

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def compact_output(value, *, lines=8, characters=1800):
    """Keep the latest log context, removing terminal progress control codes."""
    value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
    recent = []
    for line in value.replace("\r", "\n").splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"((?:DOWNLOAD\s+[^:]+)|(?:VRAM GPU\s+\S+)):", line)
        if match:
            recent = [old for old in recent if not old.startswith(match.group(1)+":")]
        recent.append(line)
    return "\n".join(line[:240] for line in recent[-lines:])[-characters:]


class LiveStatus:
    """One changing Notebook output box; terminal updates at most every 30s."""
    def __init__(self):
        self.handle = None
        self.last_update = 0
        try:
            from IPython import get_ipython
            from IPython.display import display
            shell = get_ipython()
            if shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell":
                self.handle = display({"text/plain": "Starting..."}, raw=True, display_id=True)
        except ImportError:
            pass

    def update(self, message, *, final=False):
        if self.handle is not None:
            self.handle.update({"text/plain": message}, raw=True)
        elif final or time.monotonic()-self.last_update >= 30:
            print(message, flush=True)
            self.last_update = time.monotonic()


def error_context(value):
    """Show the worker's first traceback rather than only torchrun wrappers."""
    value = value.rsplit("\nRUN: ", 1)[-1]
    lines = value.splitlines()
    starts = [i for i, line in enumerate(lines) if "Traceback (most recent call last)" in line]
    if starts:
        first = starts[0]
        end = starts[1] if len(starts) > 1 else len(lines)
        block = "\n".join(lines[first:min(end, first+100)])
        return block[:10000]
    return compact_output(value, lines=30, characters=6000)


def run(command, *, cwd=None, env=None, log):
    command = list(map(str, command))
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    runtime_env = os.environ.copy()
    runtime_env.update({"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
    runtime_env.update(env or {})
    print("RUN:", " ".join(command), flush=True)
    start = time.monotonic()
    status = LiveStatus()
    recent = ""
    with open(log, "a", encoding="utf-8") as output:
        output.write("\nRUN: " + repr(command) + "\n")
        output.flush()
        log_offset = output.tell()
        process = subprocess.Popen(command, cwd=cwd, env=runtime_env, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=os.name != "nt")
        save_json(log.with_suffix(".process.json"), {"pid": process.pid, "status": "running", "command": command})
        try:
            with open(log, encoding="utf-8", errors="replace", newline="") as reader:
                reader.seek(log_offset)
                while process.poll() is None:
                    time.sleep(2)
                    tail = reader.read()
                    if tail:
                        recent = compact_output(recent+"\n"+tail)
                    status.update(f"运行中 {(time.monotonic()-start)/60:.1f} min；完整日志：{log}\n{recent}")
                recent = compact_output(recent+"\n"+reader.read())
                label = "完成" if process.returncode == 0 else f"失败（{process.returncode}）"
                status.update(f"{label}，耗时 {(time.monotonic()-start)/60:.1f} min；完整日志：{log}\n{recent}", final=True)
        except BaseException:
            if process.poll() is None:
                if os.name == "nt":
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        process.kill()
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            save_json(log.with_suffix(".process.json"), {"pid": process.pid, "status": "interrupted"})
            status.update(f"已中断；完整日志：{log}", final=True)
            raise
    save_json(log.with_suffix(".process.json"), {"pid": process.pid, "returncode": process.returncode,
                                               "status": "success" if process.returncode == 0 else "failed"})
    if process.returncode:
        with open(log, encoding="utf-8", errors="replace") as stream:
            stream.seek(log_offset)
            original_error = error_context(stream.read())
        print("原始错误：\n"+original_error, flush=True)
        raise RuntimeError(f"步骤失败（{process.returncode}），检查 {log}")


def http_session(url):
    import requests
    session = requests.Session()
    # Keep HF mirror/CDN requests direct throughout redirects; the user's
    # academic proxy reports SSLError, while the direct mirror is working.
    session.trust_env = urlsplit(url).hostname != "hf-mirror.com"
    return session


def configure_hub_transport():
    if urlsplit(os.environ.get("HF_ENDPOINT", "https://huggingface.co")).hostname == "hf-mirror.com":
        from huggingface_hub import configure_http_backend
        configure_http_backend(backend_factory=lambda: http_session("https://hf-mirror.com"))


def download(url, target, *, checksum=None, algorithm="sha256", size=None):
    import requests
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    def digest(path):
        h = hashlib.new(algorithm)
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1024*1024), b""):
                h.update(b)
        return h.hexdigest()
    if target.exists() and (size is None or target.stat().st_size == size) and (checksum is None or digest(target) == checksum):
        return target
    part = target.with_suffix(target.suffix + ".part")
    for attempt in range(6):
        offset = part.stat().st_size if part.exists() else 0
        if size is not None and offset > size:
            part.unlink()
            offset = 0
        try:
            if size is not None and offset == size:
                if checksum is not None and digest(part) != checksum:
                    part.unlink()
                    continue
                part.replace(target)
                return target
            with http_session(url) as session, session.get(url, headers={"Range": f"bytes={offset}-"} if offset else {}, stream=True,
                                                         timeout=(30, 120)) as response:
                response.raise_for_status()
                resume = offset > 0 and response.status_code == 206
                if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("Unexpected resume response")
                last = time.monotonic()
                with open(part, "ab" if resume else "wb") as output:
                    for chunk in response.iter_content(1024*1024):
                        if chunk:
                            output.write(chunk)
                        if time.monotonic()-last > 20:
                            print(f"下载 {target.name}: {output.tell()/10**6:.1f} MB", flush=True)
                            last = time.monotonic()
            if size is not None and part.stat().st_size != size:
                raise RuntimeError(f"Size mismatch: {target}")
            if checksum is not None and digest(part) != checksum:
                part.unlink()
                raise RuntimeError(f"Checksum mismatch: {target}")
            part.replace(target)
            return target
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 15))


def runtime_selection(settings):
    path = Path(settings["work_root"])/"environment_selection.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"python": str(Path(settings["work_root"])/"env/bin/python"),
            "prefix": str(Path(settings["work_root"])/"env")}


def runtime_python(settings):
    return Path(runtime_selection(settings)["python"])


def enable_academic_acceleration():
    script = Path("/etc/network_turbo")
    if not script.is_file():
        raise RuntimeError("找不到 AutoDL /etc/network_turbo；关闭 ENABLE_ACADEMIC_ACCELERATION 或检查平台服务。")
    result = subprocess.run(["bash", "-c", "source /etc/network_turbo >/dev/null && env"],
                            capture_output=True, text=True, check=True)
    allowed = {"http_proxy", "https_proxy", "all_proxy", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"}
    count = 0
    for line in result.stdout.splitlines():
        name, separator, value = line.partition("=")
        if separator and name in allowed:
            os.environ[name] = value
            count += 1
    if count == 0:
        raise RuntimeError("AutoDL 学术加速未返回代理设置；请检查平台服务。")
    print("已在 Notebook 内核启用 AutoDL 学术加速（HF/GitHub），后续下载子进程会继承。", flush=True)


def environment(settings):
    work = Path(settings["work_root"])
    for directory in ("cache/pip", "cache/conda", "cache/huggingface", "cache/modelscope", "tmp"):
        (work/directory).mkdir(parents=True, exist_ok=True)
    result = {"CUDA_VISIBLE_DEVICES": settings["gpu_ids"], "PYTHONUTF8": "1", "MPLBACKEND": "Agg",
              "PIP_PROGRESS_BAR": "off", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TQDM_DISABLE": "1",
              "PIP_CACHE_DIR": str(work/"cache/pip"), "CONDA_PKGS_DIRS": str(work/"cache/conda"),
              "HF_HOME": str(work/"cache/huggingface"), "HF_ENDPOINT": settings["hf_endpoint"],
              "HF_HUB_DISABLE_XET": "1", "MODELSCOPE_CACHE": str(work/"cache/modelscope"),
              "TMPDIR": str(work/"tmp"), "EMB_DATA_ROOT": str(work/"data"), "OMP_NUM_THREADS": "4"}
    result["LIBRISPEECH_SOURCE"] = settings.get("librispeech_source", "auto")
    result["LIBRISPEECH_MIN_MBPS"] = str(settings.get("minimum_download_mbps", 0))
    if settings.get("enable_academic_acceleration", False):
        # The academic proxy is for HF/GitHub. Keep ordinary package sources
        # direct so turning it on does not disrupt pip/conda installation.
        direct = ".pytorch.org,.aliyun.com,.tsinghua.edu.cn,.pythonhosted.org,pypi.org,.anaconda.com,.anaconda.org,hf-mirror.com"
        previous = os.environ.get("no_proxy", os.environ.get("NO_PROXY", ""))
        result["NO_PROXY"] = result["no_proxy"] = ",".join(filter(None, (previous, direct)))
    selection = runtime_selection(settings)
    prefix = Path(selection["prefix"])
    libraries = sorted(prefix.glob("lib/python*/site-packages/nvidia/*/lib"))
    libraries += sorted(prefix.glob("lib/python*/site-packages/torch/lib"))
    result["PATH"] = str(Path(selection["python"]).parent)+os.pathsep+os.environ.get("PATH", "")
    constraints = work/"environment/protected_gpu_constraints.txt"
    if constraints.exists():
        result["PIP_CONSTRAINT"] = str(constraints)
    if libraries:
        result["LD_LIBRARY_PATH"] = os.pathsep.join(map(str, libraries))
        if os.environ.get("LD_LIBRARY_PATH"):
            result["LD_LIBRARY_PATH"] += os.pathsep+os.environ["LD_LIBRARY_PATH"]
    return result


def probe_runtime(python, *, check_audio=False):
    """Inspect cloud image in a subprocess; never called by local static checks."""
    script = '''
import importlib.metadata as md
import json, sys, torch
packages = {}
for item in md.distributions():
    name = item.metadata.get("Name", "").lower().replace("_", "-")
    if name in ("torch", "torchaudio", "torchvision") or name.startswith("nvidia-"):
        packages[name] = item.version
print(json.dumps({"python": sys.executable, "prefix": sys.prefix,
                 "python_version": list(sys.version_info[:3]), "torch": torch.__version__,
                 "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                 "packages": packages}))
'''
    if check_audio:
        script = "import torchaudio, torchvision\n"+script
    result = subprocess.run([str(python), "-c", script], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("无法使用所选镜像 Python；不会自动下载另一套 PyTorch。\n"+result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])


def gpu_companions(torch_version):
    versions = {"2.7.0": ("2.7.0", "0.22.0"), "2.7.1": ("2.7.1", "0.22.1"),
                "2.8.0": ("2.8.0", "0.23.0")}
    base = torch_version.split("+")[0]
    if base not in versions:
        raise RuntimeError(f"未验证的镜像 PyTorch {torch_version}；请选 2.7.x/2.8.0，或显式使用 isolated 模式。")
    audio, vision = versions[base]
    return {"torchaudio": audio, "torchvision": vision}


def protect_gpu_packages(work, probe):
    constraints = work/"environment/protected_gpu_constraints.txt"
    constraints.parent.mkdir(parents=True, exist_ok=True)
    constraints.write_text("".join(f"{name}=={version}\n" for name, version in sorted(probe["packages"].items())),
                           encoding="utf-8")


def bootstrap(settings):
    if platform.system() != "Linux":
        raise RuntimeError("本 Notebook 只在 AutoDL Linux CUDA 上执行；不在本机 Windows 训练。")
    name, version = platform.libc_ver()
    if name != "glibc" or tuple(map(int, version.split(".")[:2])) < (2, 28):
        raise RuntimeError("请选择 Ubuntu 20.04/22.04 或更新系统（glibc >= 2.28）。")
    if settings.get("enable_academic_acceleration", False):
        enable_academic_acceleration()
    work = Path(settings["work_root"])
    if not work.is_absolute() or not str(work).isascii():
        raise ValueError("WORK_ROOT 必须是英文绝对路径")
    work.mkdir(parents=True, exist_ok=True)
    conda = shutil.which("conda") or os.environ.get("CONDA_EXE")
    if not conda:
        raise RuntimeError("找不到 conda；请选择含 Miniconda 的 AutoDL PyTorch 镜像。")
    minimum = settings["minimum_free_gb"] if not (work/"prepared/data_report.json").exists() else 15
    if shutil.disk_usage(work).free < minimum*10**9:
        raise RuntimeError(f"数据盘至少需要空余 {minimum} GB；请扩容或选用已有完整数据。")
    run(["nvidia-smi"], log=work/"logs/environment.log")
    mode = settings.get("environment_mode", "image")
    if mode == "image":
        python = Path(settings.get("image_python") or sys.executable)
        probe = probe_runtime(python)
        if not (3, 10) <= tuple(probe["python_version"][:2]) <= (3, 12):
            raise RuntimeError("当前依赖固定支持 Python 3.10～3.12；请使用相应镜像。")
        if not probe["cuda_available"] or not probe["cuda"]:
            raise RuntimeError("镜像 PyTorch 的 CUDA 不可用；不会自动重装 PyTorch。")
        companions = gpu_companions(probe["torch"])
        if not (Path(probe["prefix"])/"conda-meta").is_dir():
            raise RuntimeError("镜像模式需要 Conda Python 前缀，供官方评估激活；请设置 IMAGE_PYTHON。")
        save_json(work/"environment_selection.json", {**probe, "mode": mode})
        # Protect existing torch/CUDA libraries, while allowing small missing
        # companion wheels to be added. --no-deps prevents a second torch download.
        protected = {**probe, "packages": {k: v for k, v in probe["packages"].items()
                                           if k not in companions}}
        protect_gpu_packages(work, protected)
        index = "cu"+probe["cuda"].replace(".", "")
        for name, expected in companions.items():
            installed = probe["packages"].get(name, "")
            if installed.split("+")[0] != expected or ("+" in probe["torch"] and installed != expected+"+"+index):
                run([python, "-m", "pip", "install", "--no-deps", name+"=="+expected,
                     "--index-url", "https://download.pytorch.org/whl/"+index],
                    env=environment(settings), log=work/"logs/environment.log")
        probe = probe_runtime(python, check_audio=True)
        protect_gpu_packages(work, probe)
        print(f"复用镜像 Python {python}，PyTorch {probe['torch']}，CUDA {probe['cuda']}；不重装 torch/CUDA 库。", flush=True)
    elif mode == "isolated":
        python = work/"env/bin/python"
        save_json(work/"environment_selection.json", {"python": str(python), "prefix": str(work/"env"), "mode": mode})
        if not python.exists():
            run([conda, "create", "-y", "-p", work/"env", "--override-channels", "-c", "conda-forge", "python=3.10", "pip"],
                env=environment(settings), log=work/"logs/environment.log")
        if settings["torch_cuda"] not in ("cu126", "cu128"):
            raise ValueError("TORCH_CUDA 必须是 cu126/cu128。")
        # Explicit fallback only; the default image mode never enters this branch.
        install_env = environment(settings)
        install_env.pop("PIP_CONSTRAINT", None)
        run([python, "-m", "pip", "install", "torch==2.7.1", "torchaudio==2.7.1", "torchvision==0.22.1",
             "--index-url", "https://download.pytorch.org/whl/"+settings["torch_cuda"]],
            env=install_env, log=work/"logs/environment.log")
        probe = probe_runtime(python, check_audio=True)
        protect_gpu_packages(work, probe)
    else:
        raise ValueError("ENVIRONMENT_MODE 只能是 image 或 isolated")
    save_json(work/"environment_selection.json", {**probe, "mode": mode})
    req = PACKAGE/"requirements-cloud.txt"
    signature = {"requirements_sha256": sha256(req), "mode": mode, "python": probe["python"],
                 "torch": probe["torch"], "cuda": probe["cuda"]}
    marker = work/"environment/cloud_environment.json"
    if not marker.exists() or json.loads(marker.read_text(encoding="utf-8")) != signature:
        run([python, "-m", "pip", "install", "-r", req], env=environment(settings), log=work/"logs/environment.log")
        run([python, "-m", "pip", "check"], env=environment(settings), log=work/"logs/environment.log")
        save_json(marker, signature)
    run([python, PACKAGE/"gpu_preflight.py", "--report", work/"gpu_preflight.json"],
        env=environment(settings), log=work/"logs/environment.log")
    return python


def call_worker(settings, stage):
    if stage in ("train", "average", "evaluate", "deliver") and not settings["run_training"]:
        print(f"仅预检：跳过 {stage}。要完整训练请设置 RUN_TRAINING=True。")
        return
    if settings.get("enable_academic_acceleration", False):
        # A user may enable this after bootstrap and rerun only this stage.
        # Load it here too, so editing SETTINGS actually affects this download.
        enable_academic_acceleration()
    work = Path(settings["work_root"])
    save_json(work/"pipeline_settings.json", settings)
    run([runtime_python(settings), PACKAGE/"pipeline_worker.py", "--settings", work/"pipeline_settings.json", "--stage", stage],
        cwd=REPO, env=environment(settings), log=work/"logs"/(stage+".log"))
