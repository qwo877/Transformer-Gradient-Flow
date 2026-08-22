#實驗出處紀錄(provenance)
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch


def _git(*args, cwd=None):
    #跑一個 git 指令 失敗時回傳 None
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def collect(base_dir=None):
    #蒐集出處資訊 base_dir 是 repo 根目錄(預設為本檔所在目錄)
    base = Path(base_dir or Path(__file__).parent)
    commit = _git("rev-parse", "HEAD", cwd=base)
    # git diff --quiet 有改動時回傳 1 沒改動回傳 0
    dirty, dirty_files = None, None
    if commit is not None:
        r = subprocess.run(["git", "diff", "--quiet"], cwd=base,
                           capture_output=True)
        dirty = (r.returncode != 0)
        if dirty:
            out = _git("diff", "--name-only", cwd=base)
            dirty_files = out.splitlines() if out else []

    info = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "git_commit": commit,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=base),
        "git_dirty": dirty,
        "git_dirty_files": dirty_files,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None),
        "platform": platform.platform(),
    }
    return info


def write(results_dir, extra=None, info=None):
    info = dict(info) if info else collect()
    if extra:
        info.update(extra)
    path = Path(results_dir) / "provenance.json"
    if path.exists():
        try:
            old = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            old = None
        if old:
            history = old.pop("previous_runs", [])
            info["previous_runs"] = history + [old]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)

    if info["git_commit"] is None:
        print("WARNING: 不在 git repo 裡,無法記錄 commit —— "
              "這組結果的程式碼版本無法回溯", flush=True)
    elif info["git_dirty"]:
        print("WARNING: working tree 有未提交的改動,"
              f"git_commit={info['git_commit'][:8]} 不足以完整描述這次實驗",
              flush=True)
    return info
