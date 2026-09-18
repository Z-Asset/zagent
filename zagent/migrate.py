"""迁移体检（doctor）—— 数据包 sha1 对账 + 依赖 + 数据口径 + CUDA 探测。

被 CLI（zagent doctor）和顶层 migrate_doctor.py 共用。
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_manifest(pkg: Path) -> tuple[int, int]:
    mf = pkg / "manifest.json"
    if not mf.exists():
        return 0, 0
    data = json.loads(mf.read_text(encoding="utf-8"))
    ok = fail = 0
    for f in data["files"]:
        p = pkg / f["path"]
        if not p.exists():
            print(f"  [FAIL] 缺失 {f['path']}")
            fail += 1
            continue
        if sha1(p) != f["sha1"]:
            print(f"  [FAIL] sha1 不符 {f['path']}")
            fail += 1
        else:
            ok += 1
    print(f"  sha1 对账: {ok} 通过 / {fail} 失败")
    return ok, fail


def check_deps() -> tuple[int, int]:
    ok = fail = 0
    for mod in ["numpy", "pandas", "scipy", "pyarrow", "torch"]:
        try:
            importlib.import_module(mod)
            ok += 1
        except ImportError:
            print(f"  [FAIL] 缺依赖 {mod}")
            fail += 1
    print(f"  依赖: {ok} 可用 / {fail} 缺失")
    return ok, fail


def check_cuda() -> None:
    import torch
    if torch.cuda.is_available():
        print(f"  CUDA: 可用 ({torch.cuda.get_device_name(0)})，可跑全量 GPU 搜索")
    else:
        print("  CUDA: 不可用（CPU 版 torch 或无卡），只能跑小规模验证")


def doctor(pkg: str | Path) -> int:
    """体检一个数据包。返回 0=通过，1=有失败项。

    只做与实验对象无关的通用检查：sha1 对账 + 依赖 + CUDA。数据张量的
    具体形状/口径由训练入口（train_from_xy.py）自己校验，框架不硬编码。
    """
    pkg = Path(pkg).resolve()
    print(f"[zagent-doctor] 体检 {pkg}")
    ok1, fail1 = check_manifest(pkg)
    ok2, fail2 = check_deps()
    check_cuda()
    total_fail = fail1 + fail2
    if total_fail:
        print(f"\n[FAIL] {total_fail} 项未通过")
        return 1
    print("\n[PASS] 全部通过，可运行 zagent")
    return 0
