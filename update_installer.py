from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from gta_helper.updater import UpdateError, apply_update


def main() -> int:
    if len(sys.argv) != 3:
        print("사용법: update_installer.py <업데이트 ZIP> <프로그램 폴더>")
        return 2
    archive = Path(sys.argv[1]).resolve()
    root = Path(sys.argv[2]).resolve()
    try:
        apply_update(archive, root)
    except UpdateError as exc:
        print(f"[업데이트 실패] {exc}")
        return 1
    print("업데이트를 적용했습니다. 프로그램을 다시 시작합니다.")
    if os.name == "nt":
        os.startfile(str(root / "run.bat"))
    else:
        subprocess.Popen([str(root / "run.bat")], cwd=str(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
