#!/usr/bin/env bash
# Yee 一键安装：在 Mia dsv41 方案的 head 宿主机上运行（其它机器自动走 pip 模式）。
#   bash install.sh --check      只检查
#   bash install.sh --restart    安装到每台机器 + 重启 + 健康检查与冒烟，失败自动回退
#   bash install.sh --uninstall --restart
# 全部参数：bash install.sh --help
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$(command -v python3 || true)"
if [[ -z "$PY" ]]; then
  echo "[yee-install] 错误：找不到 python3。DGX OS / Ubuntu 上用 'sudo apt-get install -y python3' 装好后再运行。" >&2
  exit 1
fi
exec "$PY" "$HERE/yee_install.py" "$@"
