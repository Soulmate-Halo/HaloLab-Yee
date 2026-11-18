#!/usr/bin/env bash
# Yee 一键安装（4.3）：
#   Mia 方案：在 Mia dsv41 方案的 head 宿主机上运行
#     bash install.sh --check      只检查
#     bash install.sh --restart    安装到每台机器 + 重启 + 健康检查与冒烟，失败自动回退
#     bash install.sh --uninstall --restart
#   其它任何部署（本机进程、docker 容器、多台机器、K8s）：自动走 generic 模式
#     bash install.sh --check              只诊断，不装
#     bash install.sh                      装进发现的 Python 引擎，然后打印诊断报告
#     bash install.sh --hosts h1,h2        多台机器一起看、一起装
#     bash install.sh --restart            装完重启 docker 容器，再诊断一次
#     bash install.sh --uninstall          卸载
#   只要装进当前 Python（老路径）：bash install.sh --pip
# 全部参数：bash install.sh --help
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$(command -v python3 || true)"
if [[ -z "$PY" ]]; then
  echo "[yee-install] 错误：找不到 python3。DGX OS / Ubuntu 上用 'sudo apt-get install -y python3' 装好后再运行。" >&2
  exit 1
fi
exec "$PY" "$HERE/yee_install.py" "$@"
