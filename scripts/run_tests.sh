#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"

# Ubuntu 终端可能已 source /opt/ros/humble/setup.bash。禁止加载全局 ROS pytest
# 插件，确保本项目的非 ROS 测试只使用当前 Conda 环境。
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest "$@"
