$ErrorActionPreference = "Stop"

conda env create -f environment.yml

# 某些 Windows Conda 配置会让 pip 看见全局用户 site-packages。
# 使用 -s 再核对一遍，确保所有传递依赖实际安装在 realsense 环境内。
conda run -n realsense python -s -m pip install --upgrade -r requirements.txt

Write-Host "环境已创建。运行：conda activate realsense"

