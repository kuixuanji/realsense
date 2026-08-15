# RealSense D435 自然语言目标定位

这是一个面向 Ubuntu 的独立 Python 原型：D435 提供彩色图和对齐深度，YOLO 检测并持续跟踪常见物体，自然语言选择器确定用户指向的候选，程序输出目标相对相机正前方的距离和 `0°～180°` 方向角。

```text
D435 彩色/深度 → YOLO26m + BoT-SORT → 前景深度簇 → 三维坐标
                              ↓                 ↓
                         跨帧目标记忆 ← 自然语言选择目标 ID
```

这里的“方向”是当前相机坐标系内的观测方向，不是带地图、避障、里程计和底盘控制的完整导航路线。机器人型号确定后，再把稳定目标点接入机器人坐标变换和导航系统。

## 本版关键改进

- Ubuntu 22.04 安装脚本，使用阿里云镜像安装 Python 3.11 依赖和 CUDA 12.1 PyTorch；
- 默认模型从最小的 `yolo26n.pt` 升级为精度更高的 `yolo26m.pt`；
- 使用 Ultralytics BoT-SORT 的运动模型和相机运动补偿维持跨帧 ID；
- 普通目标漏检后记忆 4 秒，已选目标记忆 15 秒；
- 选中目标被跟踪器重新分配 ID 时，按类别、框位置和三维位置重新绑定，并持续把新 ID 归一为原 ID；
- 视觉仍可见但深度暂时无效时沿用最近有效深度，不再立即“失焦”；
- 三维位置和置信度采用 EMA 平滑，单帧抖动不会直接改变方向；
- 深度框内先选择最近可信前景簇，再用中位数和 MAD 排除飞点；
- 候选至少连续出现两次才提供给语言选择器，降低单帧误检；
- 默认并排显示彩色检测画面和实时对齐深度图，按 `D` 可切换视图；
- 深度图默认使用 librealsense 官方直方图均衡着色，并包含动态米制图例；
- D435 自检会报告 USB 类型、深度覆盖率、中央深度，并可保存诊断图。

## Ubuntu 快速开始

硬件验证基线是 Ubuntu 22.04、NVIDIA RTX 4060 Laptop、CUDA Toolkit 12.1 和 Intel RealSense D435。进入项目目录后运行：

```bash
bash scripts/setup_ubuntu.sh
conda activate realsense
```

脚本先创建/更新 `realsense` 环境，再从阿里云安装 `torch 2.4.1+cu121` 和其余 PyPI 依赖。安装完成后会打印 PyTorch、CUDA 可用状态和 GPU 名称。

检查相机并保存当前画面：

```bash
python main.py --check-device --diagnostic-dir /tmp/realsense_diagnostic
```

正常结果应包含：

- 设备名和序列号；
- `USB 3.x`；
- 彩色图与对齐深度图尺寸；
- 有效深度像素比例和中央中位深度；
- `/tmp/realsense_diagnostic/color.png`、`depth_colormap.png`、`preview.png`。

如果设备能被 `lsusb` 看到但程序无权限访问，安装 librealsense 的 udev rules 后重新插拔相机，并确认当前用户拥有 `video`/`plugdev` 设备权限。不要使用 USB 2.0 集线器。

## 运行

```bash
conda activate realsense
python main.py
```

首次使用 `yolo26m.pt` 时 Ultralytics 会下载官方预训练权重。下载完成后文件位于项目目录。窗口默认左侧显示彩色检测结果、右侧显示逐像素对齐的实时深度图。彩色画面显示目标 `#ID`、类别、置信度、距离和方向；虚线 `MEMORY` 框表示目标当前漏检但仍在记忆期。

深度图默认直接使用 librealsense 的官方 `rs.colorizer` 和直方图均衡，与 RealSense Viewer 的着色链一致。它会展开当前场景中密集的深度区间，因此室内近距离表面的层次明显多于 `0.2～8m` 线性压缩。右侧 `AUTO` 图例根据当前帧颜色映射给出 5 个真实米制刻度；黑色表示无效或超出设置范围，中央十字旁的 `center` 是中心像素实时深度。

按 `H` 可在两种深度着色之间切换：

```text
official    RealSense SDK 直方图均衡，细节优先，图例量程随画面变化
metric      固定 --min-depth～--max-depth 量程，跨帧颜色可直接比较
```

按 `D` 依次切换窗口布局：

```text
彩色/深度并排 → 仅彩色 → 仅深度 → 彩色/深度并排
```

也可以在启动时指定初始视图：

```bash
python main.py --view-mode split
python main.py --view-mode color
python main.py --view-mode depth
python main.py --depth-color-mode metric
```

终端可输入：

```text
去椅子那里
前往右边的椅子
选择最近的瓶子
去画面中央的物体
去编号 3
去第二把椅子
```

按 `D` 切换视图，按 `H` 切换深度着色，按 `Q` / `Esc` 或输入 `/quit` 退出。脚本化烟雾测试示例：

```bash
python main.py --headless --max-frames 100 --selector local --command "去最近的物体"
```

## 模型、性能与精度

默认使用官方 COCO 预训练 `yolo26m.pt`。它仍只认识 COCO 的 80 个类别；“充电桩”“特定工位”“某型号零件”等领域对象需要自定义数据训练或另接开放词汇模型。

```bash
# 更快、精度稍低
python main.py --model yolo26s.pt

# 最小模型，仅用于低性能设备
python main.py --model yolo26n.pt

# 明确指定第一块 GPU
python main.py --device 0
```

`--confidence` 默认从旧版的 `0.45` 降至 `0.15`，让 BoT-SORT 能利用低置信框救回轨迹；只有平滑置信度达到 `--candidate-confidence 0.30` 且至少连续出现两次的目标才进入自然语言候选。不要只靠提高检测阈值处理误检，否则会显著增加漏检和目标中断。

## 目标记忆

默认行为：

```text
首次单帧出现        → 暂不进入自然语言候选
连续观测至少 2 次   → 成为可选目标
普通目标暂时漏检    → 保留 4 秒
已选目标暂时漏检    → 保留 15 秒并显示最后方向
再次观测            → 恢复实时位置并继续使用原 ID
跟踪器产生新 ID     → 建立新 ID → 原 ID 别名，不创建重复 memory
旧/新 ID 同帧出现   → 合并为原 ID，只显示并更新一个目标
视觉可见、深度空洞  → 更新框，暂用最近有效三维坐标
超过记忆时限        → 释放目标锁定
```

按场景调整：

```bash
python main.py \
  --memory-seconds 5 \
  --selected-memory-seconds 20 \
  --position-smoothing 0.35 \
  --min-observations 2
```

相机快速移动时保留默认 `botsort.yaml`；固定相机且只追求速度时可尝试 `bytetrack.yaml`。

## DeepSeek

复制配置模板：

```bash
cp local_config.example.py local_config.py
```

在不会提交到版本控制的 `local_config.py` 中填写：

```python
DEEPSEEK_API_KEY = "你的 DeepSeek API Key"
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
```

运行：

```bash
python main.py --selector deepseek
```

`auto` 在配置了密钥时使用 DeepSeek，否则使用本地规则；DeepSeek API、网络或 JSON 校验失败时也会回退本地规则。大模型只接收结构化候选并返回 ID，距离与角度始终由本地计算。

DeepSeek 调用使用兼容 Responses API 的 Python SDK，并在代码中将客户端显式命名为 `DeepSeekClient(api_key=..., base_url=...)`。实际请求固定发往 `DEEPSEEK_BASE_URL`，并调用 `client.responses.create(...)`。项目不再提供 OpenAI 服务、模型或 API Key 配置。

也可通过环境变量配置，环境变量优先于 `local_config.py`：

```bash
export DEEPSEEK_API_KEY="你的密钥"
export DEEPSEEK_MODEL="deepseek-v4-flash"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
python main.py --selector auto
```

DeepSeek 客户端始终忽略 `HTTP_PROXY`、`HTTPS_PROXY` 和 `ALL_PROXY` 等系统代理变量并直接连接，避免错误的系统代理导致初始化或请求失败。

不配置任何 API Key 时，本地中英文规则仍可完成类别、编号、左右、中央、远近和序号选择。

## 角度与距离约定

RealSense 相机坐标：`X` 向画面右侧，`Y` 向下，`Z` 向前。

```text
相对正前方偏转 = atan2(X, Z)
0～180°方向角 = 90° + 相对正前方偏转
```

- `0°`：左侧；
- `90°`：相机正前方；
- `180°`：右侧。

单台 D435 只能输出当前视场内的方向，无法直接覆盖完整前方半圆。程序报告的“水平距离”为 `hypot(X, Z)`，不包含垂直分量 `Y`。

## 常用参数

```bash
python main.py --help
python main.py --serial 你的相机序列号
python main.py --rotate 90
python main.py --confidence 0.25 --imgsz 768
python main.py --confidence 0.15 --candidate-confidence 0.35
python main.py --min-depth 0.25 --max-depth 6.0 --max-depth-mad 0.10
python main.py --view-mode depth
python main.py --depth-color-mode official
python main.py --no-depth-filter
```

默认彩色流为 `640×480@30`，深度流为 `848×480@30`，随后将深度对齐到彩色流。

## 测试

```bash
conda activate realsense
bash scripts/run_tests.sh
```

测试覆盖方向角、前景深度簇、MAD 过滤、反投影、本地中文解析、API 结构校验、IoU 跟踪以及目标记忆/重捕获。
测试脚本会禁止自动加载全局 ROS pytest 插件，避免已经 `source /opt/ros/humble/setup.bash` 的终端把 Python 3.10 ROS 包混入 Python 3.11 Conda 测试。

## Windows 旧环境

旧 PowerShell 安装脚本仍保留在 `scripts/setup_env.ps1`，但当前维护和实机验证以 Ubuntu 为准。Windows 不应运行 Ubuntu CUDA wheel 安装脚本。

详细模块和调用关系见根目录的 `代码结构说明.md`。
