# RealSense D435 自然语言目标定位

这是一个可在不同操作系统上手动创建 Conda 环境的独立 Python 原型：D435 提供彩色图和对齐深度，YOLO 检测并持续跟踪常见物体，自然语言选择器确定用户指向的候选，程序输出目标相对相机正前方的距离和 `0°～180°` 方向角。

```text
D435 彩色/深度 → YOLO26m + BoT-SORT → 前景深度簇 → 三维坐标
                              ↓                 ↓
                         跨帧目标记忆 ← 自然语言选择目标 ID
```

这里的“方向”是当前相机坐标系内的观测方向，不是带地图、避障、里程计和底盘控制的完整导航路线。机器人型号确定后，再把稳定目标点接入机器人坐标变换和导航系统。

## 本版关键改进

- 默认模型从最小的 `yolo26n.pt` 升级为精度更高的 `yolo26m.pt`；
- 使用 Ultralytics BoT-SORT 的运动模型和相机运动补偿维持跨帧 ID；
- 普通目标漏检后记忆 4 秒，已选目标记忆 15 秒；
- 普通目标被短时完全遮挡后，按类别、框位置和三维位置把新跟踪 ID 绑定回原 ID；
- 选中目标被跟踪器重新分配 ID 时，按类别、框位置和三维位置重新绑定，并持续把新 ID 归一为原 ID；
- 视觉仍可见但深度暂时无效时沿用最近有效深度，不再立即“失焦”；
- 三维位置和置信度采用 EMA 平滑，单帧抖动不会直接改变方向；
- 深度框内先选择最近可信前景簇，再用中位数和 MAD 排除飞点；
- 候选至少连续出现两次才提供给语言选择器，降低单帧误检；
- 默认并排显示彩色检测画面和实时对齐深度图，按 `D` 可切换视图；
- 深度图默认使用 librealsense 官方直方图均衡着色，并包含动态米制图例；
- D435 自检会报告 USB 类型、深度覆盖率、中央深度，并可保存诊断图。

## 环境准备

建议使用 Python 3.11。项目的实机验证基线是 Ubuntu 22.04、NVIDIA RTX 4060 Laptop、CUDA 12.1 和 Intel RealSense D435；其他系统需先确保 D435 驱动和 `pyrealsense2` 可用。

### 1. 创建 Conda 环境

```bash
conda create -n realsense python=3.11 pip -y
conda activate realsense
```

如果普通终端无法执行 `conda activate`，先运行 `conda init` 并重新打开终端。后续命令都在已激活的 `realsense` 环境中执行。

### 2. 安装 PyTorch

CPU 和 CUDA 需要不同的 PyTorch wheel。根据运行设备选择一种 pip 命令，不要重复安装。

CPU 版（Windows / Linux，无 NVIDIA GPU 或只做功能验证）：

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
```

macOS 版：

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1
```

CUDA 12.1 版（Windows / Linux 的 NVIDIA GPU）：

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

CUDA 版需要已正确安装兼容的 NVIDIA 驱动。安装后可检查实际运行设备：

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 3. 安装其余依赖

```bash
conda install -c conda-forge "numpy>=1.26,<3" "opencv>=4.10,<5" "pyrealsense2>=2.55" "ultralytics>=8.3" "lap>=0.5.12" "openai>=3,<4" "pydantic>=2.8" "pytest>=8.3"
```

上述命令使用 conda-forge 安装运行和测试依赖。项目的直接依赖如下：

| 依赖 | 用途 |
| --- | --- |
| `torch==2.4.1` | CPU 或 CUDA 模型推理 |
| `torchvision==0.19.1` | PyTorch 视觉算子 |
| `numpy>=1.26,<3` | 深度数组与稳健统计 |
| `opencv>=4.10,<5` | 提供 `cv2`，用于图像处理、窗口、标注和诊断图 |
| `pyrealsense2>=2.55` | RealSense D435 采集、对齐和深度处理 |
| `ultralytics>=8.3` | YOLO 检测与 BoT-SORT 跟踪 |
| `lap>=0.5.12` | 跟踪器线性分配 |
| `openai>=3,<4` | DeepSeek Responses API 兼容客户端 |
| `pydantic>=2.8` | 大模型输出结构校验 |
| `pytest>=8.3` | 单元测试 |

### 4. 检查相机

检查相机并保存当前画面：

```bash
python main.py --check-device --diagnostic-dir diagnostics
```

正常结果应包含：

- 设备名和序列号；
- `USB 3.x`；
- 彩色图与对齐深度图尺寸；
- 有效深度像素比例和中央中位深度；
- `diagnostics/color.png`、`depth_colormap.png`、`preview.png`。

Ubuntu 上如果设备能被 `lsusb` 看到但程序无权限访问，安装 librealsense 的 udev rules 后重新插拔相机，并确认当前用户拥有 `video`/`plugdev` 设备权限。不要使用 USB 2.0 集线器。

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
短时遮挡后产生新 ID → 位置基本不变时绑定回原 ID
旧/新 ID 同帧出现   → 合并为原 ID，只显示并更新一个目标
视觉可见、深度空洞  → 更新框，暂用最近有效三维坐标
超过记忆时限        → 释放目标锁定
用户可见 ID          → 固定在 1～1000，释放后循环复用
```

BoT-SORT 的内部 ID 可以持续增长，记忆层会将其映射为 `1～1000` 的用户可见
ID。编号到达上限时不会清空全部记忆：当前帧目标、尚未过期的记忆和已选目标会保留，
新目标优先复用已经释放的编号。编号池确实占满时，先回收最旧的不可见且未选中目标；
当前可见目标和已选目标绝不会被覆盖。

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

默认直接运行 `python main.py` 时使用完全离线的本地规则。`auto` 在配置了密钥时使用 DeepSeek，否则使用本地规则；DeepSeek API、网络或 JSON 校验失败时也会回退本地规则。大模型只接收结构化候选并返回 ID，距离与角度始终由本地计算。

DeepSeek 调用使用兼容 Responses API 的 Python SDK，并在代码中将客户端显式命名为 `DeepSeekClient(api_key=..., base_url=...)`。实际请求固定发往 `DEEPSEEK_BASE_URL`，并调用 `client.responses.create(...)`。项目不再提供 OpenAI 服务、模型或 API Key 配置。

项目从 `local_config.py` 读取上述配置。

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
python -m pytest
```

测试覆盖方向角、前景深度簇、MAD 过滤、反投影、本地中文解析、API 结构校验、IoU 跟踪以及目标记忆/重捕获。Linux 终端如果已经 `source /opt/ros/humble/setup.bash`，可改用 `bash scripts/run_tests.sh`，防止全局 ROS pytest 插件混入当前 Conda 环境。

详细模块和调用关系见根目录的 `代码结构说明.md`。
