from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .camera import RealSenseCamera
from .detector import YoloDetector
from .geometry import localize_detection
from .memory import TargetMemory
from .models import LocalizedObject, SelectionResult
from .selector import TargetSelector, build_selector
from .settings import get_setting, has_setting
from .visualizer import draw_depth_scene, draw_live_view, next_depth_color_mode, next_view_mode


QUIT_COMMANDS = {"/quit", "/exit", "quit", "exit", "退出", "结束"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RealSense D435 + 预训练 YOLO + 自然语言目标表示方向",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=get_setting("YOLO_MODEL", "yolo26m.pt"), help="YOLO 权重路径")
    parser.add_argument("--confidence", type=float, default=0.15, help="送入跟踪器的最低检测置信度")
    parser.add_argument("--candidate-confidence", type=float, default=0.30, help="自然语言候选的最低平滑置信度")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 推理图像尺寸")
    parser.add_argument("--device", default="auto", help="YOLO 设备，如 auto、cpu 或 0")
    parser.add_argument("--tracker", default="botsort.yaml", help="Ultralytics 跟踪器配置")
    parser.add_argument("--selector", choices=("auto", "local", "deepseek"), default="auto")
    parser.add_argument("--deepseek-model", default=get_setting("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument(
        "--deepseek-base-url",
        default=get_setting("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    parser.add_argument("--color-width", type=int, default=640)
    parser.add_argument("--color-height", type=int, default=480)
    parser.add_argument("--depth-width", type=int, default=848)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial", default="", help="多相机时指定 RealSense 序列号")
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0, help="顺时针旋转彩色/深度画面")
    parser.add_argument("--min-depth", type=float, default=0.2, help="有效最近深度（米）")
    parser.add_argument("--max-depth", type=float, default=8.0, help="有效最远深度（米）")
    parser.add_argument("--depth-roi", type=float, default=0.45, help="检测框中央取深度区域比例")
    parser.add_argument("--max-depth-mad", type=float, default=0.15, help="允许的框内深度 MAD 上限（米）")
    parser.add_argument("--memory-seconds", type=float, default=4.0, help="普通目标漏检后的记忆时长")
    parser.add_argument("--selected-memory-seconds", type=float, default=15.0, help="选中目标的记忆时长")
    parser.add_argument("--position-smoothing", type=float, default=0.35, help="三维位置 EMA 的新观测权重")
    parser.add_argument("--min-observations", type=int, default=2, help="进入候选列表前至少连续观测次数")
    parser.add_argument("--no-depth-filter", action="store_true", help="关闭空间/时间/空洞深度滤波")
    parser.add_argument(
        "--view-mode",
        choices=("split", "color", "depth"),
        default="split",
        help="窗口初始视图：彩色/深度并排、仅彩色或仅深度",
    )
    parser.add_argument(
        "--depth-color-mode",
        choices=("official", "metric"),
        default="official",
        help="深度着色：RealSense 官方直方图均衡或固定米制色标",
    )
    parser.add_argument("--headless", action="store_true", help="不创建 OpenCV 窗口")
    parser.add_argument("--max-frames", type=int, default=0, help="处理指定帧数后退出；0 为持续运行")
    parser.add_argument("--command", default="", help="启动后自动提交一次自然语言命令，便于脚本化测试")
    parser.add_argument("--command-wait-frames", type=int, default=15, help="脚本命令等待候选稳定的最长帧数")
    parser.add_argument("--check-device", action="store_true", help="只检查设备和彩色/深度帧，不加载 YOLO")
    parser.add_argument(
        "--diagnostic-dir",
        default="",
        help="设备检查时保存彩色图、深度伪彩色图和并排预览到此目录",
    )
    return parser


def _make_camera(args: argparse.Namespace) -> RealSenseCamera:
    return RealSenseCamera(
        color_width=args.color_width,
        color_height=args.color_height,
        depth_width=args.depth_width,
        depth_height=args.depth_height,
        fps=args.fps,
        use_filters=not args.no_depth_filter,
        create_depth_color=(not args.headless and not args.check_device) or bool(args.diagnostic_dir),
        serial=args.serial,
        rotate=args.rotate,
    )


def check_device(args: argparse.Namespace) -> int:
    devices = RealSenseCamera.list_devices()
    if not devices:
        print("未发现 RealSense 设备。", file=sys.stderr)
        return 2
    print(f"发现 {len(devices)} 台 RealSense：")
    for index, info in enumerate(devices):
        print(
            f"  [{index}] {info.get('name', 'Unknown')} | 序列号 {info.get('serial', '?')} | "
            f"固件 {info.get('firmware', '?')} | USB {info.get('usb_type', '?')}"
        )
    camera = _make_camera(args)
    try:
        camera.start(warmup_frames=20)
        frames = [camera.get_frame() for _ in range(5)]
        frame = frames[-1]
        valid = frame.depth_m[np.isfinite(frame.depth_m) & (frame.depth_m > 0)]
        median = float(np.median(valid)) if valid.size else float("nan")
        height, width = frame.depth_m.shape
        center = frame.depth_m[height // 3 : height * 2 // 3, width // 3 : width * 2 // 3]
        center_valid = center[np.isfinite(center) & (center > 0)]
        center_median = float(np.median(center_valid)) if center_valid.size else float("nan")
        coverage = 100.0 * valid.size / frame.depth_m.size
        print(f"彩色帧：{frame.color_bgr.shape[1]}x{frame.color_bgr.shape[0]}")
        print(f"对齐深度帧：{frame.depth_m.shape[1]}x{frame.depth_m.shape[0]}")
        print(f"有效深度像素：{valid.size}（{coverage:.1f}%），全图中位深度：{median:.3f} m")
        print(f"画面中央有效深度：{center_valid.size}，中央中位深度：{center_median:.3f} m")
        if args.diagnostic_dir:
            output_dir = Path(args.diagnostic_dir).expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            depth_color = draw_depth_scene(
                frame.depth_m,
                [],
                {},
                depth_color_bgr=frame.depth_color_bgr,
                depth_color_mode=args.depth_color_mode,
                min_depth_m=args.min_depth,
                max_depth_m=args.max_depth,
            )
            preview = np.hstack((frame.color_bgr, depth_color))
            cv2.imwrite(str(output_dir / "color.png"), frame.color_bgr)
            cv2.imwrite(str(output_dir / "depth_colormap.png"), depth_color)
            cv2.imwrite(str(output_dir / "preview.png"), preview)
            print(f"诊断图已保存：{output_dir}")
        print("设备采集检查通过。")
        return 0
    finally:
        camera.stop()


def _console_input(commands: queue.Queue[str], stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            command = input("\n请输入目标（例：前往右边的椅子；/quit 退出）：").strip()
        except (EOFError, KeyboardInterrupt):
            commands.put("/quit")
            return
        if command:
            commands.put(command)
        if command.casefold() in QUIT_COMMANDS:
            return


def _print_candidates(objects: list[LocalizedObject]) -> None:
    print("\n当前可定位目标：")
    for obj in sorted(objects, key=lambda item: item.signed_angle_deg):
        state = "当前" if obj.is_visible else f"记忆{obj.last_seen_age_s:.1f}s"
        print(
            f"  [#{obj.id}] {obj.class_name:<16} {obj.horizontal_position:<6} "
            f"距离 {obj.distance_m:.2f} m，方向角 {obj.direction_deg:.1f}°，{state}"
        )


def _print_selection(
    command: str,
    result: SelectionResult,
    objects: list[LocalizedObject],
) -> int | None:
    by_id = {obj.id: obj for obj in objects}
    print(f"\n命令：{command}")
    if result.status == "matched" and result.target_id in by_id:
        target = by_id[result.target_id]
        side = "右" if target.signed_angle_deg > 0 else "左" if target.signed_angle_deg < 0 else "正前"
        print(f"匹配来源：{result.source}（{result.reason}）")
        print(f"目标：{target.class_name} #{target.id}")
        if not target.is_visible:
            print(f"状态：目标暂时不可见，使用 {target.last_seen_age_s:.1f} 秒前的记忆位置")
        elif target.depth_age_s > 0.25:
            print(f"状态：视觉可见，深度沿用 {target.depth_age_s:.1f} 秒前的有效值")
        print(f"水平距离：{target.distance_m:.2f} m")
        print(f"相机坐标：X={target.x_m:+.3f} m，Y={target.y_m:+.3f} m，Z={target.z_m:.3f} m")
        if side == "正前":
            print(f"相对正前方：{abs(target.signed_angle_deg):.1f}°")
        else:
            print(f"相对正前方：向{side}偏转 {abs(target.signed_angle_deg):.1f}°")
        print(f"0～180°方向角：{target.direction_deg:.1f}°（0°左，90°正前，180°右）")
        return target.id
    if result.status == "ambiguous":
        labels = [f"{by_id[item].class_name} #{item}" for item in result.candidate_ids if item in by_id]
        print(f"目标有歧义：{result.reason}")
        print("候选：" + ("、".join(labels) if labels else "无"))
        print("请补充左/右、最近/最远或画面编号。")
    else:
        print(f"未找到目标：{result.reason}")
    return None


def run(args: argparse.Namespace) -> int:
    selector: TargetSelector = build_selector(
        args.selector,
        deepseek_model=args.deepseek_model,
        deepseek_base_url=args.deepseek_base_url,
    )
    selector_name = type(selector).__name__
    print(f"自然语言选择器：{selector_name}")
    if args.selector == "auto" and not has_setting("DEEPSEEK_API_KEY"):
        print("未检测到 DEEPSEEK_API_KEY，使用本地规则解析。")
    print(f"加载预训练 YOLO：{args.model}")
    detector = YoloDetector(
        args.model,
        confidence=args.confidence,
        image_size=args.imgsz,
        device=args.device,
        tracker=args.tracker,
    )
    print(f"YOLO 推理设备：{detector.runtime_device}")

    commands: queue.Queue[str] = queue.Queue()
    scripted_command = args.command.strip()
    stop_event = threading.Event()
    input_thread: threading.Thread | None = None
    if args.max_frames == 0 or not args.headless:
        input_thread = threading.Thread(target=_console_input, args=(commands, stop_event), daemon=True)
        input_thread.start()

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="target-selector")
    pending: Future[SelectionResult] | None = None
    pending_context: tuple[str, list[LocalizedObject]] | None = None
    selected_id: int | None = None
    camera = _make_camera(args)
    frame_count = 0
    detection_count = 0
    localization_count = 0
    started_at = time.perf_counter()
    latest_objects: list[LocalizedObject] = []
    memory = TargetMemory(
        memory_seconds=args.memory_seconds,
        selected_memory_seconds=args.selected_memory_seconds,
        smoothing=args.position_smoothing,
        min_observations=args.min_observations,
        candidate_confidence=args.candidate_confidence,
    )
    previous_selected_visible: bool | None = None
    view_mode = args.view_mode
    depth_color_mode = args.depth_color_mode

    try:
        camera.start()
        print(
            "RealSense 已启动。请在终端直接输入自然语言目标；"
            "窗口按 D 切换视图，按 H 切换官方/固定深度着色，按 Q 或 ESC 退出。"
        )
        if not args.headless:
            cv2.namedWindow("RealSense Natural Language Target", cv2.WINDOW_NORMAL)
        while True:
            frame = camera.get_frame()
            detections = detector.detect(frame.color_bgr)
            detection_count += len(detections)
            frame_objects: list[LocalizedObject] = []
            for detection in detections:
                target = localize_detection(
                    detection,
                    frame.depth_m,
                    frame.intrinsics,
                    deproject=camera.deproject,
                    central_ratio=args.depth_roi,
                    min_depth_m=args.min_depth,
                    max_depth_m=args.max_depth,
                    max_depth_mad_m=args.max_depth_mad,
                )
                if target is not None:
                    frame_objects.append(target)
            localization_count += len(frame_objects)
            all_objects = memory.update(frame_objects, detections=detections)
            display_detections = memory.canonicalize_detections(detections)
            latest_objects = memory.objects(confirmed_only=True)
            localized_by_id = {obj.id: obj for obj in all_objects}

            if scripted_command:
                frame_limit = args.command_wait_frames
                if args.max_frames > 0:
                    frame_limit = min(frame_limit, args.max_frames)
                if latest_objects or frame_count + 1 >= frame_limit:
                    commands.put(scripted_command)
                    scripted_command = ""

            if pending is not None and pending.done():
                assert pending_context is not None
                command, snapshot = pending_context
                try:
                    result = pending.result()
                except Exception as exc:
                    print(f"目标选择失败：{type(exc).__name__}: {exc}", file=sys.stderr)
                else:
                    new_selected_id = _print_selection(command, result, snapshot)
                    if new_selected_id is not None:
                        selected_id = new_selected_id
                        memory.select(selected_id)
                pending = None
                pending_context = None

            if pending is None:
                try:
                    command = commands.get_nowait()
                except queue.Empty:
                    command = ""
                if command:
                    if command.casefold() in QUIT_COMMANDS:
                        break
                    snapshot = list(latest_objects)
                    _print_candidates(snapshot)
                    pending_context = (command, snapshot)
                    pending = executor.submit(selector.select, command, snapshot)

            if not args.headless:
                canvas = draw_live_view(
                    frame.color_bgr,
                    frame.depth_m,
                    display_detections,
                    localized_by_id,
                    depth_color_bgr=frame.depth_color_bgr,
                    depth_color_mode=depth_color_mode,
                    view_mode=view_mode,
                    selected_id=selected_id,
                    min_depth_m=args.min_depth,
                    max_depth_m=args.max_depth,
                )
                cv2.imshow("RealSense Natural Language Target", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (ord("d"), ord("D")):
                    view_mode = next_view_mode(view_mode)
                    print(f"窗口视图已切换为：{view_mode}")
                if key in (ord("h"), ord("H")):
                    depth_color_mode = next_depth_color_mode(depth_color_mode)
                    print(f"深度着色已切换为：{depth_color_mode}")

            selected = memory.get(selected_id)
            selected_visible = selected.is_visible if selected is not None else None
            if previous_selected_visible is True and selected_visible is False:
                print(
                    f"目标 #{selected_id} 暂时不可见，继续使用最近位置记忆 "
                    f"{args.selected_memory_seconds:.1f} 秒。"
                )
            elif previous_selected_visible is False and selected_visible is True:
                suffix = "（已重新绑定跟踪 ID）" if memory.reacquired_on_last_update else ""
                print(f"目标 #{selected_id} 已重新捕获{suffix}。")
            elif previous_selected_visible is False and selected_visible is None:
                print(f"目标 #{selected_id} 超出记忆时限，锁定已释放。")
                selected_id = None
            previous_selected_visible = selected_visible

            frame_count += 1
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break

        # 脚本化命令即使在最后一帧才提交，也应给出确定结果。
        if pending is not None and pending_context is not None:
            command, snapshot = pending_context
            try:
                result = pending.result(timeout=20.0)
            except Exception as exc:
                print(f"目标选择失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            else:
                new_selected_id = _print_selection(command, result, snapshot)
                if new_selected_id is not None:
                    selected_id = new_selected_id
                    memory.select(selected_id)
        if args.max_frames > 0:
            elapsed = max(1e-6, time.perf_counter() - started_at)
            print(
                f"端到端烟雾测试：{frame_count} 帧，{detection_count} 次检测，"
                f"{localization_count} 次有效三维定位，平均 {frame_count / elapsed:.2f} FPS。"
            )
        return 0
    finally:
        stop_event.set()
        camera.stop()
        executor.shutdown(wait=False, cancel_futures=True)
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.confidence <= 1.0:
        raise SystemExit("--confidence 必须位于 (0, 1]")
    if not args.confidence <= args.candidate_confidence <= 1.0:
        raise SystemExit("--candidate-confidence 必须位于 [--confidence, 1]")
    if args.min_depth <= 0 or args.max_depth <= args.min_depth:
        raise SystemExit("深度范围参数无效")
    if args.max_depth_mad <= 0:
        raise SystemExit("--max-depth-mad 必须大于 0")
    if args.memory_seconds <= 0 or args.selected_memory_seconds < args.memory_seconds:
        raise SystemExit("目标记忆时长参数无效")
    if not 0.0 < args.position_smoothing <= 1.0:
        raise SystemExit("--position-smoothing 必须位于 (0, 1]")
    if args.min_observations < 1:
        raise SystemExit("--min-observations 必须至少为 1")
    if args.command_wait_frames < 1:
        raise SystemExit("--command-wait-frames 必须至少为 1")
    try:
        if args.check_device:
            return check_device(args)
        return run(args)
    except RuntimeError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
