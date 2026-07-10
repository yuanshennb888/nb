"""
摄像头实时图片靶检测 + 裁剪 + 100类分类 (v4)
==============================================
- YOLOv8n 单类检测 (target) → 找到靶标位置
- YOLOv8n-cls 100类分类 → 识别中心图片类别（CIFAR-100）
- Top-5 白名单门控: 取 Top-5 预测，命中指定目标类即确定类别
- 支持类别过滤模式

部署目标: Jetson Orin Nano TensorRT FP16
预计性能: 30-60 FPS (含摄像头)

按键:
  q / ESC  - 退出
  s        - 保存当前裁剪图
  +/-      - 调整置信度阈值
  r        - 切换是否记录视频
  f        - 切换类别过滤模式
  g        - 切换 Top-5 白名单门控模式

用法:
  python camera_detect.py [--cam 0] [--conf 0.25] [--targets cloud,hamster,rocket,train]
"""

import cv2
from ultralytics import YOLO
import numpy as np
import time
import os
import argparse
from datetime import datetime

# ===================== 默认配置 =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
V3_DIR = os.path.join(os.path.dirname(BASE_DIR), "ring_detect_v3")

# 检测模型（复用 v3 训练的单类 target 检测模型）
DEFAULT_MODEL = os.path.join(V3_DIR, "runs", "target_detect_n", "weights", "best.pt")
if not os.path.exists(DEFAULT_MODEL):
    DEFAULT_MODEL = os.path.join(V3_DIR, "deploy", "best.pt")
# 如果 v3 没有，尝试 v4 自身
if not os.path.exists(DEFAULT_MODEL):
    DEFAULT_MODEL = os.path.join(BASE_DIR, "deploy", "best.pt")

# 分类模型（v4 训练的 100 类模型）
DEFAULT_CLS_MODEL = os.path.join(BASE_DIR, "runs", "classify",
                                  "target_cls_100", "weights", "best.pt")
if not os.path.exists(DEFAULT_CLS_MODEL):
    DEFAULT_CLS_MODEL = os.path.join(BASE_DIR, "deploy_cls", "cls_best.pt")

CAMERA_ID = 0
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMGSZ = 640
SAVE_DIR = os.path.join(BASE_DIR, "captures")

# ===================== Top-5 白名单门控配置 =====================
# 指定的目标类别（只关注这4个类，其余视为 unknown）
# 可通过 --targets 命令行参数覆盖
TARGET_CLASSES = ['cloud', 'hamster', 'rocket', 'train']

# 门控模式（默认开启）
#  - True:  Top-5 中命中 TARGET_CLASSES → 确定为该类别，否则标记 unknown
#  - False: 直接使用 Top-1 结果（传统模式）
GATING_MODE = True

# 类别过滤（None=显示所有100类, 列表=仅显示指定类）
# 例如: ACTIVE_CLASSES = ['apple', 'bear', 'bicycle', 'bus', 'cat', 'dog']
ACTIVE_CLASSES = None

# 显示颜色
COLOR_TARGET = (0, 255, 0)
COLOR_TARGET_LOW = (0, 200, 255)
COLOR_BG_BAR = (40, 40, 40)
COLOR_MATCHED = (0, 255, 200)       # 白名单命中 (青绿)
COLOR_UNKNOWN = (80, 80, 255)        # 未命中/unknown (橙红)

os.makedirs(SAVE_DIR, exist_ok=True)


# ===================== 工具函数 =====================

def draw_detection(frame, box, conf, top_predictions=None, gating_result=None):
    """绘制检测框和 Top-3 分类结果

    参数:
        frame:           原始画面
        box:             (x1, y1, x2, y2)
        conf:            检测置信度
        top_predictions: [(class_name, prob), ...] 最多3个
        gating_result:   (final_class, final_conf, is_matched) 或 None
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box

    # 根据门控结果选择颜色
    if gating_result is not None:
        _, _, is_matched = gating_result
        if is_matched:
            color = COLOR_MATCHED     # 青绿 = 命中
        else:
            color = COLOR_UNKNOWN     # 橙红 = unknown
        thickness = 3
    elif conf >= 0.80:
        color = COLOR_TARGET
        thickness = 3
    elif conf >= 0.60:
        color = (0, 255, 255)
        thickness = 2
    else:
        color = COLOR_TARGET_LOW
        thickness = 2

    # 绘制 bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # 标签（显示最终判定类别）
    if gating_result is not None:
        final_cls, final_conf, is_matched = gating_result
        if is_matched:
            label = f"[GATED] {final_cls} {final_conf:.0%}"
        else:
            label = f"[UNKNOWN] target {conf:.0%}"
    elif top_predictions and len(top_predictions) > 0:
        cls_name, cls_prob = top_predictions[0]
        label = f"{cls_name} {conf:.0%}"
    else:
        label = f"target {conf:.0%}"

    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # 中心十字
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cross_sz = 15
    cv2.line(frame, (cx - cross_sz, cy), (cx + cross_sz, cy), color, 2)
    cv2.line(frame, (cx, cy - cross_sz), (cx, cy + cross_sz), color, 2)

    # 侧边栏：Top-3 预测详情（含门控标记）
    if top_predictions and len(top_predictions) > 1:
        sidebar_x = x2 + 8
        sidebar_w = 220
        if sidebar_x + sidebar_w > w:
            sidebar_x = x1 - sidebar_w - 8

        # 半透明背景
        n_lines = min(len(top_predictions), 3) + (1 if gating_result else 0)
        bar_h = 16 * n_lines + 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (sidebar_x, y1),
                      (sidebar_x + sidebar_w, y1 + bar_h),
                      (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

        ty = y1 + 16

        # 门控状态行
        if gating_result is not None:
            final_cls, final_conf, is_matched = gating_result
            if is_matched:
                gate_text = f"[GATED] -> {final_cls} ({final_conf:.0%})"
                gate_color = (0, 255, 200)
            else:
                gate_text = "[GATED] -> UNKNOWN"
                gate_color = (80, 80, 255)
            cv2.putText(frame, gate_text, (sidebar_x + 4, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, gate_color, 1)
            ty += 18

        cv2.putText(frame, "Top-5 Predictions:", (sidebar_x + 4, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        ty += 16
        for rank, (name, prob) in enumerate(top_predictions[:3]):
            bar_w = int((sidebar_w - 85) * prob)
            # 命中目标类的用绿色，否则灰色
            if gating_result and gating_result[2]:
                bar_color = (0, 200, 0) if rank == 0 else (100, 100, 100)
            else:
                bar_color = (0, 200, 0) if rank == 0 else (100, 100, 100)
            cv2.rectangle(frame, (sidebar_x + 76, ty - 9),
                          (sidebar_x + 76 + bar_w, ty + 5), bar_color, -1)
            text = f"#{rank + 1} {name}"
            cv2.putText(frame, text, (sidebar_x + 4, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            cv2.putText(frame, f"{prob:.1%}", (sidebar_x + 78 + bar_w + 2, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
            ty += 16

    return frame


def crop_target(frame, box, margin_ratio=0.05):
    """从画面中裁剪靶标区域（带小边距）"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box

    bw, bh = x2 - x1, y2 - y1
    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)

    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)

    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def put_info_bar(frame, text_lines, fps=None):
    """在画面上方绘制半透明信息条"""
    h, w = frame.shape[:2]
    bar_h = 24 * len(text_lines) + 12

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), COLOR_BG_BAR, -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    y = 20
    for line in text_lines:
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 22

    if fps is not None:
        fps_text = f"FPS: {fps:.1f}"
        (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(frame, fps_text, (w - tw - 10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    return frame


def resize_to_display(img, max_size=400):
    """按比例缩放图片用于显示"""
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h))


def apply_top5_gating(top5_predictions, target_set):
    """Top-5 白名单门控 — 在 Top-5 预测中查找目标类别

    核心逻辑:
      1. 遍历 Top-5 预测，筛选出属于 target_set 的类别
      2. 如果命中 → 取置信度最高的作为最终类别
      3. 如果未命中 → 标记为 "unknown"

    参数:
        top5_predictions: [(class_name, confidence), ...] 最多5个
        target_set:       set[str], 目标类别集合

    返回:
        (final_class, final_conf, is_matched, display_preds)
        - final_class:  str, 最终判定类别 (或 "unknown")
        - final_conf:   float, 对应置信度
        - is_matched:   bool, 是否命中白名单
        - display_preds: list, 用于显示的预测列表
    """
    if not top5_predictions:
        return ("unknown", 0.0, False, [])

    # 在 Top-5 中筛选目标类别
    matched = [(name, conf) for name, conf in top5_predictions
               if name in target_set]

    if matched:
        # 命中：取置信度最高的目标类（已按 Top-5 顺序排好）
        best_name, best_conf = matched[0]
        # 显示：命中的排在前面，后面跟其他 top 预测
        display = matched[:3]
        others = [(n, c) for n, c in top5_predictions if n not in target_set]
        display += others[:max(0, 3 - len(display))]
        return (best_name, best_conf, True, display)
    else:
        # 未命中：标记 unknown，但仍显示 Top-3 供参考
        return ("unknown", 0.0, False, top5_predictions[:3])


def get_top_predictions(cls_results, cls_names, k=3):
    """从分类结果中提取 Top-K 预测

    参数:
        cls_results: YOLO classification results
        cls_names:   list[str], 类别名列表
        k:           int, 返回前 K 个

    返回:
        [(class_name, probability), ...]
    """
    if cls_results[0].probs is None:
        return []

    probs = cls_results[0].probs
    # 获取 top-k 索引
    if hasattr(probs, 'topk'):
        topk_indices = probs.topk(k)
        topk_confs = probs.topkconf(k)
    else:
        # fallback: 手动排序
        data = probs.data.cpu().numpy() if hasattr(probs.data, 'cpu') else probs.data
        indices = np.argsort(data)[::-1][:k]
        topk_indices = indices
        topk_confs = data[indices]

    predictions = []
    for idx, conf in zip(topk_indices, topk_confs):
        idx_int = int(idx)
        if idx_int < len(cls_names):
            predictions.append((cls_names[idx_int], float(conf)))
    return predictions


# ===================== 主函数 =====================

def main():
    parser = argparse.ArgumentParser(description="图片靶实时检测 + 100类分类")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="检测模型路径")
    parser.add_argument("--cls-model", default=DEFAULT_CLS_MODEL, help="100类分类模型路径")
    parser.add_argument("--cam", type=int, default=CAMERA_ID, help="摄像头编号")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="检测置信度阈值")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=IMGSZ, help="检测推理尺寸")
    parser.add_argument("--cls-imgsz", type=int, default=224, help="分类推理尺寸")
    parser.add_argument("--augment", action="store_true", help="启用测试时增强")
    parser.add_argument("--filter", type=str, default=None,
                        help="逗号分隔的类别过滤，如 'apple,bear,bicycle'")
    parser.add_argument("--targets", type=str, default=None,
                        help="逗号分隔的 Top-5 白名单目标类，如 'cloud,hamster,rocket,train'")
    parser.add_argument("--no-gating", action="store_true",
                        help="关闭 Top-5 白名单门控，使用传统 Top-1 模式")
    parser.add_argument("--topk", type=int, default=3, help="显示 Top-K 预测")
    args = parser.parse_args()

    # 处理目标类别（Top-5 白名单）
    target_classes = TARGET_CLASSES
    if args.targets:
        target_classes = [c.strip() for c in args.targets.split(',')]
    target_set = set(target_classes)

    # 门控模式
    gating_mode = GATING_MODE and not args.no_gating

    # 处理类别过滤
    active_classes = ACTIVE_CLASSES
    if args.filter:
        active_classes = [c.strip() for c in args.filter.split(',')]
        print(f"  类别过滤: {active_classes}")

    # ===================== 加载模型 =====================
    print("=" * 62)
    print("  图片靶实时检测 + 100类分类 (v4)")
    print("=" * 62)
    print(f"  检测模型: {args.model}")
    print(f"  分类模型: {args.cls_model}")
    print(f"  摄像头: {args.cam}")
    print(f"  检测阈值: {args.conf}  |  IoU: {args.iou}")
    print(f"  检测尺寸: {args.imgsz}  |  分类尺寸: {args.cls_imgsz}")

    # 检测模型
    if not os.path.exists(args.model):
        print(f"\n[ERROR] 检测模型不存在: {args.model}")
        print("  请先在 v3 中训练检测模型: cd ../ring_detect_v3 && python train.py")
        return

    model = YOLO(args.model)
    print(f"  检测模型加载成功")

    # 分类模型
    cls_model = None
    cls_names = []
    if os.path.exists(args.cls_model):
        cls_model = YOLO(args.cls_model)
        cls_names = list(cls_model.names.values()) if cls_model.names else []
        print(f"  分类模型加载成功 ({len(cls_names)} 类)")
        if len(cls_names) <= 10:
            print(f"    类别: {', '.join(cls_names)}")
        else:
            print(f"    类别: {', '.join(cls_names[:5])} ... ({len(cls_names)} total)")
    else:
        print(f"  [WARN] 分类模型不存在: {args.cls_model}，仅做检测")
        print(f"  [INFO] 请先运行: python generate_cls_dataset.py && python train_cls.py")

    # ===================== 打开摄像头 =====================
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 (ID={args.cam})")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  摄像头分辨率: {actual_w}×{actual_h}")
    if active_classes:
        print(f"  活跃类别: {active_classes}")
    print(f"  Top-5 白名单门控: {'ON' if gating_mode else 'OFF'}  "
          f"目标类: {target_classes}")
    if gating_mode:
        print(f"    逻辑: 取Top-5预测 → 命中{target_classes}中任一类 → 确定为该类")
        print(f"          未命中 → 标记为 'unknown'")

    print("\n  摄像头已开启...")
    print("  q=退出 | s=保存裁剪图 | +/-=阈值 | r=录制 | f=切换过滤 | g=切换门控")
    print()

    # ===================== 状态变量 =====================
    threshold = args.conf
    record = False
    video_writer = None
    fps = 0
    fps_counter = 0
    fps_timer = time.time()
    last_crop = None
    last_detection = None
    last_top_predictions = []
    last_cls_raw = []
    last_gating_result = None   # (final_class, final_conf, is_matched)
    save_count = 0
    filter_mode = active_classes is not None  # 是否开启过滤模式
    active_set = set(active_classes) if active_classes else set()

    # ===================== 主循环 =====================
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 读取画面失败")
            time.sleep(0.1)
            continue

        t0 = time.perf_counter()

        # --- 检测推理 ---
        results = model(
            frame,
            imgsz=args.imgsz,
            verbose=False,
            conf=threshold,
            iou=args.iou,
            max_det=5,
            augment=args.augment,
            agnostic_nms=False,
        )
        boxes = results[0].boxes

        # --- 处理检测结果 ---
        detected = False
        best_box = None
        best_conf = 0
        top_predictions = []

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    xyxy = box.xyxy[0].cpu().numpy()
                    best_box = (int(xyxy[0]), int(xyxy[1]),
                                int(xyxy[2]), int(xyxy[3]))

            if best_box is not None:
                detected = True
                last_detection = (best_box, best_conf)

                # 裁剪靶标
                crop = crop_target(frame, best_box, margin_ratio=0.08)
                if crop is not None:
                    last_crop = crop

                    # 100 类分类推理
                    if cls_model is not None:
                        cls_results = cls_model(crop, verbose=False,
                                                imgsz=args.cls_imgsz)
                        all_preds = get_top_predictions(cls_results, cls_names, k=5)
                        last_cls_raw = all_preds

                        # --- Top-5 白名单门控 ---
                        if gating_mode and target_set:
                            final_cls, final_conf, is_matched, display_preds = \
                                apply_top5_gating(all_preds, target_set)
                            last_gating_result = (final_cls, final_conf, is_matched)

                            if filter_mode and active_set:
                                # 门控 + 过滤: 从 display 中再筛一次
                                top_predictions = [
                                    (n, p) for n, p in display_preds if n in active_set
                                ][:args.topk]
                                if not top_predictions:
                                    top_predictions = display_preds[:args.topk]
                            else:
                                top_predictions = display_preds[:args.topk]
                        elif filter_mode and active_set:
                            # 仅过滤模式（无门控）
                            top_predictions = [
                                (n, p) for n, p in all_preds if n in active_set
                            ][:args.topk]
                            if not top_predictions:
                                top_predictions = all_preds[:args.topk]
                            last_gating_result = None
                        else:
                            # 传统模式
                            top_predictions = all_preds[:args.topk]
                            last_gating_result = None

                        last_top_predictions = top_predictions

        # --- 绘制 ---
        display_frame = frame.copy()

        if detected and best_box is not None:
            display_frame = draw_detection(
                display_frame, best_box, best_conf,
                last_top_predictions, last_gating_result
            )

        # FPS
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps = fps_counter / (time.time() - fps_timer)
            fps_counter = 0
            fps_timer = time.time()

        # 信息条
        if detected and last_gating_result is not None:
            final_cls, final_conf, is_matched = last_gating_result
            gate_status = f"Gating: HIT->{final_cls}" if is_matched else "Gating: MISS"
            if last_top_predictions:
                pred_str = " | ".join(
                    [f"{n}({p:.0%})" for n, p in last_top_predictions[:3]]
                )
            else:
                pred_str = "--"
            info_lines = [
                f"Conf: {threshold:.2f}  IoU: {args.iou:.2f}  "
                f"Classes: {len(cls_names)}  {gate_status}",
                f"Det: {best_conf:.2%}  |  {pred_str}",
            ]
        elif detected and last_top_predictions:
            pred_str = " | ".join(
                [f"{n}({p:.0%})" for n, p in last_top_predictions[:3]]
            )
            info_lines = [
                f"Conf: {threshold:.2f}  IoU: {args.iou:.2f}  "
                f"Classes: {len(cls_names)}  Gating: {'ON' if gating_mode else 'OFF'}",
                f"Det: {best_conf:.2%}  |  {pred_str}",
            ]
        else:
            info_lines = [
                f"Conf: {threshold:.2f}  IoU: {args.iou:.2f}  "
                f"Classes: {len(cls_names)}  Gating: {'ON' if gating_mode else 'OFF'}",
                f"Detected: {best_conf:.2%}" if detected else
                "Detected: --  (try lowering threshold with - key)",
            ]
        display_frame = put_info_bar(display_frame, info_lines, fps)

        # --- 显示 ---
        cv2.imshow("Target Detection", display_frame)

        # 裁剪窗口
        if last_crop is not None:
            crop_disp = resize_to_display(last_crop, 400)
            hc, wc = crop_disp.shape[:2]

            # 底部信息栏
            n_preds = len(last_top_predictions) if last_top_predictions else 0
            has_gating = last_gating_result is not None
            bar_h = 22 * (n_preds + (1 if has_gating else 0)) + 8 if n_preds > 0 else 28
            cv2.rectangle(crop_disp, (0, max(0, hc - bar_h)),
                          (wc, hc), (0, 0, 0), -1)

            ty = hc - bar_h + 16

            # 门控结果行
            if has_gating:
                final_cls, final_conf, is_matched = last_gating_result
                if is_matched:
                    gate_color = (0, 255, 200)
                    gate_text = f"[GATED] -> {final_cls} ({final_conf:.0%})"
                else:
                    gate_color = (80, 80, 255)
                    gate_text = "[GATED] -> UNKNOWN"
                cv2.putText(crop_disp, gate_text, (4, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, gate_color, 2)
                ty += 20

            if last_top_predictions:
                for rank, (name, prob) in enumerate(last_top_predictions[:3]):
                    color = (0, 255, 255) if rank == 0 else (180, 180, 180)
                    cv2.putText(crop_disp, f"#{rank + 1} {name} {prob:.1%}",
                                (4, ty), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5 if rank == 0 else 0.4, color, 2 if rank == 0 else 1)
                    ty += 20
            elif not has_gating:
                cv2.putText(crop_disp, f"target {best_conf:.2%}",
                            (5, hc - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            cv2.imshow("Target Crop", crop_disp)
        else:
            blank = np.full((200, 280, 3), 40, dtype=np.uint8)
            cv2.putText(blank, "No Target", (55, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1)
            cv2.putText(blank, "place target in view", (25, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
            cv2.imshow("Target Crop", blank)

        # --- 录制 ---
        if record:
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_path = os.path.join(SAVE_DIR, f"record_{timestamp}.mp4")
                video_writer = cv2.VideoWriter(
                    video_path, fourcc, 15.0,
                    (display_frame.shape[1], display_frame.shape[0])
                )
                print(f"  开始录制: {video_path}")
            video_writer.write(display_frame)
        else:
            if video_writer is not None:
                video_writer.release()
                video_writer = None
                print("  录制停止")

        # --- 按键处理 ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            if last_crop is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                # 门控模式下用最终判定类别作为文件名前缀
                if last_gating_result is not None:
                    prefix = last_gating_result[0]  # final_class
                elif last_top_predictions:
                    prefix = last_top_predictions[0][0]
                else:
                    prefix = "target"
                save_path = os.path.join(SAVE_DIR, f"{prefix}_{timestamp}.jpg")
                cv2.imwrite(save_path, last_crop)
                save_count += 1
                print(f"  [SAVE] {save_path}")
            else:
                print("  [INFO] 无靶标可保存")
        elif key == ord('+') or key == ord('='):
            threshold = min(0.95, threshold + 0.05)
            print(f"  阈值: {threshold:.2f}")
        elif key == ord('-') or key == ord('_'):
            threshold = max(0.05, threshold - 0.05)
            print(f"  阈值: {threshold:.2f}")
        elif key == ord('r'):
            record = not record
            if not record and video_writer is not None:
                video_writer.release()
                video_writer = None
            print(f"  录制: {'ON' if record else 'OFF'}")
        elif key == ord('f'):
            filter_mode = not filter_mode
            print(f"  类别过滤: {'ON' if filter_mode else 'OFF'} "
                  f"({'all' if not filter_mode else active_classes})")
        elif key == ord('g'):
            gating_mode = not gating_mode
            last_gating_result = None
            status = "ON" if gating_mode else "OFF (Top-1 mode)"
            print(f"  Top-5 白名单门控: {status}  "
                  f"目标类: {target_classes}")

    # ===================== 清理 =====================
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    print(f"\n  退出。共保存 {save_count} 张裁剪图 -> {SAVE_DIR}/")


if __name__ == "__main__":
    main()
