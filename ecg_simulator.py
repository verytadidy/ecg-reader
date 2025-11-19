"""
ECG 仿真器 V46 - 终极融合版

融合特性:
1. ✅ V38的优质渲染 (分隔符、纹理、背景)
2. ✅ V46的完整标注 (13层文字、定标脉冲、真值信号)
3. ✅ 100%兼容渐进式模型架构
4. ✅ 位置和字体随机性

运行:
    python ecg_simulator_v46_ultimate.py --workers 8 --limit 100
"""

import numpy as np
import pandas as pd
import cv2
import os
import random
import multiprocessing
from functools import partial
from tqdm import tqdm
import json
import sys

# ============================
# 1. 核心常量
# ============================
class DegradationType:
    CLEAN = "0001"
    PRINTED_COLOR = "0003"
    PRINTED_BW = "0004"
    PHOTO_PRINT = "0005"
    PHOTO_SCREEN = "0006"
    STAINED = "0009"
    DAMAGED = "0010"
    MOLD_COLOR = "0011"
    MOLD_BW = "0012"

class LayoutType:
    LAYOUT_3X4_PLUS_II = "3x4+1"
    LAYOUT_3X4 = "3x4"
    LAYOUT_6X2 = "6x2"
    LAYOUT_12X1 = "12x1"

LAYOUT_CONFIGS = {
    LayoutType.LAYOUT_3X4_PLUS_II: {
        'leads': {
            'I': (0,0), 'aVR': (0,1), 'V1': (0,2), 'V4': (0,3),
            'II': (1,0), 'aVL': (1,1), 'V2': (1,2), 'V5': (1,3),
            'III': (2,0), 'aVF': (2,1), 'V3': (2,2), 'V6': (2,3)
        },
        'long_lead': 'II', 'rows': 3, 'cols': 4
    },
    LayoutType.LAYOUT_3X4: {
        'leads': {
            'I': (0,0), 'aVR': (0,1), 'V1': (0,2), 'V4': (0,3),
            'II': (1,0), 'aVL': (1,1), 'V2': (1,2), 'V5': (1,3),
            'III': (2,0), 'aVF': (2,1), 'V3': (2,2), 'V6': (2,3)
        },
        'long_lead': None, 'rows': 3, 'cols': 4
    },
    LayoutType.LAYOUT_6X2: {
        'leads': {
            'I': (0,0), 'II': (1,0), 'III': (2,0), 'aVR': (3,0), 'aVL': (4,0), 'aVF': (5,0),
            'V1': (0,1), 'V2': (1,1), 'V3': (2,1), 'V4': (3,1), 'V5': (4,1), 'V6': (5,1)
        },
        'long_lead': None, 'rows': 6, 'cols': 2
    },
    LayoutType.LAYOUT_12X1: {
        'leads': {
            'I': (0,0), 'II': (1,0), 'III': (2,0), 'aVR': (3,0), 'aVL': (4,0), 'aVF': (5,0),
            'V1': (6,0), 'V2': (7,0), 'V3': (8,0), 'V4': (9,0), 'V5': (10,0), 'V6': (11,0)
        },
        'long_lead': None, 'rows': 12, 'cols': 1
    }
}

LEAD_TO_ID_MAP = {
    'I': 1, 'II': 2, 'III': 3,
    'aVR': 4, 'aVL': 5, 'aVF': 6,
    'V1': 7, 'V2': 8, 'V3': 9,
    'V4': 10, 'V5': 11, 'V6': 12
}

LEAD_NAMES_ORDERED = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                       'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

COLOR_GRID_MINOR_BASE_OPTIONS = [(180, 160, 255), (170, 150, 255)]
COLOR_GRID_MAJOR_BASE_OPTIONS = [(160, 140, 255), (150, 130, 255)]

FONT_LIST = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL
]

# ============================
# 2. 辅助函数 (融合V38的高质量纹理)
# ============================
def random_color_variations(base_color, variation=30):
    c = np.array(base_color, dtype=np.int16)
    delta = np.random.randint(-variation, variation+1, 3)
    result = np.clip(c + delta, 0, 255).astype(np.int32)
    return tuple(int(x) for x in result)

def get_random_paper_color():
    paper_types = [
        (255, 255, 255), (255, 252, 250), (252, 250, 248), (250, 248, 245),
        (248, 245, 242), (245, 242, 240), (242, 240, 238), (255, 250, 248),
    ]
    base = random.choice(paper_types)
    return random_color_variations(base, 3)

def get_random_ink_color():
    if random.random() < 0.85:
        return random_color_variations((0, 0, 0), 20)
    else:
        if random.random() < 0.5:
            return random_color_variations((80, 50, 0), 30)
        else:
            return random_color_variations((50, 50, 150), 30)

def generate_paper_texture(h, w, color, grid_img=None):
    """V38的高质量纸张纹理"""
    base_color = np.array(color, dtype=np.float32)
    texture = np.full((h, w, 3), base_color, dtype=np.float32)
    
    # 纤维纹理
    fiber_noise = np.random.normal(0, 0.8, (h//4, w//4, 3))
    fiber_noise = cv2.resize(fiber_noise, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 颗粒
    grain = np.random.normal(0, 0.3, (h, w, 3))
    
    # 织物图案
    X, Y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    pattern = np.sin(X * 50) * np.cos(Y * 50) * 0.3
    
    texture += fiber_noise + grain + pattern[..., None]
    texture = np.clip(texture, 0, 255).astype(np.uint8)
    
    # 融合网格
    if grid_img is not None:
        line_mask = ((grid_img[:,:,2] > 200) & 
                    (grid_img[:,:,1] < 220) & 
                    (grid_img[:,:,0] < 220)).astype(np.uint8) * 255
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
        line_mask_f = line_mask.astype(np.float32) / 255.0
        line_mask_f = line_mask_f[..., None]
        texture = (texture.astype(np.float32) * (1 - line_mask_f) + 
                  grid_img.astype(np.float32) * line_mask_f).astype(np.uint8)
    
    # 轻微光照噪声
    light_noise = np.random.normal(0, 0.1, (h, w, 3))
    texture = np.clip(texture.astype(np.float32) + light_noise, 0, 255).astype(np.uint8)
    
    return texture

def generate_scanner_background(h, w):
    """V38的多样化扫描仪背景"""
    mode = random.choice(['dark_gray', 'black', 'wood'])
    if mode == 'dark_gray':
        color = random_color_variations((40, 40, 40), 10)
    elif mode == 'black':
        color = random_color_variations((5, 5, 5), 5)
    else:
        color = random_color_variations((50, 80, 120), 20)
    return generate_paper_texture(h, w, color)

def transform_bbox(bbox, offset, M_geo):
    """坐标变换"""
    x1, y1, x2, y2 = bbox
    ox, oy = offset
    
    pts = np.array([
        [x1 + ox, y1 + oy],
        [x2 + ox, y1 + oy],
        [x2 + ox, y2 + oy],
        [x1 + ox, y2 + oy]
    ], dtype=np.float32)
    
    pts = np.array([pts])
    transformed_pts = cv2.perspectiveTransform(pts, M_geo)
    
    pts_squeezed = transformed_pts[0]
    new_x1 = int(np.min(pts_squeezed[:, 0]))
    new_y1 = int(np.min(pts_squeezed[:, 1]))
    new_x2 = int(np.max(pts_squeezed[:, 0]))
    new_y2 = int(np.max(pts_squeezed[:, 1]))
    
    return [new_x1, new_y1, new_x2, new_y2]

def sample_physical_params(layout_type):
    """物理参数采样"""
    if layout_type in [LayoutType.LAYOUT_3X4_PLUS_II, LayoutType.LAYOUT_3X4]:
        gain_mm_mv = random.choice([5.0, 10.0])
    elif layout_type == LayoutType.LAYOUT_12X1:
        r = random.random()
        if r < 0.70:
            gain_mm_mv = 5.0
        elif r < 0.85:
            gain_mm_mv = 10.0
        else:
            gain_mm_mv = 2.5
    elif layout_type == LayoutType.LAYOUT_6X2:
        gain_mm_mv = random.choices([10.0, 5.0, 20.0], weights=[0.50, 0.40, 0.10], k=1)[0]
    else:
        gain_mm_mv = random.choice([5.0, 10.0])
    
    paper_speed_mm_s = random.choice([25.0, 50.0])
    
    return {
        'paper_speed_mm_s': paper_speed_mm_s,
        'gain_mm_mv': gain_mm_mv,
        'lead_durations': {'long': 10.0, 'short': 2.5}
    }

# ============================
# 3. 渲染子模块 (融合版)
# ============================
def render_calibration_pulse(img, alpha_auxiliary, alpha_text, x_start, y_baseline,
                            px_per_mm, px_per_mv, paper_speed_mm_s, ink_color, thick):
    """
    渲染定标脉冲 (V38样式 + V46的掩码)
    
    返回: (x_end, bbox)
    """
    pulse_duration_s = 0.2
    pulse_width_px = int(pulse_duration_s * paper_speed_mm_s * px_per_mm * random.uniform(0.9, 1.1))
    pulse_height_px = int(1.0 * px_per_mv * random.uniform(0.95, 1.05))
    
    ink_color = tuple(int(c) for c in ink_color)
    
    x_mid1 = x_start + int(pulse_width_px * 0.2)
    x_mid2 = x_start + int(pulse_width_px * 0.8)
    x_end = x_start + pulse_width_px
    
    pts = np.array([
        [x_start, y_baseline],
        [x_mid1, y_baseline],
        [x_mid1, y_baseline - pulse_height_px],
        [x_mid2, y_baseline - pulse_height_px],
        [x_mid2, y_baseline],
        [x_end, y_baseline]
    ], dtype=np.int32).reshape((-1, 1, 2))
    
    # 绘制在RGB图和辅助掩码上
    cv2.polylines(img, [pts], False, ink_color, thick, cv2.LINE_AA)
    cv2.polylines(alpha_auxiliary, [pts], False, 255, thick, cv2.LINE_AA)
    
    # 移除"1mV"文字标注
    
    bbox = [x_start, y_baseline - pulse_height_px, x_end, y_baseline]
    
    return x_end, bbox

def draw_lead_separator(img, alpha_auxiliary, x, y_baseline, px_per_mm, ink_color):
    """
    绘制短粗实黑分隔符 (V38样式)
    """
    is_floating = random.random() > 0.7
    
    length_mm = random.uniform(4.0, 8.0)
    length_px = int(length_mm * px_per_mm)
    
    thickness = random.randint(2, 4)
    
    if is_floating:
        float_offset_mm = random.uniform(1.5, 3.0)
        float_offset_px = int(float_offset_mm * px_per_mm)
        y2 = y_baseline - float_offset_px
        y1 = y2 - length_px
    else:
        y_center = y_baseline + random.randint(-2, 2)
        y1 = y_center - length_px // 2
        y2 = y_center + length_px // 2
    
    x_jitter = x + random.randint(-1, 1)
    
    pts = np.array([[x_jitter, y1], [x_jitter, y2]], dtype=np.int32).reshape((-1, 1, 2))
    
    cv2.polylines(img, [pts], False, ink_color, thickness, cv2.LINE_AA)
    cv2.polylines(alpha_auxiliary, [pts], False, 255, thickness, cv2.LINE_AA)

def render_lead_text(img, text_mask, text, x, y, font, scale, color, thick, lead_id):
    """
    渲染导联文字 (带位置随机性)
    
    Args:
        text_mask: (13, H, W) 多层文字掩码
        lead_id: 1-12 (对应text_mask的通道1-12)
    """
    # 添加位置抖动
    x_jitter = x + random.randint(-2, 2)
    y_jitter = y + random.randint(-2, 2)
    
    # 字体缩放随机性
    scale_jitter = scale * random.uniform(0.95, 1.05)
    
    cv2.putText(img, text, (x_jitter, y_jitter), font, scale_jitter, color, thick, cv2.LINE_AA)
    cv2.putText(text_mask[lead_id], text, (x_jitter, y_jitter), font, scale_jitter, 255, thick, cv2.LINE_AA)
    
    (w, h), _ = cv2.getTextSize(text, font, scale_jitter, thick)
    return [x_jitter, y_jitter - h, x_jitter + w, y_jitter]

def render_layout_3x4_plus_II_ultimate(df, sig_rgb, wave_label, text_masks, alpha_auxiliary,
                                       baseline_heatmaps, params, ink_color, font, fs,
                                       render_params, lead_rois_dict, calibration_pulse_bboxes):
    """
    终极融合版: 3x4+II 布局渲染
    
    特性:
    - V38的渲染质量 (分隔符、脉冲样式)
    - V46的完整标注 (13层文字、辅助掩码、真值信号)
    - 位置和字体随机性
    """
    h, w = render_params['h'], render_params['w']
    MT_px = render_params['MT_px']
    MB_px = render_params['MB_px']
    signal_start_x = render_params['signal_start_x']
    px_per_s = render_params['px_per_s_on_paper']
    eff_px_mm = render_params['effective_px_per_mm']
    eff_px_mv = render_params['effective_px_per_mv']
    
    main_h = (h - MT_px - MB_px) * 0.75
    rhythm_h = (h - MT_px - MB_px) * 0.25
    row_h = main_h / 3
    TIME_PER_COL = 2.5
    
    thick_signal = random.randint(1, 2)
    thick_pulse = thick_signal + 1
    font_scale = random.uniform(0.9, 1.2)
    
    # 定标脉冲起始位置 (带随机性)
    x_pulse_start = int(signal_start_x - random.uniform(10.0, 12.0) * eff_px_mm)
    x_pulse_end_max = 0
    
    # 渲染主网格的定标脉冲
    for r in range(3):
        base_y = int(MT_px + (r + 0.5) * row_h)
        x_end, pulse_bbox = render_calibration_pulse(
            sig_rgb, alpha_auxiliary[0], alpha_auxiliary[0],  # 两个参数都用alpha_auxiliary
            x_pulse_start, base_y,
            eff_px_mm, eff_px_mv, params['paper_speed_mm_s'],
            ink_color, thick_pulse
        )
        x_pulse_end_max = max(x_pulse_end_max, x_end)
        calibration_pulse_bboxes.append(pulse_bbox)
    
    # 渲染长导联的定标脉冲
    base_y_long = int(MT_px + main_h + rhythm_h / 2)
    x_end_long, pulse_bbox_long = render_calibration_pulse(
        sig_rgb, alpha_auxiliary[0], alpha_auxiliary[0],  # 两个参数都用alpha_auxiliary
        x_pulse_start, base_y_long,
        eff_px_mm, eff_px_mv, params['paper_speed_mm_s'],
        ink_color, thick_pulse
    )
    calibration_pulse_bboxes.append(pulse_bbox_long)
    
    total_samples = min(len(df), int(fs * 10.0))
    
    # 渲染12个短导联
    for lead, (r, c) in LAYOUT_CONFIGS[LayoutType.LAYOUT_3X4_PLUS_II]['leads'].items():
        if lead not in df.columns:
            continue
        
        base_y = int(MT_px + (r + 0.5) * row_h)
        t_start = c * TIME_PER_COL
        t_end = t_start + TIME_PER_COL
        
        idx_start = int(t_start * fs)
        idx_end = min(int(t_end * fs), total_samples)
        sig = df[lead].iloc[idx_start:idx_end].dropna().values
        
        x_start_line = int(signal_start_x + t_start * px_per_s)
        x_end_line = int(signal_start_x + t_end * px_per_s)
        
        lead_id = LEAD_TO_ID_MAP[lead]
        
        # 绘制基线
        cv2.line(baseline_heatmaps[lead_id - 1], (x_start_line, base_y),
                (x_end_line, base_y), 255, thick_signal, cv2.LINE_AA)
        
        # 绘制波形
        if len(sig) > 0:
            t_axis = np.linspace(t_start, t_end, len(sig))
            xs = signal_start_x + t_axis * px_per_s
            ys = base_y - sig * eff_px_mv
            xs = np.clip(xs, 0, w - 1)
            pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape((-1, 1, 2))
            
            cv2.polylines(sig_rgb, [pts], False, ink_color, thick_signal, cv2.LINE_AA)
            cv2.polylines(wave_label, [pts], False, lead_id, thick_signal, cv2.LINE_AA)
        
        # 绘制导联文字 (带位置随机性)
        txt_y = int(base_y - row_h * 0.3)
        if c == 0:
            txt_x_gap_mm = random.uniform(2.0, 5.0)
            txt_x = int(x_pulse_end_max + txt_x_gap_mm * eff_px_mm)
        else:
            txt_x = int(signal_start_x + (c * TIME_PER_COL) * px_per_s + 
                       random.uniform(2, 4) * eff_px_mm)
        
        txt_x = max(0, min(txt_x, w - 1))
        txt_y = max(10, min(txt_y, h - 1))
        
        txt_bbox = render_lead_text(
            sig_rgb, text_masks, lead,
            txt_x, txt_y, font, font_scale, ink_color, 2, lead_id
        )
        
        # 保存RoI
        lead_rois_dict[lead] = {
            'bbox': [x_start_line, int(base_y - row_h/2), x_end_line, int(base_y + row_h/2)],
            'text_bbox': txt_bbox,
            'baseline_y': base_y,
            'time_range': [t_start, t_end]
        }
    
    # 绘制分隔符
    for c in range(1, 4):
        sep_x = int(signal_start_x + (c * TIME_PER_COL) * px_per_s)
        for r in range(3):
            base_y = int(MT_px + (r + 0.5) * row_h)
            draw_lead_separator(sig_rgb, alpha_auxiliary[0], sep_x, base_y, eff_px_mm, ink_color)
    
    # 渲染长导联
    long_lead = 'II'
    if long_lead in df.columns:
        base_y = int(MT_px + main_h + rhythm_h / 2)
        
        # 文字
        txt_x = int(x_end_long + random.uniform(2, 4) * eff_px_mm)
        txt_y = int(base_y - rhythm_h * 0.3)
        txt_x = max(0, min(txt_x, w - 1))
        txt_y = max(10, min(txt_y, h - 1))
        
        lead_id = LEAD_TO_ID_MAP[long_lead]
        txt_bbox = render_lead_text(
            sig_rgb, text_masks, long_lead,
            txt_x, txt_y, font, font_scale, ink_color, 2, lead_id
        )
        
        # 基线
        cv2.line(baseline_heatmaps[lead_id - 1],
                (signal_start_x, base_y),
                (signal_start_x + render_params['signal_draw_w_px'], base_y),
                255, thick_signal, cv2.LINE_AA)
        
        # 波形
        idx_start = 0
        idx_end = min(int(10.0 * fs), total_samples)
        sig_full = df[long_lead].iloc[idx_start:idx_end].dropna().values
        
        if len(sig_full) > 0:
            t_axis = np.linspace(0, 10.0, len(sig_full))
            xs = signal_start_x + t_axis * px_per_s
            ys = base_y - sig_full * eff_px_mv
            xs = np.clip(xs, 0, w - 1)
            pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape((-1, 1, 2))
            
            cv2.polylines(sig_rgb, [pts], False, ink_color, thick_signal, cv2.LINE_AA)
            cv2.polylines(wave_label, [pts], False, lead_id, thick_signal, cv2.LINE_AA)
        
        # 保存RoI
        lead_rois_dict[long_lead] = {
            'bbox': [signal_start_x, int(base_y - 150),
                    signal_start_x + render_params['signal_draw_w_px'], int(base_y + 150)],
            'text_bbox': txt_bbox,
            'baseline_y': base_y,
            'time_range': [0.0, 10.0]
        }
    
    # 不再在这里生成背景通道，而是在主渲染函数中统一处理

def render_clean_ecg_ultimate(df, layout_type, params, fs, sig_len):
    """
    终极融合版主渲染函数
    
    特性:
    - V38的高质量纸张和网格纹理
    - V46的完整标注体系
    - 所有标注符合渐进式模型需求
    """
    h, w = 1700, 2200
    MT_px = int(h * 0.08)
    MB_px = int(h * 0.05)
    ML_px = int(w * 0.05)
    MR_px = int(w * 0.05)
    lead_in_area_px = int(w * 0.1)
    
    paper_speed = params['paper_speed_mm_s']
    gain = params['gain_mm_mv']
    
    signal_start_x = ML_px + lead_in_area_px
    signal_draw_w_px = w - signal_start_x - MR_px
    PAPER_DURATION_S = 10.0
    px_per_s = signal_draw_w_px / PAPER_DURATION_S
    eff_px_mm = px_per_s / paper_speed
    eff_px_mv = eff_px_mm * gain
    
    # 1. 纸张和网格 (V38质量)
    paper_color = get_random_paper_color()
    plain_paper = generate_paper_texture(h, w, paper_color)
    temp_base = plain_paper.copy()
    
    grid_minor_color = random_color_variations(random.choice(COLOR_GRID_MINOR_BASE_OPTIONS), 3)
    grid_major_color = random_color_variations(random.choice(COLOR_GRID_MAJOR_BASE_OPTIONS), 3)
    
    # 绘制网格
    for x in np.arange(0, w, eff_px_mm):
        cv2.line(temp_base, (int(x), 0), (int(x), h), grid_minor_color, 2)
    for y in np.arange(0, h, eff_px_mm):
        cv2.line(temp_base, (0, int(y)), (w, int(y)), grid_minor_color, 2)
    for x in np.arange(0, w, eff_px_mm * 5):
        cv2.line(temp_base, (int(x), 0), (int(x), h), grid_major_color, 3)
    for y in np.arange(0, h, eff_px_mm * 5):
        cv2.line(temp_base, (0, int(y)), (w, int(y)), grid_major_color, 3)
    
    base = generate_paper_texture(h, w, paper_color, grid_img=temp_base)
    
    # 2. 内容图层 - 使用白色背景而不是黑色
    sig_rgb = np.full((h, w, 3), 255, dtype=np.uint8)  # 白色背景
    wave_label = np.zeros((h, w), dtype=np.uint8)
    text_masks = np.zeros((13, h, w), dtype=np.uint8)
    alpha_auxiliary = np.zeros((1, h, w), dtype=np.uint8)
    baseline_heatmaps = np.zeros((12, h, w), dtype=np.uint8)
    ps_mask = np.zeros((1, h, w), dtype=np.uint8)
    gain_mask = np.zeros((1, h, w), dtype=np.uint8)
    
    ink_color = get_random_ink_color()
    font = random.choice(FONT_LIST)
    
    # 3. 添加页眉 (带随机性)
    font_scale_header = random.uniform(0.8, 1.1)
    header_text = f"ID: {random.randint(10000, 99999)}_hr"
    header_x = ML_px + random.randint(-5, 5)
    header_y = MT_px - int(10 * eff_px_mm) + random.randint(-3, 3)
    cv2.putText(sig_rgb, header_text, (header_x, header_y),
                font, font_scale_header, ink_color, 1, cv2.LINE_AA)
    cv2.putText(alpha_auxiliary[0], header_text, (header_x, header_y),
                font, font_scale_header, 255, 1, cv2.LINE_AA)
    
    render_params = {
        'h': h, 'w': w,
        'MT_px': MT_px, 'MB_px': MB_px,
        'ML_px': ML_px, 'MR_px': MR_px,
        'signal_start_x': signal_start_x,
        'signal_draw_w_px': signal_draw_w_px,
        'px_per_s_on_paper': px_per_s,
        'effective_px_per_mm': eff_px_mm,
        'effective_px_per_mv': eff_px_mv
    }
    
    lead_rois = {}
    calibration_pulse_bboxes = []
    
    # 4. 根据布局类型渲染
    if layout_type == LayoutType.LAYOUT_3X4_PLUS_II:
        render_layout_3x4_plus_II_ultimate(
            df, sig_rgb, wave_label, text_masks, alpha_auxiliary,
            baseline_heatmaps, params, ink_color, font, fs,
            render_params, lead_rois, calibration_pulse_bboxes
        )
    elif layout_type == LayoutType.LAYOUT_3X4:
        render_layout_3x4_plus_II_ultimate(
            df, sig_rgb, wave_label, text_masks, alpha_auxiliary,
            baseline_heatmaps, params, ink_color, font, fs,
            render_params, lead_rois, calibration_pulse_bboxes
        )
    else:
        render_layout_3x4_plus_II_ultimate(
            df, sig_rgb, wave_label, text_masks, alpha_auxiliary,
            baseline_heatmaps, params, ink_color, font, fs,
            render_params, lead_rois, calibration_pulse_bboxes
        )
    
    # 5. 添加页脚 (纸速/增益，带随机位置)
    font_scale_footer = random.uniform(0.9, 1.2)
    base_x = w // 2 - int(50 * eff_px_mm)
    base_y = h - MB_px + int(5 * eff_px_mm)
    offset_x = random.randint(-int(30 * eff_px_mm), int(30 * eff_px_mm))
    offset_y = random.randint(-int(8 * eff_px_mm), int(3 * eff_px_mm))
    
    footer_x = max(ML_px, min(base_x + offset_x, w - int(100 * eff_px_mm)))
    footer_y = max(h - MB_px - int(15 * eff_px_mm), min(base_y + offset_y, h - int(5 * eff_px_mm)))
    
    ps_text = f"{params['paper_speed_mm_s']:.1f}mm/s"
    gain_text = f"{params['gain_mm_mv']:.1f}mm/mV"
    
    # 纸速文字
    cv2.putText(sig_rgb, ps_text, (footer_x, footer_y),
                font, font_scale_footer, ink_color, 2, cv2.LINE_AA)
    cv2.putText(ps_mask[0], ps_text, (footer_x, footer_y),
                font, font_scale_footer, 255, 2, cv2.LINE_AA)
    (ps_w, ps_h), _ = cv2.getTextSize(ps_text, font, font_scale_footer, 2)
    ps_bbox = [footer_x, footer_y - ps_h, footer_x + ps_w, footer_y]
    
    # 增益文字 (在纸速右侧)
    gain_x = footer_x + ps_w + int(20 * eff_px_mm)
    cv2.putText(sig_rgb, gain_text, (gain_x, footer_y),
                font, font_scale_footer, ink_color, 2, cv2.LINE_AA)
    cv2.putText(gain_mask[0], gain_text, (gain_x, footer_y),
                font, font_scale_footer, 255, 2, cv2.LINE_AA)
    (gain_w, gain_h), _ = cv2.getTextSize(gain_text, font, font_scale_footer, 2)
    gain_bbox = [gain_x, footer_y - gain_h, gain_x + gain_w, footer_y]
    
    ocr_targets = {
        'paper_speed': ps_bbox,
        'gain': gain_bbox,
        'calibration_pulses': calibration_pulse_bboxes
    }
    
    # 6. 图像融合 [修复版 - 包含所有内容]
    # 生成背景通道 (通道0) - 在融合前
    foreground_union = np.clip(text_masks[1:].sum(axis=0), 0, 255).astype(np.uint8)
    text_masks[0] = 255 - foreground_union
    
    # 收集所有需要显示的内容掩码
    wave_mask_binary = (wave_label > 0).astype(np.uint8) * 255
    
    # 文字掩码: 合并所有导联文字 (通道1-12)
    text_mask_combined = foreground_union  # 已经计算过了
    
    # 合并所有alpha通道
    combined_alpha = np.maximum(wave_mask_binary, alpha_auxiliary[0])
    combined_alpha = np.maximum(combined_alpha, text_mask_combined)  # 添加导联文字
    combined_alpha = np.maximum(combined_alpha, ps_mask[0])          # 添加纸速文字
    combined_alpha = np.maximum(combined_alpha, gain_mask[0])        # 添加增益文字
    
    # Alpha混合融合
    alpha_mask = (combined_alpha.astype(np.float32) / 255.0)[..., None]
    clean_img = (base.astype(np.float32) * (1.0 - alpha_mask) + 
                 sig_rgb.astype(np.float32) * alpha_mask).astype(np.uint8)
    
    # 7. 元数据
    metadata_params = params.copy()
    metadata_params['effective_px_per_mm'] = eff_px_mm
    metadata_params['effective_px_per_mv'] = eff_px_mv
    
    return (clean_img, base, wave_label, text_masks, alpha_auxiliary,
            baseline_heatmaps, ps_mask, gain_mask,
            paper_color, metadata_params, lead_rois, ocr_targets)

# ============================
# 4. 退化引擎 (V38质量)
# ============================
def add_stains(img):
    """V38的污渍效果"""
    h, w = img.shape[:2]
    stain_color = np.array(random_color_variations((70, 105, 140), 20), dtype=np.float32)
    stain_overlay = np.full(img.shape, 255, dtype=np.float32)
    stain_layer_f = np.zeros((h, w), dtype=np.float32)
    
    for _ in range(random.randint(1, 3)):
        center = (np.random.randint(0, w), np.random.randint(0, h))
        axes = (np.random.randint(w//5, w//2), np.random.randint(h//5, h//2))
        angle = np.random.randint(0, 180)
        cv2.ellipse(stain_layer_f, center, axes, angle, 0, 360, random.uniform(0.5, 1.0), -1)
    
    mask_blurred = cv2.GaussianBlur(stain_layer_f, (0, 0), sigmaX=w/15)
    kernel_size = random.randint(3, 7)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_eroded = cv2.erode(mask_blurred, kernel, iterations=2)
    tide_line_mask = np.clip(mask_blurred - mask_eroded, 0, 1)
    stain_mask = ((mask_eroded * random.uniform(0.3, 0.6)) + 
                  (tide_line_mask * random.uniform(0.8, 1.0)))
    stain_mask = np.clip(stain_mask, 0, 1)
    
    for c in range(3):
        stain_overlay[:, :, c] = 255 * (1 - stain_mask) + stain_color[c] * stain_mask
    
    img_stained = (img.astype(np.float32) * stain_overlay / 255.0).clip(0, 255).astype(np.uint8)
    return img_stained

def add_severe_damage(img, *masks):
    """严重损坏效果"""
    h, w = img.shape[:2]
    num_damages = random.randint(2, 5)
    
    for _ in range(num_damages):
        damage_type = random.choice(['tear', 'hole', 'crease'])
        
        if damage_type == 'tear':
            x1, y1 = random.randint(0, w), random.randint(0, h)
            x2, y2 = random.randint(0, w), random.randint(0, h)
            thickness = random.randint(5, 15)
            cv2.line(img, (x1, y1), (x2, y2), (240, 240, 240), thickness)
            for mask in masks:
                if mask.ndim == 2:
                    cv2.line(mask, (x1, y1), (x2, y2), 0, thickness)
                else:
                    for i in range(mask.shape[0]):
                        cv2.line(mask[i], (x1, y1), (x2, y2), 0, thickness)
        
        elif damage_type == 'hole':
            center = (random.randint(0, w), random.randint(0, h))
            radius = random.randint(10, 30)
            cv2.circle(img, center, radius, (240, 240, 240), -1)
            for mask in masks:
                if mask.ndim == 2:
                    cv2.circle(mask, center, radius, 0, -1)
                else:
                    for i in range(mask.shape[0]):
                        cv2.circle(mask[i], center, radius, 0, -1)
    
    return (img,) + masks

def add_mold_spots(img):
    """霉斑效果"""
    h, w = img.shape[:2]
    num_spots = random.randint(10, 30)
    
    for _ in range(num_spots):
        center = (random.randint(0, w), random.randint(0, h))
        radius = random.randint(5, 20)
        if random.random() < 0.5:
            color = random_color_variations((50, 80, 30), 20)
        else:
            color = random_color_variations((40, 40, 40), 10)
        overlay = img.copy()
        cv2.circle(overlay, center, radius, color, -1)
        alpha = random.uniform(0.3, 0.7)
        img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    
    return img

def add_printer_halftone(img):
    """打印半调效果"""
    h, w = img.shape[:2]
    dot_size = random.randint(2, 4)
    pattern = np.zeros((h, w), dtype=np.float32)
    
    for i in range(0, h, dot_size * 2):
        for j in range(0, w, dot_size * 2):
            cv2.circle(pattern, (j, i), dot_size, 1.0, -1)
    
    pattern = cv2.GaussianBlur(pattern, (3, 3), 0)
    pattern = np.clip(pattern * random.uniform(0.1, 0.3), 0, 1)
    result = img.astype(np.float32)
    
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 - pattern) + 255 * pattern
    
    return np.clip(result, 0, 255).astype(np.uint8)

def add_screen_moire(img):
    """屏幕摩尔纹"""
    h, w = img.shape[:2]
    freq = random.uniform(0.05, 0.15)
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    pattern1 = np.sin(2 * np.pi * freq * X) * np.cos(2 * np.pi * freq * Y)
    pattern2 = np.sin(2 * np.pi * freq * 1.1 * (X + Y))
    moire = (pattern1 + pattern2) * random.uniform(15, 30)
    result = img.astype(np.float32) + moire[..., None]
    return np.clip(result, 0, 255).astype(np.uint8)

def add_motion_blur(img):
    """运动模糊"""
    size = random.randint(5, 15)
    angle = random.uniform(0, 180)
    kernel = np.zeros((size, size))
    kernel[int((size - 1) / 2), :] = np.ones(size)
    kernel = kernel / size
    M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (size, size))
    return cv2.filter2D(img, -1, kernel)

def add_jpeg_compression(img, quality=None):
    """JPEG压缩"""
    if quality is None:
        quality = random.randint(40, 70)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 1)

def apply_degradation_pipeline_ultimate(img, masks_dict, degradation_type, paper_color):
    """
    终极融合版退化管道
    
    特性:
    - V38的所有退化效果
    - V46的几何变换
    - 完整的掩码处理
    """
    h, w = img.shape[:2]
    
    # 1. 应用特定退化效果
    if degradation_type == DegradationType.PRINTED_COLOR:
        img = add_printer_halftone(img)
        img = cv2.GaussianBlur(img, (3, 3), 0)
    elif degradation_type == DegradationType.PRINTED_BW:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = add_printer_halftone(img)
    elif degradation_type == DegradationType.PHOTO_PRINT:
        img = add_motion_blur(img)
        img = add_jpeg_compression(img, quality=random.randint(50, 70))
    elif degradation_type == DegradationType.PHOTO_SCREEN:
        img = add_screen_moire(img)
        img = add_jpeg_compression(img, quality=random.randint(60, 80))
        img = img.astype(np.float32) * random.uniform(0.8, 1.0) + random.uniform(10, 30)
        img = np.clip(img, 0, 255).astype(np.uint8)
    elif degradation_type == DegradationType.STAINED:
        img = add_stains(img)
    elif degradation_type == DegradationType.DAMAGED:
        all_masks = list(masks_dict.values())
        results = add_severe_damage(img, *all_masks)
        img = results[0]
        for i, (k, v) in enumerate(masks_dict.items()):
            masks_dict[k] = results[i + 1]
    elif degradation_type in [DegradationType.MOLD_COLOR, DegradationType.MOLD_BW]:
        img = add_mold_spots(img)
        if degradation_type == DegradationType.MOLD_BW:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if degradation_type == DegradationType.CLEAN:
        return img, np.eye(3), masks_dict
    
    # 2. 几何变换
    M_geo = np.eye(3)
    
    if random.random() < 0.8:
        # 旋转
        angle = random.uniform(-5, 5)
        center = (w // 2, h // 2)
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        M_rot_3x3 = np.vstack([M_rot, [0, 0, 1]])
        
        # 透视
        src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        offset = min(h, w) * random.uniform(0.01, 0.08)
        random_offset = np.random.uniform(-offset, offset, (4, 2)).astype(np.float32)
        dst_pts = src_pts + random_offset
        M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        M_geo = M_persp @ M_rot_3x3
        
        # 应用变换
        border_color = random_color_variations((20, 20, 20), 10)
        img = cv2.warpPerspective(img, M_geo, (w, h), 
                                  flags=cv2.INTER_LINEAR, 
                                  borderValue=border_color)
        
        # 变换所有掩码
        warped_masks = {}
        for k, v in masks_dict.items():
            if v.ndim == 2:
                warped_masks[k] = cv2.warpPerspective(v, M_geo, (w, h),
                                                     flags=cv2.INTER_NEAREST,
                                                     borderValue=0)
            else:
                tmp = []
                for i in range(v.shape[0]):
                    tmp.append(cv2.warpPerspective(v[i], M_geo, (w, h),
                                                   flags=cv2.INTER_NEAREST,
                                                   borderValue=0))
                warped_masks[k] = np.stack(tmp)
        masks_dict = warped_masks
    
    # 3. 光照和噪声
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    radius = np.sqrt(X**2 + Y**2)
    vignette = 1 - np.clip(radius * random.uniform(0.5, 0.8), 0, 1)
    gradient = np.tile(np.linspace(random.uniform(0.7, 1.0), 
                                   random.uniform(0.7, 1.0), w), (h, 1))
    lighting_mask = np.clip(vignette * gradient, 0.4, 1.0)
    img = (img.astype(np.float32) * lighting_mask[..., None]).astype(np.uint8)
    
    if random.random() < 0.7:
        img = cv2.GaussianBlur(img, (5, 5), 0)
    
    noise = np.random.normal(0, 12, (h, w, 3))
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return img, M_geo, masks_dict

# ============================
# 5. 真值信号保存
# ============================
def save_ground_truth_signals(df, fs, output_path, lead_names=None):
    """保存真值信号为JSON"""
    if lead_names is None:
        lead_names = LEAD_NAMES_ORDERED
    
    gt_data = {
        'fs': int(fs),
        'signals': {},
        'durations': {},
        'lengths': {}
    }
    
    for lead_name in lead_names:
        if lead_name not in df.columns:
            gt_data['signals'][lead_name] = None
            gt_data['durations'][lead_name] = None
            gt_data['lengths'][lead_name] = 0
            continue
        
        if lead_name == 'II':
            duration = 10.0
            length = int(duration * fs)
        else:
            duration = 2.5
            length = int(duration * fs)
        
        signal = df[lead_name].iloc[:length].dropna().tolist()
        
        gt_data['signals'][lead_name] = signal
        gt_data['durations'][lead_name] = duration
        gt_data['lengths'][lead_name] = len(signal)
    
    with open(output_path, 'w') as f:
        json.dump(gt_data, f)

# ============================
# 6. 元数据构建
# ============================
def create_metadata_ultimate(ecg_id, fs, sig_len, layout_type, degradation_type,
                            physical_params, lead_rois_dict, ocr_targets,
                            M_geo, h_bed, w_bed, x_offset, y_offset, paper_color):
    """构建完整元数据"""
    metadata = {
        'ecg_id': ecg_id,
        'fs': fs,
        'sig_len': sig_len,
        'layout_type': layout_type,
        'degradation_type': degradation_type,
        
        'physical_params': {
            'paper_speed_mm_s': physical_params['paper_speed_mm_s'],
            'gain_mm_mv': physical_params['gain_mm_mv'],
            'effective_px_per_mm': physical_params['effective_px_per_mm'],
            'effective_px_per_mv': physical_params['effective_px_per_mv'],
        },
        
        'ocr_targets': {
            'paper_speed': {
                'value': physical_params['paper_speed_mm_s'],
                'bbox': ocr_targets['paper_speed'],
            },
            'gain': {
                'value': physical_params['gain_mm_mv'],
                'bbox': ocr_targets['gain'],
            },
            'calibration_pulses': ocr_targets['calibration_pulses']
        },
        
        'lead_rois': {},
        
        'image_size': {'height': h_bed, 'width': w_bed},
        'paper_offset': {'x': x_offset, 'y': y_offset},
        'paper_color_bgr': paper_color,
    }
    
    # 处理导联RoI
    for lead_name in LEAD_NAMES_ORDERED:
        if lead_name in lead_rois_dict:
            metadata['lead_rois'][lead_name] = lead_rois_dict[lead_name]
        else:
            metadata['lead_rois'][lead_name] = None
    
    # 条件保存几何变换
    if not np.allclose(M_geo, np.eye(3)):
        metadata['geometric_transform'] = M_geo.tolist()
    
    return metadata

# ============================
# 7. 单任务处理
# ============================
def process_one_id_ultimate(task_tuple, train_dir, train_meta_df, output_dir):
    """终极融合版处理函数"""
    ecg_id, var_idx = task_tuple
    ecg_id_str = str(ecg_id)
    
    try:
        csv_path = os.path.join(train_dir, ecg_id_str, f"{ecg_id_str}.csv")
        if not os.path.exists(csv_path):
            return (ecg_id_str, "csv_not_found")
        
        df = pd.read_csv(csv_path)
        
        meta_row = train_meta_df[train_meta_df['id'] == int(ecg_id)]
        if len(meta_row) == 0:
            return (ecg_id_str, "meta_not_found")
        
        fs = int(meta_row.iloc[0]['fs'])
        sig_len = int(meta_row.iloc[0]['sig_len'])
        
        # 采样布局和退化类型
        layout_type = random.choice([
            LayoutType.LAYOUT_3X4_PLUS_II,
            LayoutType.LAYOUT_3X4,
            LayoutType.LAYOUT_6X2,
            LayoutType.LAYOUT_12X1
        ])
        
        degradation_type = random.choice([
            DegradationType.CLEAN,
            DegradationType.PHOTO_PRINT,
            DegradationType.STAINED,
            DegradationType.PRINTED_COLOR,
            DegradationType.PRINTED_BW,
            DegradationType.MOLD_COLOR,
            DegradationType.MOLD_BW,
            DegradationType.DAMAGED,
            DegradationType.PHOTO_SCREEN
        ])
        
        params = sample_physical_params(layout_type)
        
        # 渲染
        (clean_img, grid_base, wave_label, text_masks, alpha_auxiliary,
         baseline_heatmaps, ps_mask, gain_mask,
         paper_color, meta_params, lead_rois, ocr_targets) = render_clean_ecg_ultimate(
            df, layout_type, params, fs, sig_len
        )
        
        # 粘贴到底板
        h_p, w_p = clean_img.shape[:2]
        h_bed = h_p + random.randint(100, 300)
        w_bed = w_p + random.randint(100, 400)
        
        bed_img = generate_scanner_background(h_bed, w_bed)
        
        x_offset = random.randint(20, w_bed - w_p - 20)
        y_offset = random.randint(20, h_bed - h_p - 20)
        
        bed_img[y_offset:y_offset+h_p, x_offset:x_offset+w_p] = clean_img
        
        # 扩展掩码
        def pad_mask(m, is_3d=False):
            if is_3d:
                res = np.zeros((m.shape[0], h_bed, w_bed), dtype=np.uint8)
                res[:, y_offset:y_offset+h_p, x_offset:x_offset+w_p] = m
            else:
                res = np.zeros((h_bed, w_bed), dtype=np.uint8)
                res[y_offset:y_offset+h_p, x_offset:x_offset+w_p] = m
            return res
        
        masks_full = {
            'wave': pad_mask(wave_label),
            'text': pad_mask(text_masks, True),
            'aux': pad_mask(alpha_auxiliary, True),
            'baseline': pad_mask(baseline_heatmaps, True),
            'ps': pad_mask(ps_mask, True),
            'gain': pad_mask(gain_mask, True)
        }
        
        # 退化
        dirty_img, M_geo, warped_masks = apply_degradation_pipeline_ultimate(
            bed_img, masks_full, degradation_type, paper_color
        )
        
        # 坐标变换
        final_ocr = {}
        for k, bbox in ocr_targets.items():
            if k == 'calibration_pulses':
                final_ocr[k] = [transform_bbox(b, (x_offset, y_offset), M_geo) 
                               for b in bbox]
            else:
                final_ocr[k] = transform_bbox(bbox, (x_offset, y_offset), M_geo)
        
        final_rois = {}
        for k, v in lead_rois.items():
            if v is None:
                final_rois[k] = None
            else:
                final_rois[k] = v.copy()
                final_rois[k]['bbox'] = transform_bbox(
                    v['bbox'], (x_offset, y_offset), M_geo
                )
                final_rois[k]['text_bbox'] = transform_bbox(
                    v['text_bbox'], (x_offset, y_offset), M_geo
                )
        
        # 保存
        var_id = f"{ecg_id_str}_v{var_idx:02d}_{layout_type}_{degradation_type}"
        save_dir = os.path.join(output_dir, var_id)
        os.makedirs(save_dir, exist_ok=True)
        
        cv2.imwrite(os.path.join(save_dir, f"{var_id}_dirty.png"), dirty_img)
        
        # 保存
        var_id = f"{ecg_id_str}_v{var_idx:02d}_{layout_type}_{degradation_type}"
        save_dir = os.path.join(output_dir, var_id)
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存脏图
        cv2.imwrite(os.path.join(save_dir, f"{var_id}_dirty.png"), dirty_img)
        
        # 保存所有NPY掩码
        np.save(os.path.join(save_dir, f"{var_id}_label_baseline.npy"),
                warped_masks['baseline'].astype(np.uint8))
        np.save(os.path.join(save_dir, f"{var_id}_label_text_multi.npy"),
                warped_masks['text'].astype(np.uint8))
        np.save(os.path.join(save_dir, f"{var_id}_label_wave.npy"),
                warped_masks['wave'].astype(np.uint8))
        np.save(os.path.join(save_dir, f"{var_id}_label_auxiliary.npy"),
                warped_masks['aux'].astype(np.uint8))
        np.save(os.path.join(save_dir, f"{var_id}_label_paper_speed.npy"),
                warped_masks['ps'].astype(np.uint8))
        np.save(os.path.join(save_dir, f"{var_id}_label_gain.npy"),
                warped_masks['gain'].astype(np.uint8))
        
        # 保存真值信号JSON
        gt_signals_path = os.path.join(save_dir, f"{var_id}_gt_signals.json")
        save_ground_truth_signals(df, fs, gt_signals_path)
        
        # 保存元数据JSON
        metadata = create_metadata_ultimate(
            ecg_id_str, fs, sig_len, layout_type, degradation_type,
            meta_params, final_rois, final_ocr,
            M_geo, h_bed, w_bed, x_offset, y_offset, paper_color
        )
        
        with open(os.path.join(save_dir, f"{var_id}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return (var_id, "success")
        
    except Exception as e:
        import traceback
        return (ecg_id_str, f"Error: {str(e)}\n{traceback.format_exc()}")

# ============================
# 8. 验证工具
# ============================
def validate_sample_output(sample_dir):
    """验证单个样本的输出"""
    from pathlib import Path
    
    sample_dir = Path(sample_dir)
    sample_id = sample_dir.name
    
    report = {
        'sample_id': sample_id,
        'passed': True,
        'errors': [],
        'warnings': []
    }
    
    # 检查必需文件
    required_files = [
        f"{sample_id}_dirty.png",
        f"{sample_id}_label_baseline.npy",
        f"{sample_id}_label_text_multi.npy",
        f"{sample_id}_label_wave.npy",
        f"{sample_id}_label_auxiliary.npy",
        f"{sample_id}_label_paper_speed.npy",
        f"{sample_id}_label_gain.npy",
        f"{sample_id}_metadata.json",
        f"{sample_id}_gt_signals.json"
    ]
    
    for fname in required_files:
        fpath = sample_dir / fname
        if not fpath.exists():
            report['passed'] = False
            report['errors'].append(f"Missing file: {fname}")
    
    # 验证文字掩码
    text_mask_path = sample_dir / f"{sample_id}_label_text_multi.npy"
    if text_mask_path.exists():
        text_mask = np.load(text_mask_path)
        
        if text_mask.shape[0] != 13:
            report['passed'] = False
            report['errors'].append(f"Text mask has {text_mask.shape[0]} channels, expected 13")
        
        if text_mask[0].sum() == 0:
            report['warnings'].append("Background channel (0) is empty")
        
        for i in range(1, 13):
            if text_mask[i].sum() == 0:
                report['warnings'].append(f"Lead channel {i} is empty")
    
    # 验证辅助掩码
    aux_path = sample_dir / f"{sample_id}_label_auxiliary.npy"
    if aux_path.exists():
        aux_mask = np.load(aux_path)
        
        left_region = aux_mask[0, :, :aux_mask.shape[2]//10]
        if left_region.sum() < 100:
            report['warnings'].append("Auxiliary mask may be missing calibration pulse")
    
    # 验证真值信号
    gt_signals_path = sample_dir / f"{sample_id}_gt_signals.json"
    if gt_signals_path.exists():
        with open(gt_signals_path, 'r') as f:
            gt_data = json.load(f)
        
        num_valid_leads = sum(1 for v in gt_data['signals'].values() if v is not None)
        if num_valid_leads < 10:
            report['warnings'].append(f"Only {num_valid_leads}/12 leads have valid signals")
    
    return report

# ============================
# 9. 主程序
# ============================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ECG Simulator V46 - Ultimate Fusion Version')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of ECG IDs to process')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode (single process)')
    parser.add_argument('--variations', type=int, default=3,
                       help='Number of variations per ECG ID')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation on generated samples')
    args = parser.parse_args()
    
    # 配置路径
    BASE_DATA_DIR = "/Volumes/movie/work/physionet-ecg-image-digitization"
    OUTPUT_DIR = "/Volumes/movie/work/physionet-ecg-image-digitization-simulations-V46-Ultimate"
    TRAIN_CSV = os.path.join(BASE_DATA_DIR, "train.csv")
    TRAIN_DIR = os.path.join(BASE_DATA_DIR, "train")
    
    # 验证路径
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: train.csv not found at {TRAIN_CSV}")
        sys.exit(1)
    
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: train directory not found at {TRAIN_DIR}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载元数据
    df = pd.read_csv(TRAIN_CSV)
    ids = df['id'].tolist()
    
    if args.limit:
        ids = ids[:args.limit]
        print(f"Limited to first {args.limit} ECG IDs")
    
    # 生成任务列表
    tasks = []
    for ecg_id in ids:
        for var_idx in range(args.variations):
            tasks.append((ecg_id, var_idx))
    
    print(f"=" * 70)
    print(f"ECG Simulator V46 - Ultimate Fusion Version")
    print(f"=" * 70)
    print(f"Total ECG IDs: {len(ids)}")
    print(f"Variations per ID: {args.variations}")
    print(f"Total samples to generate: {len(tasks)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Workers: {args.workers if not args.debug else 1}")
    print(f"=" * 70)
    print("\nKey Features:")
    print("  ✅ V38 High-quality rendering (separators, textures)")
    print("  ✅ V46 Complete annotations (13-layer text, pulses)")
    print("  ✅ 100% compatible with progressive model architecture")
    print("  ✅ Position and font randomness")
    print("  ✅ Ground truth signals saved")
    print(f"=" * 70)
    
    # 处理任务
    if args.debug:
        print("\nRunning in DEBUG mode...")
        results = []
        for task in tqdm(tasks, desc="Processing"):
            result = process_one_id_ultimate(task, TRAIN_DIR, df, OUTPUT_DIR)
            results.append(result)
            
            if result[1] != "success":
                print(f"\nError processing {result[0]}: {result[1]}")
    else:
        print(f"\nRunning with {args.workers} workers...")
        with multiprocessing.Pool(args.workers) as pool:
            func = partial(process_one_id_ultimate,
                          train_dir=TRAIN_DIR,
                          train_meta_df=df,
                          output_dir=OUTPUT_DIR)
            
            results = []
            for result in tqdm(pool.imap_unordered(func, tasks),
                             total=len(tasks),
                             desc="Processing"):
                results.append(result)
                
                if result[1] != "success":
                    print(f"\nError processing {result[0]}: {result[1]}")
    
    # 统计结果
    success_count = sum(1 for r in results if r[1] == "success")
    error_count = len(results) - success_count
    
    print(f"\n" + "=" * 70)
    print(f"Processing complete!")
    print(f"Successful: {success_count}/{len(results)}")
    print(f"Errors: {error_count}/{len(results)}")
    print(f"=" * 70)
    
    # 自动验证
    if args.validate and success_count > 0:
        print("\nRunning validation...")
        
        from pathlib import Path
        sample_dirs = [d for d in Path(OUTPUT_DIR).iterdir() if d.is_dir()]
        
        if len(sample_dirs) > 10:
            sample_dirs = random.sample(sample_dirs, 10)
        
        validation_reports = []
        for sample_dir in sample_dirs:
            report = validate_sample_output(sample_dir)
            validation_reports.append(report)
        
        passed = sum(1 for r in validation_reports if r['passed'])
        print(f"\nValidation Results:")
        print(f"  Passed: {passed}/{len(validation_reports)}")
        
        for report in validation_reports:
            if not report['passed']:
                print(f"\n  Sample {report['sample_id']}:")
                for err in report['errors']:
                    print(f"    ❌ {err}")
            
            if report['warnings']:
                print(f"\n  Sample {report['sample_id']}:")
                for warn in report['warnings']:
                    print(f"    ⚠️  {warn}")
    
    print("\n" + "=" * 70)
    print("🎉 All done! Check the output directory for results.")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)