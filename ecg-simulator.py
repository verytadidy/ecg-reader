import numpy as np
import pandas as pd
import cv2
import os
import random
import multiprocessing
from functools import partial
from tqdm import tqdm
import math
import json
import sys

# ============================
# 1. 定义 (不变)
# ============================
class DegradationType:
    CLEAN = "0001"; PRINTED_COLOR = "0003"; PRINTED_BW = "0004"
    PHOTO_PRINT = "0005"; PHOTO_SCREEN = "0006"; STAINED = "0009"
    DAMAGED = "0010"; MOLD_COLOR = "0011"; MOLD_BW = "0012"

class LayoutType:
    LAYOUT_3X4_PLUS_II = "3x4+1"; LAYOUT_3X4 = "3x4"
    LAYOUT_6X2 = "6x2"; LAYOUT_12X1 = "12x1"

LAYOUT_CONFIGS = {
    LayoutType.LAYOUT_3X4_PLUS_II: {
        'leads': {'I':(0,0), 'aVR':(0,1), 'V1':(0,2), 'V4':(0,3),
                  'II':(1,0), 'aVL':(1,1), 'V2':(1,2), 'V5':(1,3),
                  'III':(2,0), 'aVF':(2,1), 'V3':(2,2), 'V6':(2,3)}, 
        'long_lead': 'II', 'rows': 3, 'cols': 4
    },
    LayoutType.LAYOUT_3X4: {
        'leads': {'I':(0,0), 'aVR':(0,1), 'V1':(0,2), 'V4':(0,3),
                  'II':(1,0), 'aVL':(1,1), 'V2':(1,2), 'V5':(1,3),
                  'III':(2,0), 'aVF':(2,1), 'V3':(2,2), 'V6':(2,3)}, 
        'long_lead': None, 'rows': 3, 'cols': 4
    },
    LayoutType.LAYOUT_6X2: {
        'leads': {'I':(0,0), 'II':(1,0), 'III':(2,0), 'aVR':(3,0), 'aVL':(4,0), 'aVF':(5,0),
                  'V1':(0,1), 'V2':(1,1), 'V3':(2,1), 'V4':(3,1), 'V5':(4,1), 'V6':(5,1)}, 
        'long_lead': None, 'rows': 6, 'cols': 2
    },
    LayoutType.LAYOUT_12X1: {
        'leads': {'I':(0,0), 'II':(1,0), 'III':(2,0), 'aVR':(3,0), 'aVL':(4,0), 'aVF':(5,0),
                  'V1':(6,0), 'V2':(7,0), 'V3':(8,0), 'V4':(9,0), 'V5':(10,0), 'V6':(11,0)}, 
        'long_lead': None, 'rows': 12, 'cols': 1
    }
}

# 🔥============================
# 🔥 1.1 新增：导联到ID的映射
# 🔥============================
LEAD_TO_ID_MAP = {
    'I': 1, 'II': 2, 'III': 3,
    'aVR': 4, 'aVL': 5, 'aVF': 6,
    'V1': 7, 'V2': 8, 'V3': 9,
    'V4': 10, 'V5': 11, 'V6': 12
}


# ============================
# 2. 颜色, 纹理, & 字体
# ============================
COLOR_GRID_MINOR_BASE_OPTIONS = [(180, 160, 255), (170, 150, 255)]
COLOR_GRID_MAJOR_BASE_OPTIONS = [(160, 140, 255), (150, 130, 255)]

COLOR_STAIN = (120, 190, 220)
COLOR_TEXT_HEADER = (10, 10, 10)
COLOR_TEXT_FOOTER = (0, 0, 0)
FONT_LIST = [
    cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL
]

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
    base_color = np.array(color, dtype=np.float32)
    texture = np.full((h, w, 3), base_color, dtype=np.float32)
    
    fiber_noise = np.random.normal(0, 0.8, (h//4, w//4, 3))
    fiber_noise = cv2.resize(fiber_noise, (w, h), interpolation=cv2.INTER_LINEAR)
    grain = np.random.normal(0, 0.3, (h, w, 3))
    X, Y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    pattern = np.sin(X * 50) * np.cos(Y * 50) * 0.3
    texture += fiber_noise + grain + pattern[..., None]
    texture = np.clip(texture, 0, 255).astype(np.uint8)
    
    if grid_img is not None:
        line_mask = ((grid_img[:,:,2] > 200) & (grid_img[:,:,1] < 220) & (grid_img[:,:,0] < 220)).astype(np.uint8) * 255
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
        line_mask_f = line_mask.astype(np.float32) / 255.0
        line_mask_f = line_mask_f[..., None]
        texture = (texture.astype(np.float32) * (1 - line_mask_f) + grid_img.astype(np.float32) * line_mask_f).astype(np.uint8)
    
    light_noise = np.random.normal(0, 0.1, (h, w, 3))
    texture = np.clip(texture.astype(np.float32) + light_noise, 0, 255).astype(np.uint8)
    return texture

def generate_scanner_background(h, w):
    mode = random.choice(['dark_gray', 'black', 'wood'])
    if mode == 'dark_gray': 
        color = random_color_variations((40, 40, 40), 10)
    elif mode == 'black': 
        color = random_color_variations((5, 5, 5), 5)
    else: 
        color = random_color_variations((50, 80, 120), 20)
    return generate_paper_texture(h, w, color)

# ============================
# 3. 物理参数 (V37 修正)
# ============================
def sample_physical_params_v37(layout_type):
    """
    V39 调整：
    - 6x2 布局的增益分布被调整，以降低平均增益。
    """
    if layout_type in [LayoutType.LAYOUT_3X4_PLUS_II, LayoutType.LAYOUT_3X4]:
        gain_mm_mv = random.choice([5.0, 10.0])
    
    elif layout_type == LayoutType.LAYOUT_12X1:
        # 真实临床分布：
        # - 70% 使用 5mm/mV（半增益，防止重叠）
        # - 15% 使用 10mm/mV（标准增益，可能轻微重叠）
        # - 15% 使用 2.5mm/mV（对极端高幅度心电）
        r = random.random()
        if r < 0.70:
            gain_mm_mv = 5.0
        elif r < 0.85:
            gain_mm_mv = 10.0  # 允许部分重叠，符合真实情况
        else:
            gain_mm_mv = 2.5
            
    # 🔥 V39 调整：为 6x2 布局设置独立的加权增益
    elif layout_type == LayoutType.LAYOUT_6X2:
        # 6x2 布局通常使用 10 或 5 mm/mV。20 mm/mV (双倍增益) 较少见。
        # 50% 10mm/mV (标准)
        # 40% 5mm/mV (半增益)
        # 10% 20mm/mV (双倍增益)
        gain_mm_mv = random.choices([10.0, 5.0, 20.0], weights=[0.50, 0.40, 0.10], k=1)[0]
        
    else:
        # 默认/回退 (等同于 3x4)
        gain_mm_mv = random.choice([5.0, 10.0])
    
    paper_speed_mm_s = random.choice([25.0, 50.0])
    px_per_mm = random.uniform(18.0, 22.0)
    
    return {
        'paper_speed_mm_s': paper_speed_mm_s,
        'gain_mm_mv': gain_mm_mv,
        'px_per_mm': px_per_mm,
        'lead_durations': {
            'long': 10.0,
            'short': 2.5
        }
    }

# ============================
# 4. 绘图子模块
# ============================
def render_calibration_pulse(img, alpha_other, x_start, y_baseline, px_per_mm, px_per_mv, paper_speed_mm_s, ink_color, thick):
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
    cv2.polylines(img, [pts], False, ink_color, thick, cv2.LINE_AA)
    cv2.polylines(alpha_other, [pts], False, 255, thick, cv2.LINE_AA)
    return x_end

# 🔥 修改： 'alpha_waveform' -> 'wave_label_semantic_mask'
def render_layout_3x4_v37(df, sig_rgb, wave_label_semantic_mask, alpha_other, alpha_baseline, h, w, params, ink_color, font_face, fs, sig_len):
    """V38 新增：3x4 纯网格布局（不带底部长导联）"""
    px_per_mm = params['px_per_mm']
    px_per_mv = params['gain_mm_mv'] * px_per_mm
    paper_speed = params['paper_speed_mm_s']
    MT = int(150 * (h/1700))
    MB = int(100 * (h/1700))
    ML = int(10*px_per_mm)
    MR = int(10*px_per_mm)
    
    lead_in_area = int(random.uniform(12.0, 18.0) * px_per_mm)
    signal_start_x = ML + lead_in_area
    signal_draw_w = w - signal_start_x - MR
    
    PAPER_DURATION_S = 10.0
    px_per_s_on_paper = signal_draw_w / PAPER_DURATION_S
    
    # 🔥 关键区别：不分配空间给长导联，3行平分整个高度
    main_h = h - MT - MB  # 使用全部可用高度
    row_h = main_h / 3    # 3行平分
    TIME_PER_COL_ON_PAPER = 2.5
    
    thick_signal = random.randint(1, 2)
    thick_pulse = thick_signal + 1
    thick_separator = thick_pulse + 1
    font_scale = random.uniform(0.9, 1.2)
    x_pulse_start_common = int(signal_start_x - random.uniform(10.0, 12.0) * px_per_mm)
    
    # 绘制 3 个定标脉冲
    x_pulse_end_main_grid = 0
    for r in range(3):
        base_y = int(MT + (r + 0.5) * row_h)
        _x_end = render_calibration_pulse(sig_rgb, alpha_other, x_pulse_start_common, base_y, px_per_mm, px_per_mv, paper_speed, ink_color, thick_pulse)
        x_pulse_end_main_grid = max(x_pulse_end_main_grid, _x_end)
    
    total_samples_10s = min(len(df), int(fs * 10.0))
    
    # 绘制 12 个短导联（与 3x4+1 相同）
    for lead, (r, c) in LAYOUT_CONFIGS[LayoutType.LAYOUT_3X4]['leads'].items():
        if lead not in df.columns: 
            continue
        base_y = int(MT + (r + 0.5) * row_h)
        
        t_start_plot = c * TIME_PER_COL_ON_PAPER
        t_end_plot = (c + 1) * TIME_PER_COL_ON_PAPER
        
        idx_start = int(t_start_plot * fs)
        idx_end = min(int(t_end_plot * fs), total_samples_10s)
        
        sig = df[lead].iloc[idx_start:idx_end].dropna().values
        
        x_start_line = int(signal_start_x + t_start_plot * px_per_s_on_paper)
        x_end_line = int(signal_start_x + t_end_plot * px_per_s_on_paper)
        cv2.line(alpha_baseline, (x_start_line, base_y), (x_end_line, base_y), 255, thick_signal, cv2.LINE_AA)
        
        if len(sig) > 0:
            t_axis_plot = np.linspace(t_start_plot, t_end_plot, len(sig))
            xs = signal_start_x + t_axis_plot * px_per_s_on_paper
            ys = base_y - sig * px_per_mv
            xs = np.clip(xs, 0, w - 1)
            pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(sig_rgb, [pts], False, ink_color, thick_signal, cv2.LINE_AA)
            
            # 🔥 修改：使用 'lead_id' 绘制到 'wave_label_semantic_mask'
            if lead in LEAD_TO_ID_MAP:
                lead_id = LEAD_TO_ID_MAP[lead]
                cv2.polylines(wave_label_semantic_mask, [pts], False, lead_id, thick_signal, cv2.LINE_AA)
        
        txt_y = int(base_y - row_h * 0.3)
        txt_x_gap_mm = random.uniform(2.0, 5.0)
        txt_x_base = int(x_pulse_end_main_grid + txt_x_gap_mm * px_per_mm)
        if c == 0: 
            txt_x = txt_x_base
        else: 
            txt_x = int(signal_start_x + (c * TIME_PER_COL_ON_PAPER) * px_per_s_on_paper + random.uniform(2, 4)*px_per_mm)
        txt_x = max(0, min(txt_x, w - 1))
        txt_y = max(10, min(txt_y, h - 1))
        cv2.putText(sig_rgb, lead, (txt_x, txt_y), font_face, font_scale, ink_color, 2, cv2.LINE_AA)
        cv2.putText(alpha_other, lead, (txt_x, txt_y), font_face, font_scale, 255, 2, cv2.LINE_AA)
    
    # 列分隔符
    tick_h_half = int(2.5 * px_per_mm)
    # 50% 几率悬浮, 50% 几率在基线
    separator_style = random.choice(['centered', 'floating']) 
    
    for c in range(1, 4):
        sep_x = int(signal_start_x + (c * TIME_PER_COL_ON_PAPER) * px_per_s_on_paper)
        for r in range(3):
            base_y = int(MT + (r + 0.5) * row_h)
            
            # 🔥 V39 调整：根据风格计算y坐标
            if separator_style == 'centered':
                y_center = base_y
            else: # 'floating'
                # 放置在行高顶部 25% 的位置 (靠近文字标签)
                y_center = int(MT + (r * row_h) + row_h * 0.25) 
            
            y1 = y_center - tick_h_half
            y2 = y_center + tick_h_half
                
            pts_tick = np.array([[sep_x, y1], [sep_x, y2]], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(sig_rgb, [pts_tick], False, ink_color, thick_separator, cv2.LINE_AA)
            cv2.polylines(alpha_other, [pts_tick], False, 255, thick_separator, cv2.LINE_AA)

# 🔥 修改： 'alpha_waveform' -> 'wave_label_semantic_mask'
def render_layout_3x4_plus_II_v37(df, sig_rgb, wave_label_semantic_mask, alpha_other, alpha_baseline, h, w, params, ink_color, font_face, fs, sig_len):
    """V37 修正：使用真实 fs 和 sig_len"""
    px_per_mm = params['px_per_mm']
    px_per_mv = params['gain_mm_mv'] * px_per_mm
    paper_speed = params['paper_speed_mm_s']
    MT = int(150 * (h/1700))
    MB = int(100 * (h/1700))
    ML = int(10*px_per_mm)
    MR = int(10*px_per_mm)
    
    lead_in_area = int(random.uniform(12.0, 18.0) * px_per_mm)
    signal_start_x = ML + lead_in_area
    signal_draw_w = w - signal_start_x - MR
    
    PAPER_DURATION_S = 10.0
    px_per_s_on_paper = signal_draw_w / PAPER_DURATION_S
    
    main_h = (h - MT - MB) * 0.75
    rhythm_h = (h - MT - MB) * 0.25
    row_h = main_h / 3
    TIME_PER_COL_ON_PAPER = 2.5  # 固定 2.5 秒每列
    
    thick_signal = random.randint(1, 2)
    thick_pulse = thick_signal + 1
    thick_separator = thick_pulse + 1
    font_scale = random.uniform(0.9, 1.2)
    x_pulse_start_common = int(signal_start_x - random.uniform(10.0, 12.0) * px_per_mm)
    
    # 定标脉冲
    x_pulse_end_main_grid = 0
    for r in range(3):
        base_y = int(MT + (r + 0.5) * row_h)
        _x_end = render_calibration_pulse(sig_rgb, alpha_other, x_pulse_start_common, base_y, px_per_mm, px_per_mv, paper_speed, ink_color, thick_pulse)
        x_pulse_end_main_grid = max(x_pulse_end_main_grid, _x_end)
    base_y_long_lead = int(MT + main_h + rhythm_h / 2)
    x_pulse_end_long_lead = render_calibration_pulse(sig_rgb, alpha_other, x_pulse_start_common, base_y_long_lead, px_per_mm, px_per_mv, paper_speed, ink_color, thick_pulse)
    
    # V37 修正：使用真实 fs
    total_samples_10s = min(len(df), int(fs * 10.0))
    
    # 绘制 12 个短导联
    for lead, (r, c) in LAYOUT_CONFIGS[LayoutType.LAYOUT_3X4_PLUS_II]['leads'].items():
        if lead not in df.columns: 
            continue
        base_y = int(MT + (r + 0.5) * row_h)
        
        t_start_plot = c * TIME_PER_COL_ON_PAPER
        t_end_plot = (c + 1) * TIME_PER_COL_ON_PAPER
        
        # V37 修正：基于真实 fs 计算索引
        idx_start = int(t_start_plot * fs)
        idx_end = min(int(t_end_plot * fs), total_samples_10s)
        
        sig = df[lead].iloc[idx_start:idx_end].dropna().values
        
        x_start_line = int(signal_start_x + t_start_plot * px_per_s_on_paper)
        x_end_line = int(signal_start_x + t_end_plot * px_per_s_on_paper)
        cv2.line(alpha_baseline, (x_start_line, base_y), (x_end_line, base_y), 255, thick_signal, cv2.LINE_AA)
        
        if len(sig) > 0:
            t_axis_plot = np.linspace(t_start_plot, t_end_plot, len(sig))
            xs = signal_start_x + t_axis_plot * px_per_s_on_paper
            ys = base_y - sig * px_per_mv
            xs = np.clip(xs, 0, w - 1)
            pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(sig_rgb, [pts], False, ink_color, thick_signal, cv2.LINE_AA)

            # 🔥 修改：使用 'lead_id' 绘制到 'wave_label_semantic_mask'
            if lead in LEAD_TO_ID_MAP:
                lead_id = LEAD_TO_ID_MAP[lead]
                cv2.polylines(wave_label_semantic_mask, [pts], False, lead_id, thick_signal, cv2.LINE_AA)
        
        txt_y = int(base_y - row_h * 0.3)
        txt_x_gap_mm = random.uniform(2.0, 5.0)
        txt_x_base = int(x_pulse_end_main_grid + txt_x_gap_mm * px_per_mm)
        if c == 0: 
            txt_x = txt_x_base
        else: 
            txt_x = int(signal_start_x + (c * TIME_PER_COL_ON_PAPER) * px_per_s_on_paper + random.uniform(2, 4)*px_per_mm)
        txt_x = max(0, min(txt_x, w - 1))
        txt_y = max(10, min(txt_y, h - 1))
        cv2.putText(sig_rgb, lead, (txt_x, txt_y), font_face, font_scale, ink_color, 2, cv2.LINE_AA)
        cv2.putText(alpha_other, lead, (txt_x, txt_y), font_face, font_scale, 255, 2, cv2.LINE_AA)
    
    # 🔥 ：列分隔符
    tick_h_half = int(2.5 * px_per_mm)
    # 50% 几率悬浮, 50% 几率在基线
    separator_style = random.choice(['centered', 'floating'])
    
    for c in range(1, 4):
        sep_x = int(signal_start_x + (c * TIME_PER_COL_ON_PAPER) * px_per_s_on_paper)
        for r in range(3):
            base_y = int(MT + (r + 0.5) * row_h)

            # 🔥 V39 调整：根据风格计算y坐标
            if separator_style == 'centered':
                y_center = base_y
            else: # 'floating'
                # 放置在行高顶部 25% 的位置 (靠近文字标签)
                y_center = int(MT + (r * row_h) + row_h * 0.25)
            
            y1 = y_center - tick_h_half
            y2 = y_center + tick_h_half
                
            pts_tick = np.array([[sep_x, y1], [sep_x, y2]], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(sig_rgb, [pts_tick], False, ink_color, thick_separator, cv2.LINE_AA)
            cv2.polylines(alpha_other, [pts_tick], False, 255, thick_separator, cv2.LINE_AA)
    
    # 长导联 (Lead II, 10秒)
    long_lead = LAYOUT_CONFIGS[LayoutType.LAYOUT_3X4_PLUS_II]['long_lead']
    if long_lead and long_lead in df.columns:
        base_y = int(MT + main_h + rhythm_h / 2)
        txt_x = int(x_pulse_end_long_lead + random.uniform(2, 4) * px_per_mm)
        txt_y = int(base_y - rhythm_h * 0.3)
        txt_x = max(0, min(txt_x, w - 1))
        txt_y = max(10, min(txt_y, h - 1))
        cv2.putText(sig_rgb, long_lead, (txt_x, txt_y), font_face, font_scale, ink_color, 2, cv2.LINE_AA)
        cv2.putText(alpha_other, long_lead, (txt_x, txt_y), font_face, font_scale, 255, 2, cv2.LINE_AA)
        cv2.line(alpha_baseline, (signal_start_x, base_y), (signal_start_x + signal_draw_w, base_y), 255, thick_signal, cv2.LINE_AA)
        
        # V37 修正：使用真实 fs
        idx_start = 0
        idx_end = min(int(10.0 * fs), total_samples_10s)
        sig_full = df[long_lead].iloc[idx_start:idx_end].dropna().values
        if len(sig_full) > 0:
            t_axis_plot = np.linspace(0, 10.0, len(sig_full))
            xs = signal_start_x + t_axis_plot * px_per_s_on_paper
            ys = base_y - sig_full * px_per_mv
            xs = np.clip(xs, 0, w - 1)
            pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(sig_rgb, [pts], False, ink_color, thick_signal, cv2.LINE_AA)
            
            # 🔥 修改：使用 'lead_id' 绘制到 'wave_label_semantic_mask' (长导联)
            if long_lead in LEAD_TO_ID_MAP:
                lead_id = LEAD_TO_ID_MAP[long_lead]
                cv2.polylines(wave_label_semantic_mask, [pts], False, lead_id, thick_signal, cv2.LINE_AA)

# 🔥 修改： 'alpha_waveform' -> 'wave_label_semantic_mask'
def render_layout_6x2_v37(df, sig_rgb, wave_label_semantic_mask, alpha_other, alpha_baseline, h, w, params, ink_color, font_face, fs, sig_len):
    """V38 修复版本：
    1. 添加列分隔符（在第2列开始前）
    2. 修复导联名称位置，让它们分别显示在各自导联附近
    3. 引入位置随机化
    """
    px_per_mm = params['px_per_mm']
    px_per_mv = params['gain_mm_mv'] * px_per_mm
    paper_speed = params['paper_speed_mm_s']
    MT = int(100 * (h/1700))
    MB = int(100 * (h/1700))
    ML = int(10*px_per_mm)
    MR = int(10*px_per_mm)
    
    lead_in_area = int(random.uniform(12.0, 18.0) * px_per_mm)
    signal_start_x = ML + lead_in_area
    signal_draw_w = w - signal_start_x - MR
    PAPER_DURATION_S = 10.0
    px_per_s_on_paper = signal_draw_w / PAPER_DURATION_S
    row_h = (h - MT - MB) / 6.0
    col_w = signal_draw_w / 2.0
    TIME_PER_COL_ON_PAPER = 5.0
    
    thick_signal = random.randint(1, 2)
    thick_pulse = thick_signal + 1
    thick_separator = thick_pulse + 1
    font_scale = random.uniform(1.0, 1.3)
    
    x_pulse_start_common = int(signal_start_x - random.uniform(10.0, 12.0) * px_per_mm)
    
    # 每行都绘制定标脉冲
    x_pulse_end_max = 0
    for r in range(6):
        base_y = int(MT + (r + 0.5) * row_h)
        x_pulse_end = render_calibration_pulse(
            sig_rgb, alpha_other, x_pulse_start_common, base_y, 
            px_per_mm, px_per_mv, paper_speed, ink_color, thick_pulse
        )
        x_pulse_end_max = max(x_pulse_end_max, x_pulse_end)
    
    # 列分隔符
    sep_x = int(signal_start_x + TIME_PER_COL_ON_PAPER * px_per_s_on_paper)
    tick_h_half = int(2.5 * px_per_mm)
    # 50% 几率悬浮, 50% 几率在基线
    separator_style = random.choice(['centered', 'floating'])
    
    for r in range(6):
        base_y = int(MT + (r + 0.5) * row_h)
        
        #  调整：根据风格计算y坐标
        if separator_style == 'centered':
            y_center = base_y
        else: # 'floating'
            # 放置在行高顶部 25% 的位置 (靠近文字标签)
            y_center = int(MT + (r * row_h) + row_h * 0.25)
        
        y1 = y_center - tick_h_half
        y2 = y_center + tick_h_half
            
        pts_tick = np.array([
            [sep_x, y1], 
            [sep_x, y2]
        ], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(sig_rgb, [pts_tick], False, ink_color, thick_separator, cv2.LINE_AA)
        cv2.polylines(alpha_other, [pts_tick], False, 255, thick_separator, cv2.LINE_AA)
    
    # 获取真实信号长度
    total_samples_10s = min(len(df), int(fs * 10.0))
    
    # 绘制12个导联
    for lead, (r, c) in LAYOUT_CONFIGS[LayoutType.LAYOUT_6X2]['leads'].items():
        if lead not in df.columns: 
            continue
        
        base_y = int(MT + (r + 0.5) * row_h)
        t_start_plot = c * TIME_PER_COL_ON_PAPER
        t_end_plot = (c + 1) * TIME_PER_COL_ON_PAPER
        
        idx_start = int(t_start_plot * fs)
        idx_end = min(int(t_end_plot * fs), total_samples_10s)
        sig = df[lead].iloc[idx_start:idx_end].dropna().values
        
        # 计算该导联所在列的实际绘制范围
        x_start_line = int(signal_start_x + c * col_w)
        x_end_line = int(signal_start_x + (c + 1) * col_w)
        cv2.line(alpha_baseline, (x_start_line, base_y), (x_end_line, base_y), 255, thick_signal, cv2.LINE_AA)
        
        # 绘制波形
        if len(sig) > 0:
            t_axis_plot = np.linspace(t_start_plot, t_end_plot, len(sig))
            xs = signal_start_x + t_axis_plot * px_per_s_on_paper
            ys = base_y - sig * px_per_mv
            xs = np.clip(xs, 0, w - 1)
            pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(sig_rgb, [pts], False, ink_color, thick_signal, cv2.LINE_AA)

            # 🔥 修改：使用 'lead_id' 绘制到 'wave_label_semantic_mask'
            if lead in LEAD_TO_ID_MAP:
                lead_id = LEAD_TO_ID_MAP[lead]
                cv2.polylines(wave_label_semantic_mask, [pts], False, lead_id, thick_signal, cv2.LINE_AA)
        
        # 🔥 修复2: 导联标签位置优化
        # 垂直位置：在该行上方，增加随机偏移
        txt_y_base = int(base_y - row_h * 0.25)
        txt_y_offset = random.uniform(-row_h * 0.05, row_h * 0.05)  # 上下浮动5%行高
        txt_y = int(txt_y_base + txt_y_offset)
        
        # 水平位置：根据所在列决定
        if c == 0:
            # 第一列：定标脉冲后方
            txt_x_base = int(x_pulse_end_max + random.uniform(2.0, 4.0) * px_per_mm)
            txt_x_offset = random.uniform(0, 2.0 * px_per_mm)  # 向右随机偏移
        else:
            # 第二列：该列信号开始后一段距离
            txt_x_base = int(signal_start_x + c * col_w + random.uniform(2.0, 5.0) * px_per_mm)
            txt_x_offset = random.uniform(0, 2.0 * px_per_mm)  # 向右随机偏移
        
        txt_x = int(txt_x_base + txt_x_offset)
        
        # 确保坐标在有效范围内
        txt_x = max(0, min(txt_x, w - 50))  # 留出文字宽度
        txt_y = max(10, min(txt_y, h - 10))
        
        # 绘制导联标签
        cv2.putText(sig_rgb, lead, (txt_x, txt_y), font_face, font_scale, ink_color, 2, cv2.LINE_AA)
        cv2.putText(alpha_other, lead, (txt_x, txt_y), font_face, font_scale, 255, 2, cv2.LINE_AA)

# 🔥 修改： 'alpha_waveform' -> 'wave_label_semantic_mask'
def render_layout_12x1_v37(df, sig_rgb, wave_label_semantic_mask, alpha_other, alpha_baseline, h, w, params, ink_color, font_face, fs, sig_len):
    """
    V38 真实版本：
    - 不额外缩放波形（临床上通过调整 mm/mV 而非事后缩放）
    - 允许极端情况下波形与上下行轻微重叠（真实场景）
    - 定标脉冲使用实际增益值（不缩放）
    """
    px_per_mm = params['px_per_mm']
    px_per_mv = params['gain_mm_mv'] * px_per_mm
    paper_speed = params['paper_speed_mm_s']
    MT = int(100 * (h/1700))
    MB = int(100 * (h/1700))
    ML = int(10*px_per_mm)
    MR = int(10*px_per_mm)
    
    lead_in_area = int(random.uniform(12.0, 18.0) * px_per_mm)
    signal_start_x = ML + lead_in_area
    signal_draw_w = w - signal_start_x - MR
    row_h = (h - MT - MB) / 12.0
    PAPER_DURATION_S = 10.0
    px_per_s_on_paper = signal_draw_w / PAPER_DURATION_S
    
    thick_signal = random.randint(1, 2)
    thick_pulse = thick_signal + 1
    font_scale = random.uniform(0.8, 1.0)
    x_pulse_start_common = int(signal_start_x - random.uniform(10.0, 12.0) * px_per_mm)
    
    # 🔥 关键：定标脉冲使用实际增益（不缩放）
    x_pulse_end_max = 0
    for r in range(12):
        base_y = int(MT + (r + 0.5) * row_h)
        x_pulse_end = render_calibration_pulse(
            sig_rgb, alpha_other, x_pulse_start_common, base_y, 
            px_per_mm, px_per_mv, paper_speed, ink_color, thick_pulse
        )
        x_pulse_end_max = max(x_pulse_end_max, x_pulse_end)
    
    
    total_samples_10s = min(len(df), int(fs * 10.0))
    lead_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # 🔥 关键：波形也使用实际增益（允许重叠）
    for r, lead in enumerate(lead_order):
        if lead not in df.columns: 
            continue
        base_y = int(MT + (r + 0.5) * row_h)
        
        t_start_plot = 0.0
        t_end_plot = 10.0
        
        idx_start = 0
        idx_end = total_samples_10s
        sig = df[lead].iloc[idx_start:idx_end].dropna().values
        
        # 绘制基线
        x_start_line = signal_start_x
        x_end_line = signal_start_x + signal_draw_w
        cv2.line(alpha_baseline, (x_start_line, base_y), (x_end_line, base_y), 255, thick_signal, cv2.LINE_AA)
        
        # 绘制波形（使用实际增益，不额外缩放）
        if len(sig) > 0:
            t_axis_plot = np.linspace(t_start_plot, t_end_plot, len(sig))
            xs = signal_start_x + t_axis_plot * px_per_s_on_paper
            ys = base_y - sig * px_per_mv  # 🔥 使用原始增益，不缩放
            xs = np.clip(xs, 0, w - 1)
            # ⚠️ 不裁剪 ys，允许波形超出行高范围（真实情况）
            pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(sig_rgb, [pts], False, ink_color, thick_signal, cv2.LINE_AA)
            
            # 🔥 修改：使用 'lead_id' 绘制到 'wave_label_semantic_mask'
            if lead in LEAD_TO_ID_MAP:
                lead_id = LEAD_TO_ID_MAP[lead]
                cv2.polylines(wave_label_semantic_mask, [pts], False, lead_id, thick_signal, cv2.LINE_AA)
        
        # 绘制导联标签
        txt_y = int(base_y - row_h * 0.2)
        txt_x_gap_mm = random.uniform(2.0, 5.0)
        txt_x = int(x_pulse_end_max + txt_x_gap_mm * px_per_mm)
        txt_x = max(0, min(txt_x, w - 1))
        txt_y = max(10, min(txt_y, h - 1))
        cv2.putText(sig_rgb, lead, (txt_x, txt_y), font_face, font_scale, ink_color, 2, cv2.LINE_AA)
        cv2.putText(alpha_other, lead, (txt_x, txt_y), font_face, font_scale, 255, 2, cv2.LINE_AA)


# ============================
# 5. 主渲染函数 (V37 修正)
# ============================
def render_clean_ecg_v37(df, layout_type, params, fs, sig_len):
    """V37 修正版：传递 fs 和 sig_len"""
    h, w = 1700, 2200
    px_per_mm = params['px_per_mm']
    
    paper_color = get_random_paper_color()
    plain_paper = generate_paper_texture(h, w, paper_color)
    
    temp_base = plain_paper.copy()
    grid_minor_color = random_color_variations(random.choice(COLOR_GRID_MINOR_BASE_OPTIONS), 3)
    grid_major_color = random_color_variations(random.choice(COLOR_GRID_MAJOR_BASE_OPTIONS), 3)
    
    # 网格绘制
    for x in np.arange(0, w, px_per_mm):
        cv2.line(temp_base, (int(x), 0), (int(x), h), grid_minor_color, 2)
    for y in np.arange(0, h, px_per_mm):
        cv2.line(temp_base, (0, int(y)), (w, int(y)), grid_minor_color, 2)
    for x in np.arange(0, w, px_per_mm * 5):
        cv2.line(temp_base, (int(x), 0), (int(x), h), grid_major_color, 3)
    for y in np.arange(0, h, px_per_mm * 5):
        cv2.line(temp_base, (0, int(y)), (w, int(y)), grid_major_color, 3)
    
    base = generate_paper_texture(h, w, paper_color, grid_img=temp_base)
    
    sig_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 🔥 修改：'alpha_waveform' -> 'wave_label_semantic_mask'
    # 这个掩码现在将存储ID（1-12），而不是255
    wave_label_semantic_mask = np.zeros((h, w), dtype=np.uint8) 
    
    alpha_other = np.zeros((h, w), dtype=np.uint8)
    alpha_baseline = np.zeros((h, w), dtype=np.uint8)
    
    ink_color = get_random_ink_color()
    RANDOM_FONT = random.choice(FONT_LIST)
    
    # 页眉/页脚
    MT_GUESS = int(150 * (h/1700))
    MB_GUESS = int(100 * (h/1700))
    ML_GUESS = int(10*px_per_mm)
    font_scale_header = random.uniform(0.8, 1.1)
    font_scale_footer = random.uniform(0.9, 1.2)
    
    # 页眉位置保持不变（打印ID）
    cv2.putText(sig_rgb, f"ID: {random.randint(10000, 99999)}_hr", 
                (ML_GUESS, MT_GUESS - int(10*px_per_mm)), 
                RANDOM_FONT, font_scale_header, ink_color, 1, cv2.LINE_AA)
    cv2.putText(alpha_other, f"ID: {random.randint(10000, 99999)}_hr", 
                (ML_GUESS, MT_GUESS - int(10*px_per_mm)), 
                RANDOM_FONT, font_scale_header, 255, 1, cv2.LINE_AA)
    
    # V38 优化：纸速位置随机化
    # 基准位置：底部中央偏左
    base_x = w // 2 - int(50 * px_per_mm)
    base_y = h - MB_GUESS + int(5 * px_per_mm)
    
    # 添加随机偏移
    offset_x = random.randint(-int(30 * px_per_mm), int(30 * px_per_mm))  # 左右浮动±30mm
    offset_y = random.randint(-int(8 * px_per_mm), int(3 * px_per_mm))    # 上下浮动 -8mm~+3mm
    
    footer_x = max(ML_GUESS, min(base_x + offset_x, w - int(100 * px_per_mm)))  # 限制在页面内
    footer_y = max(h - MB_GUESS - int(15 * px_per_mm), min(base_y + offset_y, h - int(5 * px_per_mm)))
    
    # 👇 这里是实际打印纸速的代码
    cv2.putText(sig_rgb, f"{params['paper_speed_mm_s']:.1f}mm/s", 
                (footer_x, footer_y), 
                RANDOM_FONT, font_scale_footer, ink_color, 2, cv2.LINE_AA)
    cv2.putText(alpha_other, f"{params['paper_speed_mm_s']:.1f}mm/s", 
                (footer_x, footer_y), 
                RANDOM_FONT, font_scale_footer, 255, 2, cv2.LINE_AA)
    
    # 布局渲染（传递 fs 和 sig_len）
    # 🔥 修改：传递 'wave_label_semantic_mask'
    if layout_type == LayoutType.LAYOUT_3X4_PLUS_II:
        # 3x4+1 布局（带底部长导联）
        render_layout_3x4_plus_II_v37(df, sig_rgb, wave_label_semantic_mask, alpha_other, alpha_baseline, h, w, params, ink_color, RANDOM_FONT, fs, sig_len)
    elif layout_type == LayoutType.LAYOUT_3X4:
        # 🔥 修复：3x4 纯网格布局（不带长导联）需要专门的函数
        render_layout_3x4_v37(df, sig_rgb, wave_label_semantic_mask, alpha_other, alpha_baseline, h, w, params, ink_color, RANDOM_FONT, fs, sig_len)
    elif layout_type == LayoutType.LAYOUT_6X2:
        # 6x2 布局使用新的 v38 函数
        render_layout_6x2_v37(df, sig_rgb, wave_label_semantic_mask, alpha_other, alpha_baseline, h, w, params, ink_color, RANDOM_FONT, fs, sig_len)
    elif layout_type == LayoutType.LAYOUT_12X1:
        # 12x1 布局使用新的 v38 函数
        render_layout_12x1_v37(df, sig_rgb, wave_label_semantic_mask, alpha_other, alpha_baseline, h, w, params, ink_color, RANDOM_FONT, fs, sig_len)
    else:
        # 默认使用 3x4+1 布局
        render_layout_3x4_plus_II_v37(df, sig_rgb, wave_label_semantic_mask, alpha_other, alpha_baseline, h, w, params, ink_color, RANDOM_FONT, fs, sig_len)
    
    # 🔥 修改：
    # 'wave_label_semantic_mask' (值为 0-12) 是我们想要的标签
    # 'combined_alpha' (值为 0 或 255) 用于渲染 'clean_img'
    
    # 1. 创建一个临时的二值波形掩码 (0 或 255)
    wave_mask_binary = (wave_label_semantic_mask > 0).astype(np.uint8) * 255
    
    # 2. 结合二值波形 和 其他元素（文本，脉冲）
    combined_alpha = np.maximum(wave_mask_binary, alpha_other)
    
    # 3. 使用 combined_alpha 渲染 clean_img
    alpha_mask = (combined_alpha.astype(np.float32) / 255.0)[..., None]
    clean_img = (base * (1.0 - alpha_mask) + sig_rgb * alpha_mask).astype(np.uint8)
    
    # 🔥 修改：返回 'wave_label_semantic_mask' 作为波形标签
    return clean_img, base, wave_label_semantic_mask, alpha_other, alpha_baseline, paper_color

# ============================
# 6. 退化引擎
# ============================
def add_stains_v26(img):
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
    stain_mask = ((mask_eroded * random.uniform(0.3, 0.6)) + (tide_line_mask * random.uniform(0.8, 1.0)))
    stain_mask = np.clip(stain_mask, 0, 1)
    for c in range(3): 
        stain_overlay[:, :, c] = 255 * (1 - stain_mask) + stain_color[c] * stain_mask
    img_stained = (img.astype(np.float32) * stain_overlay / 255.0).clip(0, 255).astype(np.uint8)
    return img_stained

def add_severe_damage(img, alpha_wave, alpha_other, alpha_baseline):
    # 'alpha_wave' (即 'wave_label_semantic_mask') 在这里被正确处理
    # 因为 cv2.line/circle 的颜色是 0，会清除所有导联ID
    h, w = img.shape[:2]
    num_damages = random.randint(2, 5)
    for _ in range(num_damages):
        damage_type = random.choice(['tear', 'hole', 'crease'])
        if damage_type == 'tear':
            x1, y1 = random.randint(0, w), random.randint(0, h)
            x2, y2 = random.randint(0, w), random.randint(0, h)
            thickness = random.randint(5, 15)
            cv2.line(img, (x1, y1), (x2, y2), (240, 240, 240), thickness)
            cv2.line(alpha_wave, (x1, y1), (x2, y2), 0, thickness) # 用 0 擦除
            cv2.line(alpha_other, (x1, y1), (x2, y2), 0, thickness)
            cv2.line(alpha_baseline, (x1, y1), (x2, y2), 0, thickness)
        elif damage_type == 'hole':
            center = (random.randint(0, w), random.randint(0, h))
            radius = random.randint(10, 30)
            cv2.circle(img, center, radius, (240, 240, 240), -1)
            cv2.circle(alpha_wave, center, radius, 0, -1) # 用 0 擦除
            cv2.circle(alpha_other, center, radius, 0, -1)
            cv2.circle(alpha_baseline, center, radius, 0, -1)
    return img, alpha_wave, alpha_other, alpha_baseline

def add_mold_spots(img):
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
    mask = np.zeros((h, w), dtype=np.uint8)
    for _ in range(num_spots): 
        center = (random.randint(0, w), random.randint(0, h))
        cv2.circle(mask, center, random.randint(10, 30), 255, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0) / 255.0
    img = img.astype(np.float32) * (1 - mask[..., None] * 0.3)
    return np.clip(img, 0, 255).astype(np.uint8)

def add_printer_halftone(img):
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
    h, w = img.shape[:2]
    freq = random.uniform(0.05, 0.15)
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    pattern1 = np.sin(2 * np.pi * freq * X) * np.cos(2 * np.pi * freq * Y)
    pattern2 = np.sin(2 * np.pi * freq * 1.1 * (X + Y))
    moire = (pattern1 + pattern2) * random.uniform(15, 30)
    result = img.astype(np.float32) + moire[..., None]
    return np.clip(result, 0, 255).astype(np.uint8)

def add_motion_blur(img):
    size = random.randint(5, 15)
    angle = random.uniform(0, 180)
    kernel = np.zeros((size, size))
    kernel[int((size - 1) / 2), :] = np.ones(size)
    kernel = kernel / size
    M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (size, size))
    return cv2.filter2D(img, -1, kernel)

def add_jpeg_compression(img, quality=None):
    if quality is None: 
        quality = random.randint(40, 70)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 1)

def apply_degradation_pipeline_v32(img, grid, wave, other, baseline, degradation_type, paper_color):
    """
    V32 版本：保持与原代码一致
    'wave' (即 'wave_label_semantic_mask') 在这里被正确处理, 
    因为它将使用 'cv2.INTER_NEAREST' 进行变换，这对于标签掩码是正确的。
    """
    h, w = img.shape[:2]
    
    # 类型特定退化
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
        img = add_stains_v26(img)
    elif degradation_type == DegradationType.DAMAGED:
        img, wave, other, baseline = add_severe_damage(img, wave, other, baseline)
    elif degradation_type in [DegradationType.MOLD_COLOR, DegradationType.MOLD_BW]:
        img = add_mold_spots(img)
        if degradation_type == DegradationType.MOLD_BW:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if degradation_type == DegradationType.CLEAN:
        return img, np.eye(3), grid, wave, other, baseline
    
    # 几何变换
    M_geo = np.eye(3)
    border_color_img = random_color_variations((20, 20, 20), 10)
    border_color_grid = (0, 0, 0)
    border_color_mask = 0
    
    if random.random() < 0.8:
        angle = random.uniform(-5, 5)
        center = (w // 2, h // 2)
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        M_rot_3x3 = np.vstack([M_rot, [0, 0, 1]])
        src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        offset = min(h, w) * random.uniform(0.01, 0.08)
        random_offset = np.random.uniform(-offset, offset, (4, 2)).astype(np.float32)
        dst_pts = src_pts + random_offset
        M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
        M_geo = M_persp @ M_rot_3x3
        
        img = cv2.warpPerspective(img, M_geo, (w, h), flags=cv2.INTER_LINEAR, borderValue=border_color_img)
        grid = cv2.warpPerspective(grid, M_geo, (w, h), flags=cv2.INTER_LINEAR, borderValue=border_color_grid)
        
        # 🔥 关键：对 'wave' (语义掩码) 使用 INTER_NEAREST 
        wave = cv2.warpPerspective(wave, M_geo, (w, h), flags=cv2.INTER_NEAREST, borderValue=border_color_mask)
        
        other = cv2.warpPerspective(other, M_geo, (w, h), flags=cv2.INTER_NEAREST, borderValue=border_color_mask)
        baseline = cv2.warpPerspective(baseline, M_geo, (w, h), flags=cv2.INTER_NEAREST, borderValue=border_color_mask)
    
    # 光学效果
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    radius = np.sqrt(X**2 + Y**2)
    vignette = 1 - np.clip(radius * random.uniform(0.5, 0.8), 0, 1)
    gradient = np.tile(np.linspace(random.uniform(0.7, 1.0), random.uniform(0.7, 1.0), w), (h, 1))
    lighting_mask = np.clip(vignette * gradient, 0.4, 1.0)
    img = (img.astype(np.float32) * lighting_mask[..., None]).astype(np.uint8)
    if random.random() < 0.7: 
        img = cv2.GaussianBlur(img, (5, 5), 0)
    noise = np.random.normal(0, 12, (h, w, 3))
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return img, M_geo, grid, wave, other, baseline

# ============================
# 7. 批量处理器 (V37 修正)
# ============================
BASE_DATA_DIR = "/Volumes/movie/work/physionet-ecg-image-digitization"
OUTPUT_DIR = "/Volumes/movie/work/physionet-ecg-image-digitization-simulations-V37"
CONFIG = {
    "NUM_VARIATIONS_PER_CSV": 2,
    "LAYOUT_DISTRIBUTION": {
        LayoutType.LAYOUT_3X4_PLUS_II: 0.60, LayoutType.LAYOUT_3X4: 0.20,
        LayoutType.LAYOUT_6X2: 0.15, LayoutType.LAYOUT_12X1: 0.05,
    },
    "DEGRADATION_DISTRIBUTION": {
        DegradationType.CLEAN: 0.05, DegradationType.PRINTED_COLOR: 0.15, DegradationType.PRINTED_BW: 0.15,
        DegradationType.PHOTO_PRINT: 0.20, DegradationType.PHOTO_SCREEN: 0.15, DegradationType.STAINED: 0.10,
        DegradationType.DAMAGED: 0.10, DegradationType.MOLD_COLOR: 0.05, DegradationType.MOLD_BW: 0.05,
    }
}
TRAIN_CSV_LIST_PATH = os.path.join(BASE_DATA_DIR, "train.csv")
TRAIN_DIR = os.path.join(BASE_DATA_DIR, "train")
NUM_WORKERS = max(1, os.cpu_count() - 2)

def sample_layout_type():
    types = list(CONFIG["LAYOUT_DISTRIBUTION"].keys())
    probs = list(CONFIG["LAYOUT_DISTRIBUTION"].values())
    return random.choices(types, weights=probs, k=1)[0]

def sample_degradation_type():
    types = list(CONFIG["DEGRADATION_DISTRIBUTION"].keys())
    probs = list(CONFIG["DEGRADATION_DISTRIBUTION"].values())
    return random.choices(types, weights=probs, k=1)[0]

def process_one_id_v37(task_tuple, train_dir, train_meta_df, output_dir):
    """V37 修正版：从 train.csv 读取 fs 和 sig_len"""
    ecg_id = None
    variation_index = None
    try:
        ecg_id, variation_index = task_tuple
        ecg_id_str = str(ecg_id)
        
        # V37 修正：从 metadata 获取真实 fs 和 sig_len
        meta_row = train_meta_df[train_meta_df['id'] == int(ecg_id)]
        if len(meta_row) == 0:
            return (ecg_id_str, "metadata_not_found")
        
        fs = int(meta_row.iloc[0]['fs'])
        sig_len = int(meta_row.iloc[0]['sig_len'])
        
        layout_type = sample_layout_type()
        degradation_type = sample_degradation_type()
        params = sample_physical_params_v37(layout_type)
        
        variation_id = f"{ecg_id_str}_v{variation_index:02d}_{layout_type}_{degradation_type}"
        csv_path = os.path.join(train_dir, ecg_id_str, f"{ecg_id_str}.csv")
        output_subdir = os.path.join(output_dir, variation_id)
        os.makedirs(output_subdir, exist_ok=True)
        
        dirty_path = os.path.join(output_subdir, f"{variation_id}_dirty.png")
        grid_label_path = os.path.join(output_subdir, f"{variation_id}_label_grid.png")
        wave_label_path = os.path.join(output_subdir, f"{variation_id}_label_wave.png")
        other_label_path = os.path.join(output_subdir, f"{variation_id}_label_other.png")
        baseline_label_path = os.path.join(output_subdir, f"{variation_id}_label_baseline.png")
        metadata_path = os.path.join(output_subdir, f"{variation_id}_metadata.json")
        
        if all(os.path.exists(p) for p in [dirty_path, grid_label_path, wave_label_path, other_label_path, baseline_label_path, metadata_path]):
            return (variation_id, "skipped")
        
        if not os.path.exists(csv_path):
            return (ecg_id_str, "csv_not_found")
        df = pd.read_csv(csv_path)
        
        # V37 修正：传递 fs 和 sig_len
        # 🔥 修改：'alpha_wave_paper' 现在是 'wave_label_semantic_mask' (值为 0-12)
        clean_img_paper, grid_img_paper, alpha_wave_paper, alpha_other_paper, alpha_baseline_paper, paper_color = \
            render_clean_ecg_v37(df, layout_type=layout_type, params=params, fs=fs, sig_len=sig_len)
        
        h_paper, w_paper = clean_img_paper.shape[:2]
        h_bed = h_paper + random.randint(100, 300)
        w_bed = w_paper + random.randint(100, 400)
        
        bed_img = generate_scanner_background(h_bed, w_bed)
        bed_label_grid = np.zeros((h_bed, w_bed, 3), dtype=np.uint8)
        bed_label_wave = np.zeros((h_bed, w_bed), dtype=np.uint8) # 仍然是 uint8
        bed_label_other = np.zeros((h_bed, w_bed), dtype=np.uint8)
        bed_label_baseline = np.zeros((h_bed, w_bed), dtype=np.uint8)
        x_offset = random.randint(20, w_bed - w_paper - 20)
        y_offset = random.randint(20, h_bed - h_paper - 20)
        
        bed_img[y_offset:y_offset+h_paper, x_offset:x_offset+w_paper] = clean_img_paper
        bed_label_grid[y_offset:y_offset+h_paper, x_offset:x_offset+w_paper] = grid_img_paper
        bed_label_wave[y_offset:y_offset+h_paper, x_offset:x_offset+w_paper] = alpha_wave_paper # 粘贴语义掩码
        bed_label_other[y_offset:y_offset+h_paper, x_offset:x_offset+w_paper] = alpha_other_paper
        bed_label_baseline[y_offset:y_offset+h_paper, x_offset:x_offset+w_paper] = alpha_baseline_paper
        
        dirty_img, M_geo, grid_label_warped, wave_label_warped, other_label_warped, baseline_label_warped = \
            apply_degradation_pipeline_v32(
                bed_img, bed_label_grid, bed_label_wave, bed_label_other, bed_label_baseline,
                degradation_type, paper_color
            )
        
        cv2.imwrite(dirty_path, dirty_img)
        cv2.imwrite(grid_label_path, grid_label_warped)
        cv2.imwrite(wave_label_path, wave_label_warped) # 保存 0-12 值的语义掩码
        cv2.imwrite(other_label_path, other_label_warped)
        cv2.imwrite(baseline_label_path, baseline_label_warped)
        
        # V37 修正：保存 fs 和 sig_len 到 metadata
        metadata = {
            "ecg_id": ecg_id_str,
            "fs": fs,
            "sig_len": sig_len,
            "layout_type": layout_type,
            "degradation_type": degradation_type,
            "physical_params": params,
            "image_size_bed": {"height": h_bed, "width": w_bed},
            "paper_paste_offset": {"x": x_offset, "y": y_offset},
            "geometric_transform": M_geo.tolist() if not np.allclose(M_geo, np.eye(3)) else None,
            "paper_color_bgr": paper_color,
            # 🔥 新增：保存导联ID映射
            "lead_to_id_map": LEAD_TO_ID_MAP
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return (variation_id, "success")
    
    except Exception as e:
        import traceback
        print(f"Error processing {ecg_id} v{variation_index}: {e}")
        traceback.print_exc() # 打印完整堆栈
        
        if ecg_id is not None and variation_index is not None:
            variation_id = f"{ecg_id}_v{variation_index:02d}"
        else:
            variation_id = "unknown"
        error_msg = f"Error: {type(e).__name__}: {str(e)}"
        return (variation_id, error_msg)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='ECG仿真器 V37 - 修复采样率问题')
    parser.add_argument('--debug', action='store_true', help='单进程调试模式')
    parser.add_argument('--limit', type=int, default=None, help='限制处理的ID数量')
    parser.add_argument('--workers', type=int, default=None, help='并行worker数量')
    args = parser.parse_args()
    
    if args.workers: 
        NUM_WORKERS = args.workers
    
    print("=" * 70)
    print("ECG 仿真器 V37 (修复采样率和信号长度问题)")
    print("🔥 (修改版：输出语义分割掩码 0-12)")
    print("=" * 70)
    print(f"数据源: {TRAIN_CSV_LIST_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"每个CSV变体数: {CONFIG['NUM_VARIATIONS_PER_CSV']}")
    
    if args.debug: 
        print("\n⚠️ 运行在调试模式（单进程）")
        NUM_WORKERS = 1
    if args.limit: 
        print(f"⚠️ 限制处理 {args.limit} 个ID")
    
    print(f"\n布局类型分布:")
    for layout, prob in CONFIG["LAYOUT_DISTRIBUTION"].items(): 
        print(f" {layout:15s}: {prob*100:5.1f}%")
    print(f"\n退化类型分布:")
    for dtype, prob in CONFIG["DEGRADATION_DISTRIBUTION"].items(): 
        print(f" {dtype:15s}: {prob*100:5.1f}%")
    print(f"\n使用 {NUM_WORKERS} 个并行Worker")
    print("-" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        train_meta_df = pd.read_csv(TRAIN_CSV_LIST_PATH)
        
        # V37 新增：验证 fs 和 sig_len
        print("\n验证 train.csv 数据...")
        print(f"总ID数: {len(train_meta_df)}")
        print(f"\nfs 分布:")
        print(train_meta_df['fs'].value_counts().sort_index())
        print(f"\nfs 与 sig_len 关系检查（前5条）:")
        print(train_meta_df[['id', 'fs', 'sig_len']].head())
        
        # 验证 sig_len = fs * 10
        train_meta_df['expected_sig_len'] = train_meta_df['fs'] * 10
        mismatch = train_meta_df[train_meta_df['sig_len'] != train_meta_df['expected_sig_len']]
        if len(mismatch) > 0:
            print(f"\n⚠️ 警告: {len(mismatch)} 个ID的 sig_len != fs * 10")
            print("这些ID可能有截断或填充，仿真时会使用实际 sig_len")
        else:
            print("\n✓ 所有ID的 sig_len = fs * 10")
        
        all_ids_base = train_meta_df['id'].astype(str).tolist()
        if args.limit: 
            all_ids_base = all_ids_base[:args.limit]
        
        all_tasks = []
        for ecg_id in all_ids_base:
            for i in range(CONFIG["NUM_VARIATIONS_PER_CSV"]):
                all_tasks.append((ecg_id, i))
        
        print(f"\n总共找到 {len(all_ids_base)} 个基础ID")
        print(f"总任务数: {len(all_tasks)}")
    except FileNotFoundError:
        print(f"错误: 找不到 train.csv 于 {TRAIN_CSV_LIST_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 读取 train.csv 失败: {e}")
        sys.exit(1)
    
    # V37 修正：传递 train_meta_df
    worker_func = partial(process_one_id_v37, train_dir=TRAIN_DIR, train_meta_df=train_meta_df, output_dir=OUTPUT_DIR)
    
    results = []
    print("\n开始生成...")
    
    try:
        if args.debug:
            print("⚠️ 调试模式：如果出错会显示完整堆栈\n")
            for task in tqdm(all_tasks, desc="生成仿真数据"):
                result = worker_func(task)
                results.append(result)
                if result[1] not in ["success", "skipped"]:
                    print(f"\n❌ 失败: {result[0]} - {result[1]}")
        else:
            with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
                for result in tqdm(pool.imap_unordered(worker_func, all_tasks),
                                   total=len(all_tasks), desc="生成仿真数据"):
                    results.append(result)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n\n❌ 严重错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("-" * 70)
    print("批量处理完成")
    
    status_counts = {"success": 0, "skipped": 0}
    failures = []
    for sample_id, status in results:
        if status == "success": 
            status_counts["success"] += 1
        elif status == "skipped": 
            status_counts["skipped"] += 1
        else: 
            failures.append((sample_id, status))
    
    print(f"\n成功生成: {status_counts['success']}")
    print(f"跳过(已存在): {status_counts['skipped']}")
    print(f"失败: {len(failures)}")
    
    if len(failures) > 0:
        print("\n--- 失败详情 (最多显示20条) ---")
        for i, (sample_id, status) in enumerate(failures):
            if i >= 20: 
                print(f"... 还有 {len(failures) - 20} 条未显示 ...")
                break
            print(f"ID: {sample_id} -> {status}")
        
        failures_path = os.path.join(OUTPUT_DIR, "failed_samples.txt")
        with open(failures_path, 'w') as f:
            for sample_id, status in failures: 
                f.write(f"{sample_id}\t{status}\n")
        print(f"\n失败列表已保存至: {failures_path}")
    
    print("\n" + "=" * 70)
    print("统计实际生成的布局和退化分布...")
    
    layout_counts = {layout: 0 for layout in LAYOUT_CONFIGS.keys()}
    degradation_counts = {dtype: 0 for dtype in CONFIG["DEGRADATION_DISTRIBUTION"].keys()}
    
    for sample_id, status in results:
        if status == "success":
            parts = sample_id.split('_')
            if len(parts) >= 4:
                layout = parts[2]
                degradation = parts[3]
                if layout in layout_counts:
                    layout_counts[layout] += 1
                if degradation in degradation_counts:
                    degradation_counts[degradation] += 1
    
    total_success = status_counts['success']
    if total_success > 0:
        print("\n实际布局分布:")
        for layout, count in layout_counts.items():
            percentage = (count / total_success) * 100
            print(f" {layout:15s}: {count:5d} ({percentage:5.1f}%)")
        
        print("\n实际退化分布:")
        for dtype, count in degradation_counts.items():
            percentage = (count / total_success) * 100
            print(f" {dtype:15s}: {count:5d} ({percentage:5.1f}%)")
    
    stats_path = os.path.join(OUTPUT_DIR, "generation_stats.json")
    stats = {
        "total_tasks": len(all_tasks),
        "success": status_counts["success"],
        "skipped": status_counts["skipped"],
        "failed": len(failures),
        "layout_distribution": layout_counts,
        "degradation_distribution": degradation_counts,
        "config": {k: v for k, v in CONFIG.items() if k in ["NUM_VARIATIONS_PER_CSV", "LAYOUT_DISTRIBUTION", "DEGRADATION_DISTRIBUTION"]},
        "version": "V37-SemanticMask",
        "fixes": [
            "使用真实 fs 从 train.csv",
            "使用真实 sig_len 从 train.csv", 
            "修正导联时长逻辑（固定 2.5s/10s）",
            "修正时间索引计算（基于真实 fs）",
            "🔥 修改：'wave' 标签现在是 0-12 的语义分割掩码",
            "🔥 修改：'metadata.json' 包含 'lead_to_id_map' "
        ]
    }
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n统计信息已保存至: {stats_path}")
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)
    
    # V37 新增：额外验证
    if total_success > 0:
        print("\n正在验证生成的样本...")
        
        # 修复：找到第一个成功的样本
        sample_id_to_check = None
        for sample_id, status in results:
            if status == "success":
                sample_id_to_check = sample_id
                break
        
        if sample_id_to_check:
            sample_dir = os.path.join(OUTPUT_DIR, sample_id_to_check)
            metadata_file = os.path.join(sample_dir, f"{sample_id_to_check}_metadata.json")
            
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        sample_meta = json.load(f)
                    print(f"\n示例样本元数据 (ID: {sample_id_to_check}):")
                    print(f"   ECG ID: {sample_meta.get('ecg_id')}")
                    print(f"   采样率 (fs): {sample_meta.get('fs')} Hz")
                    print(f"   信号长度 (sig_len): {sample_meta.get('sig_len')} 样本")
                    print(f"   布局类型: {sample_meta.get('layout_type')}")
                    print(f"   退化类型: {sample_meta.get('degradation_type')}")
                    print(f"   纸速: {sample_meta.get('physical_params', {}).get('paper_speed_mm_s')} mm/s")
                    print(f"   增益: {sample_meta.get('physical_params', {}).get('gain_mm_mv')} mm/mV")
                    
                    # 🔥 验证新修改
                    if 'lead_to_id_map' in sample_meta:
                        print("   ✓ 包含 'lead_to_id_map'")
                        print(f"     (II -> {sample_meta['lead_to_id_map'].get('II')}, V1 -> {sample_meta['lead_to_id_map'].get('V1')})")
                    else:
                        print("   ⚠️ 警告: 'metadata.json' 缺失 'lead_to_id_map'")
                    
                    # 验证 fs 和 sig_len 的关系
                    expected_sig_len = sample_meta.get('fs') * 10
                    actual_sig_len = sample_meta.get('sig_len')
                    if expected_sig_len == actual_sig_len:
                        print(f"   ✓ sig_len ({actual_sig_len}) = fs ({sample_meta.get('fs')}) × 10")
                    else:
                        print(f"   ⚠️ sig_len ({actual_sig_len}) ≠ fs ({sample_meta.get('fs')}) × 10 (expected {expected_sig_len})")
                    
                    print("\n✓ 元数据验证通过")
                except Exception as e:
                    print(f"\n⚠️ 警告: 读取元数据失败: {e}")
            else:
                print(f"\n⚠️ 警告: 找不到元数据文件: {metadata_file}")
        else:
            print("\n⚠️ 警告: 没有找到成功生成的样本用于验证")
    
    print("\n" + "=" * 70)
    print("所有任务完成！")
    print("=" * 70)