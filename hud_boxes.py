#!/usr/bin/env python3
"""Breakcore video effect: tracking-HUD boxes + glow, glitch, datamosh, beat-synced hits.

Put videos into the  input/  folder next to this script and run:
    python hud_boxes.py
Results are written to the  output/  folder with the same name.

Effects (all on by default, each can be turned off with --no-<name>):
    brighten   lift very dark footage so there is something to work with
    grade      crushed blacks, contrast, strong saturation
    shake      handheld jitter + zoom punches + zoom blur on beats
    mosh       datamosh-style smear (pixels dragged along motion), triggered on beats
    trails     light trails / ghosting
    recolor    bright areas flash flat blue / pink / cyan / purple
    bloom      glow around lights
    flash      white flashes (and rare inverts) on beats
    letterbox  black cinematic bars
    HUD        corner-bracket boxes with x:/y: labels on the brightest spots
Beats are detected from the video's own audio (no audio -> random hits).

Use --intensity 0.5 for a calmer look, 1.5 for chaos.
Any format ffmpeg can read works. Requires ffmpeg + ffprobe installed on the system.
"""
import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE / "input"
OUTPUT_DIR = HERE / "output"

VIDEO_EXTS = {
    ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".flv", ".wmv", ".mpg", ".mpeg",
    ".ts", ".mts", ".m2ts", ".3gp", ".3g2", ".ogv", ".vob", ".mxf", ".gif", ".asf", ".dv",
}
# Containers we write back in the same format; anything else is saved as .mp4.
KEEP_EXTS = {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi"}

WHITE = (255, 255, 255)
# Highlight recolor palette (BGR): electric blue, hot pink, cyan, purple
PALETTE = [(255, 70, 20), (200, 60, 255), (255, 225, 40), (255, 40, 150)]


# =====================================================================
# ffmpeg helpers
# =====================================================================

def probe(path):
    """Return (width, height, fps, frame_count_or_0, has_audio) — size is after rotation."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    w, h = int(v["width"]), int(v["height"])

    # Phone videos often store rotation as metadata; ffmpeg auto-rotates, so swap size.
    rot = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(sd["rotation"])
    rot = int(v.get("tags", {}).get("rotate", rot))
    if abs(rot) % 180 == 90:
        w, h = h, w

    num, den = (v.get("avg_frame_rate") or "0/0").split("/")
    fps = float(num) / float(den) if float(den or 0) else 0
    if not fps or fps > 240:
        num, den = v.get("r_frame_rate", "30/1").split("/")
        fps = float(num) / float(den) if float(den) else 30.0

    frames = int(v.get("nb_frames", 0) or 0)
    if not frames:
        dur = float(v.get("duration") or info.get("format", {}).get("duration") or 0)
        frames = int(dur * fps)
    return w, h, fps, frames, has_audio


def encoder_args(ext, has_audio, fast=False):
    if ext == ".webm" and not fast:
        v = ["-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0", "-row-mt", "1"]
        a = ["-c:a", "libopus", "-b:a", "160k"]
    else:
        v = ["-c:v", "libx264", "-preset", "ultrafast" if fast else "medium",
             "-crf", "26" if fast else "18", "-pix_fmt", "yuv420p"]
        a = ["-c:a", "aac", "-b:a", "128k" if fast else "192k"]
        if ext in (".mp4", ".m4v", ".mov"):
            v += ["-movflags", "+faststart"]
    return v + (a if has_audio else [])


def media_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def read_audio(path, start=0.0, duration=None, sr=22050):
    """Decode (a part of) the audio track to mono float32, or None if there is none."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start}", "-i", str(path)]
    if duration:
        cmd += ["-t", f"{duration}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        return None
    return np.frombuffer(p.stdout, np.float32)


def loudness_env(audio, sr, fps, n_frames):
    """Per-video-frame loudness 0..1 (smoothed), so quiet parts get calmer effects."""
    hop = sr / fps
    win = max(1, int(hop))
    cs = np.concatenate([[0.0], np.cumsum(audio.astype(np.float64) ** 2)])
    s = np.clip((np.arange(n_frames) * hop).astype(np.int64), 0, len(audio))
    e = np.clip(s + win, 0, len(audio))
    rms = np.sqrt(np.maximum(cs[e] - cs[s], 0) / np.maximum(e - s, 1))
    k = max(1, int(fps * 0.3))
    rms = np.convolve(rms, np.ones(k) / k, "same")
    ref = np.percentile(rms, 95) if rms.any() else 1.0
    return np.clip(rms / (ref + 1e-9), 0, 1).astype(np.float32)


# =====================================================================
# Beat detection (spectral flux onsets, kick-weighted) -> per-video-frame envelope
# =====================================================================

def detect_beats(audio, sr, fps, n_frames, sensitivity=1.0):
    """Returns (env, beats): env[i] in 0..1 is a decaying 'hit' level, beats[i] is True on onsets."""
    win = 2048
    hop = sr / fps
    window = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, 1 / sr)
    weight = np.where(freqs < 200, 3.0, 1.0).astype(np.float32)  # kicks count more

    padded = np.concatenate([np.zeros(win // 2, np.float32), audio, np.zeros(win, np.float32)])
    starts = np.clip((np.arange(n_frames) * hop).astype(np.int64), 0, len(padded) - win)

    flux = np.zeros(n_frames, np.float32)
    prev = None
    for c in range(0, n_frames, 512):
        s = starts[c:c + 512]
        mag = np.log1p(np.abs(np.fft.rfft(padded[s[:, None] + np.arange(win)[None, :]] * window, axis=1)))
        prevs = np.vstack([mag[:1] if prev is None else prev[None], mag[:-1]])
        flux[c:c + len(s)] = (np.maximum(0, mag - prevs) * weight).sum(axis=1)
        prev = mag[-1]

    flux /= flux.max() + 1e-9
    k = max(1, int(fps * 0.5))
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    mean = np.convolve(flux, kernel, "same")
    std = np.sqrt(np.maximum(np.convolve(flux ** 2, kernel, "same") - mean ** 2, 0))
    thr = mean + std / max(sensitivity, 1e-3)

    beats = np.zeros(n_frames, bool)
    min_gap = max(2, int(fps * 0.12))
    last = -10 ** 9
    for i in range(1, n_frames - 1):
        if (flux[i] > thr[i] and flux[i] > 0.08 and flux[i] >= flux[i - 1]
                and flux[i] >= flux[i + 1] and i - last >= min_gap):
            beats[i] = True
            last = i
    return beats_to_env(beats, flux, fps), beats


def random_beats(n_frames, fps, rng):
    """Fallback when there's no audio: hits roughly every 0.25-0.75 s."""
    beats = np.zeros(n_frames, bool)
    i = int(rng.integers(0, max(1, int(fps * 0.5))))
    while i < n_frames:
        beats[i] = True
        i += int(rng.integers(max(1, int(fps * 0.25)), max(2, int(fps * 0.75))))
    strength = rng.uniform(0.4, 1.0, n_frames).astype(np.float32)
    return beats_to_env(beats, strength, fps), beats


def beats_to_env(beats, strength, fps):
    ref = np.percentile(strength[beats], 90) if beats.any() else 1.0
    decay = 0.5 ** (1 / max(1.0, fps * 0.08))  # 80 ms half-life
    env = np.zeros(len(beats), np.float32)
    level = 0.0
    for i in range(len(beats)):
        level *= decay
        if beats[i]:
            level = max(level, min(1.0, 0.35 + 0.65 * strength[i] / (ref + 1e-9)))
        env[i] = level
    return env


# =====================================================================
# Individual effects
# =====================================================================

class AutoBrightness:
    """Lifts dark frames toward a target average brightness with a gamma curve.
    Only brightens, ignores pure-black areas, smoothed over time (no flicker)."""

    def __init__(self, target=0.30, strength=1.0, smooth=0.9, max_boost=0.35, clahe=False):
        self.target, self.strength, self.smooth, self.max_boost = target, strength, smooth, max_boost
        self.gamma = None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if clahe else None
        self._lut_gamma = None
        self._lut = None

    def __call__(self, frame):
        small = cv2.cvtColor(cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        content = small[small > 10]
        mean = (content if content.size > small.size * 0.05 else small).mean() / 255.0
        mean = min(max(mean, 1e-3), 0.999)

        g = np.log(self.target) / np.log(mean)
        g = 1.0 - (1.0 - g) * self.strength
        g = float(np.clip(g, self.max_boost, 1.0))
        self.gamma = g if self.gamma is None else self.smooth * self.gamma + (1 - self.smooth) * g

        if self.gamma < 0.995:
            gq = round(self.gamma, 3)
            if gq != self._lut_gamma:
                self._lut = (((np.arange(256) / 255.0) ** gq) * 255).clip(0, 255).astype(np.uint8)
                self._lut_gamma = gq
            frame = cv2.LUT(frame, self._lut)
        if self.clahe is not None:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return frame


def draw_brackets(img, x, y, w, h, thickness=1):
    """Draw only the four corners of a rectangle."""
    L = max(4, min(w, h) // 4)
    for px, py, dx, dy in [(x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)]:
        cv2.line(img, (px, py), (px + dx * L, py), WHITE, thickness, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * L), WHITE, thickness, cv2.LINE_AA)


def find_bright_blobs(frame, threshold, max_boxes, min_area, auto_range=40, auto_floor=80):
    """threshold=None -> auto: everything within `auto_range` of the frame's brightest pixel."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if threshold is None:
        threshold = max(auto_floor, int(gray.max()) - auto_range)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    contours.sort(key=cv2.contourArea, reverse=True)
    return [cv2.boundingRect(c) for c in contours[:max_boxes]]


def letterbox_band(w, h, setting):
    """Return (top, bottom) rows of the visible band, or None for no bars."""
    if setting == "0" or setting.lower() == "off":
        return None
    if setting == "auto":
        r = w / h
        aspect = 2.35 if r >= 1.5 else (1.78 if r >= 0.9 else 1.0)
    else:
        aspect = float(setting)
    band = int(round(w / aspect))
    if band >= h - 4:
        return None
    top = (h - band) // 2
    return top, top + band


class BreakcoreFX:
    def __init__(self, w, h, fps, env, beats, args, rng, loud=None):
        self.w, self.h, self.fps = w, h, fps
        self.env, self.beats = env, beats
        self.a, self.rng = args, rng
        self.base_k = self.k = args.intensity
        self.loud = loud  # None -> constant intensity

        self.brighten = None if args.no_brighten else AutoBrightness(
            target=args.bright_target, strength=args.bright_strength, clahe=args.contrast)

        # grade LUTs
        x = np.arange(256) / 255.0
        x = np.clip((x - args.crush) / (1 - args.crush), 0, 1)
        c = float(np.clip(args.grade_contrast * self.k, 0, 1))
        self.tone_lut = ((x * (1 - c) + x * x * (3 - 2 * x) * c) * 255).clip(0, 255).astype(np.uint8)
        sat = 1 + (args.saturation - 1) * self.k
        self.sat_lut = (np.arange(256) * sat).clip(0, 255).astype(np.uint8)

        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        self.gx, self.gy = gx, gy

        self.band = letterbox_band(w, h, args.letterbox)

        # state
        self.prev_small = None
        self.prev_out = None
        self.trail = None
        self.mosh_left = 0
        self.recolor_left = 0
        self.recolor_color = PALETTE[0]
        self.recolor_thr = 0.7
        self.invert_left = 0
        self.prev_boxes = []
        self.label_pool = []
        self.font_scale = max(0.35, min(w, h) / 1400)

    # --- helpers -------------------------------------------------------
    def warp(self, img, scale, rot, tx, ty):
        M = cv2.getRotationMatrix2D((self.w / 2, self.h / 2), rot, scale)
        M[:, 2] += (tx, ty)
        return cv2.warpAffine(img, M, (self.w, self.h), borderMode=cv2.BORDER_REFLECT)

    # --- effects -------------------------------------------------------
    def grade(self, img):
        img = cv2.LUT(img, self.tone_lut)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.LUT(hsv[:, :, 1], self.sat_lut)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def camera(self, img, e):
        k, m = self.k, min(self.w, self.h)
        amp = (0.003 + 0.025 * e) * k * m
        tx, ty = self.rng.normal(0, amp, 2)
        rot = float(self.rng.normal(0, 1.5 * e * k))
        scale = 1.03 + self.a.punch * e * k  # 3% base zoom hides the shake edges
        out = self.warp(img, scale, rot, tx, ty)

        zb = 0.10 * e * k  # zoom blur on hits = fake speed streaks
        if zb > 0.015:
            acc = out.astype(np.float32)
            steps = np.linspace(1, 1 + zb, 5)[1:]
            for s in steps:
                acc += self.warp(out, s, 0, 0, 0)
            out = (acc / (len(steps) + 1)).astype(np.uint8)
        return out

    def mosh(self, img, beat):
        """Fake datamosh: move the PREVIOUS output along the current motion (like a codec
        that lost its keyframes), in 16px macroblocks."""
        sw, sh = max(8, self.w // 4), max(8, self.h // 4)
        small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (sw, sh), interpolation=cv2.INTER_AREA)
        prev_small, self.prev_small = self.prev_small, small

        if self.mosh_left <= 0 and beat and self.rng.random() < min(0.9, self.a.mosh_chance * self.k):
            self.mosh_left = int(self.rng.integers(max(2, int(self.fps * 0.3)), max(3, int(self.fps * 1.2))))
        if self.mosh_left <= 0 or prev_small is None or self.prev_out is None:
            return img
        self.mosh_left -= 1

        flow = cv2.calcOpticalFlowFarneback(small, prev_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        bw, bh = max(1, self.w // 16), max(1, self.h // 16)
        flow = cv2.resize(flow, (bw, bh), interpolation=cv2.INTER_AREA)
        flow = cv2.resize(flow, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        flow *= (self.w / sw) * 1.4  # back to full-res pixels, exaggerated
        warped = cv2.remap(self.prev_out, self.gx + flow[..., 0], self.gy + flow[..., 1],
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return cv2.addWeighted(warped, 0.88, img, 0.12, 0)

    def trails(self, img):
        decay = float(np.clip(self.a.trails * min(1.0, self.k), 0, 0.95))
        f = img.astype(np.float32)
        if self.trail is None:
            self.trail = f
        self.trail = np.maximum(f, self.trail * decay)  # lighten blend -> light streaks
        return self.trail.astype(np.uint8)

    def recolor(self, img, beat):
        if self.recolor_left <= 0:
            p = 0.01 * self.k + (0.35 * self.k if beat else 0)
            if self.rng.random() < p:
                self.recolor_left = int(self.rng.integers(max(2, int(self.fps * 0.15)), max(3, int(self.fps * 0.7))))
                self.recolor_color = PALETTE[int(self.rng.integers(len(PALETTE)))]
                self.recolor_thr = float(self.rng.uniform(0.55, 0.8))
        if self.recolor_left <= 0:
            return img
        self.recolor_left -= 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        thr = max(0.25, float(gray.max()) * self.recolor_thr)
        m = np.clip((gray - thr) / 0.1, 0, 1)
        m = cv2.GaussianBlur(m, (5, 5), 0)[..., None]
        layer = np.array(self.recolor_color, np.float32) * np.clip(0.55 + 0.7 * gray, 0, 1.2)[..., None]
        return (img * (1 - m) + np.clip(layer, 0, 255) * m).astype(np.uint8)

    def bloom_and_flash(self, img, e, beat):
        o = img.astype(np.float32) / 255
        if not self.a.no_bloom:
            sw, sh = max(8, self.w // 4), max(8, self.h // 4)
            small = cv2.resize(o, (sw, sh), interpolation=cv2.INTER_AREA)
            thr = 0.6
            hi = small * np.clip((small.max(axis=2) - thr) / (1 - thr), 0, 1)[..., None]
            s = min(sw, sh)
            glow = cv2.GaussianBlur(hi, (0, 0), s * 0.02) + 0.6 * cv2.GaussianBlur(hi, (0, 0), s * 0.07)
            glow = cv2.resize(glow, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            strength = (0.9 + 1.5 * e) * self.k * self.a.bloom
            o = 1 - (1 - o) * (1 - np.clip(glow * strength, 0, 1))  # screen blend

        if not self.a.no_flash:
            o = o + (1 - o) * (0.35 * e * self.k)
            if beat and self.invert_left <= 0 and self.rng.random() < 0.06 * self.k:
                self.invert_left = int(self.rng.integers(1, 4))
            if self.invert_left > 0:
                self.invert_left -= 1
                o = 1 - o
        return (np.clip(o, 0, 1) * 255).astype(np.uint8)

    def apply_letterbox(self, img):
        if self.band is None:
            return img
        top, bottom = self.band
        img[:top] = 0
        img[bottom:] = 0
        return img

    def hud(self, img, detect_src, beat):
        a, rng = self.a, self.rng
        w, h = self.w, self.h

        # 1-2 labels shared by all boxes, like the original effect
        if not self.label_pool or self.frame_idx % a.label_every == 0 or beat:
            n = 2 if rng.random() < 0.3 else 1
            self.label_pool = [f"x:{rng.integers(0, w)} y:{rng.integers(0, h)}" for _ in range(n)]

        boxes = find_bright_blobs(detect_src, a.threshold, a.max_boxes, a.min_area, a.auto_range, a.auto_floor)
        if self.band is not None:
            top, bottom = self.band
            boxes = [b for b in boxes if top <= b[1] + b[3] // 2 < bottom]
        # some boxes linger one more frame -> flickery, busier HUD
        lingering = [b for b in self.prev_boxes if rng.random() < 0.35]
        self.prev_boxes = boxes
        boxes = boxes + lingering

        fs = self.font_scale
        for x, y, bw, bh in boxes:
            if bw * bh < 30:  # tiny light -> small fixed box
                cx, cy = x + bw // 2, y + bh // 2
                s = int(rng.integers(7, 12))
                x, y, bw, bh = cx - s, cy - s, 2 * s, 2 * s
            else:
                pad = a.pad + int(rng.integers(0, a.pad + 1)) + int(0.15 * max(bw, bh) * rng.random())
                x, y, bw, bh = x - pad, y - pad, bw + 2 * pad, bh + 2 * pad
            draw_brackets(img, x, y, bw, bh)
            label = self.label_pool[0] if rng.random() < 0.8 else self.label_pool[-1]
            cv2.putText(img, label, (x + bw + 3, y + bh // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, WHITE, 1, cv2.LINE_AA)
            if bw > 70 and rng.random() < 0.5:
                cv2.putText(img, label, (x + 6, y + bh - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            fs, WHITE, 1, cv2.LINE_AA)
        return img

    # --- full pipeline ---------------------------------------------------
    def __call__(self, frame, i):
        self.frame_idx = i
        a = self.a
        e = float(self.env[i]) if i < len(self.env) else 0.0
        beat = bool(self.beats[i]) if i < len(self.beats) else False
        if self.loud is not None and i < len(self.loud):
            # follow the music: quiet parts ~0.35x, loud drops ~1.3x
            self.k = self.base_k * (0.35 + 0.95 * float(self.loud[i]))

        if self.brighten is not None:
            frame = self.brighten(frame)
        if not a.no_grade:
            frame = self.grade(frame)
        if not a.no_shake:
            frame = self.camera(frame, e)
        if not a.no_mosh:
            frame = self.mosh(frame, beat)
        if not a.no_trails:
            frame = self.trails(frame)
        self.prev_out = frame  # what the datamosh smears next time
        if not a.no_recolor:
            frame = self.recolor(frame, beat)

        detect_src = frame  # detect lights before glow makes everything bright
        frame = self.bloom_and_flash(frame, e, beat)
        frame = self.apply_letterbox(frame)
        if not a.no_hud:
            frame = self.hud(frame, detect_src, beat)
        return frame


# =====================================================================
# Main processing
# =====================================================================

def _even(x):
    return max(2, int(round(x / 2)) * 2)


def process_file(src, dst, args, progress=None, preview=False, log=print):
    """Render `src` -> `dst`.

    Extra options on `args` (all optional):
      start / duration   seconds — render only a part (used by the preview)
      max_side           downscale so the longer side is at most this (preview)
      music              path to a music file
      music_mode         'replace' (music only) or 'mix' (music + quiet original sound)
      music_offset       start the music at this second
      loop_video         repeat the video until the music ends
      music_adapt        effect strength follows the music's loudness
    progress(fraction) is called while rendering.
    """
    src, dst = Path(src), Path(dst)
    w0, h0, fps, total, has_audio = probe(src)
    vid_dur = total / fps if total else media_duration(src)

    start = max(0.0, float(getattr(args, "start", 0) or 0))
    music = getattr(args, "music", None)
    music = Path(music) if music else None
    mode = getattr(args, "music_mode", "replace") if music else "original"
    moff = float(getattr(args, "music_offset", 0) or 0)
    loop = bool(getattr(args, "loop_video", False) and music)

    if loop:
        length = media_duration(music) - moff - start
    else:
        length = vid_dur - start
    if getattr(args, "duration", None):
        length = min(length, float(args.duration))
    if length <= 0.05:
        raise ValueError("start is after the end of the video/music")
    n_frames = max(1, int(round(length * fps)))

    w, h = w0, h0
    max_side = getattr(args, "max_side", None)
    if max_side and max(w0, h0) > max_side:
        f = max_side / max(w0, h0)
        w, h = _even(w0 * f), _even(h0 * f)

    log(f"{src.name}: {w}x{h} @ {fps:.2f} fps, {length:.1f}s"
        + (f", music: {music.name} ({mode})" if music else ("" if has_audio else " (no audio)")))

    # ---- beats + loudness come from the music if given, else from the video's sound
    rng = np.random.default_rng(args.seed)
    if music:
        beat_src, beat_off = music, moff + start
    elif has_audio:
        beat_src, beat_off = src, start
    else:
        beat_src, beat_off = None, 0.0
    audio = read_audio(beat_src, beat_off, length) if (beat_src and not args.no_beats) else None
    loud = None
    if audio is not None and audio.size > 22050 // 2:
        env, beats = detect_beats(audio, 22050, fps, n_frames, args.beat_sensitivity)
        if getattr(args, "music_adapt", True):
            loud = loudness_env(audio, 22050, fps, n_frames)
        log(f"  beats detected: {int(beats.sum())}")
    else:
        env, beats = random_beats(n_frames, fps, rng)
        log("  no audio beats used -> random hits")

    fx = BreakcoreFX(w, h, fps, env, beats, args, rng, loud)

    # ---- decoder: raw frames from ffmpeg
    dec = ["ffmpeg", "-v", "error"]
    if loop:
        dec += ["-stream_loop", "-1"]
    dec += ["-ss", f"{start}", "-i", str(src), "-t", f"{length}", "-map", "0:v:0"]
    if (w, h) != (w0, h0):
        dec += ["-vf", f"scale={w}:{h}"]
    dec += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-vsync", "passthrough", "-"]

    # ---- encoder: processed frames + chosen audio
    enc = ["ffmpeg", "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-"]
    amap = []
    if mode == "original":
        if has_audio:
            enc += ["-ss", f"{start}", "-i", str(src)]
            amap = ["-map", "1:a:0"]
    else:
        enc += ["-ss", f"{moff + start}", "-i", str(music)]
        if mode == "mix" and has_audio and not loop:
            enc += ["-ss", f"{start}", "-i", str(src)]
            amap = ["-filter_complex",
                    "[2:a]volume=0.35[o];[1:a][o]amix=inputs=2:duration=first:normalize=0[a]",
                    "-map", "[a]"]
        else:
            amap = ["-map", "1:a:0"]
    enc += ["-map", "0:v:0", *amap,
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # x264/vp9 need even sizes
            *encoder_args(dst.suffix.lower(), bool(amap), fast=preview),
            "-t", f"{length}", str(dst)]

    dst.parent.mkdir(parents=True, exist_ok=True)
    decoder = subprocess.Popen(dec, stdout=subprocess.PIPE)
    encoder = subprocess.Popen(enc, stdin=subprocess.PIPE)

    frame_bytes = w * h * 3
    i = 0
    try:
        while i < n_frames:
            buf = decoder.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
            frame = fx(frame, i)
            encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
            i += 1
            if i % 10 == 0:
                if progress:
                    progress(i / n_frames)
                if log is print:
                    print(f"\r  {i}/{n_frames} frames", end="", flush=True)
    finally:
        decoder.stdout.close()
        decoder.kill()
        encoder.stdin.close()
        decoder.wait()
        encoder.wait()

    if encoder.returncode != 0 or i == 0:
        log(f"\r  FAILED ({i} frames read)")
        return False
    log(f"\r  {i}/{i} frames -> {dst}")
    return True


def output_path_for(src: Path) -> Path:
    ext = src.suffix.lower() if src.suffix.lower() in KEEP_EXTS else ".mp4"
    return OUTPUT_DIR / (src.stem + ext)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("general")
    g.add_argument("--intensity", type=float, default=1.0, help="scales all effects (0.5 calm, 1 default, 1.5 chaos)")
    g.add_argument("--seed", type=int, default=None, help="random seed, for the same result every run")
    g.add_argument("--preview", action="store_true", help="fast preview: 5 s, small size, saved as <name>_preview.mp4")
    g.add_argument("--start", type=float, default=0, help="start at this second")
    g.add_argument("--duration", type=float, default=None, help="render only this many seconds")
    g.add_argument("--max-side", type=int, default=None, help="downscale so the longer side is at most this")

    g = p.add_argument_group("music")
    g.add_argument("--music", default=None, help="music file to put under the video (beats are taken from it)")
    g.add_argument("--music-mode", choices=["replace", "mix"], default="replace",
                   help="replace the original sound, or mix it in quietly (default replace)")
    g.add_argument("--music-offset", type=float, default=0, help="start the music at this second")
    g.add_argument("--loop-video", action="store_true", help="repeat the video until the music ends")
    g.add_argument("--no-music-adapt", dest="music_adapt", action="store_false",
                   help="don't make effects follow the music's loudness")

    g = p.add_argument_group("HUD boxes")
    g.add_argument("--no-hud", action="store_true", help="no boxes")
    g.add_argument("--threshold", type=int, default=None,
                   help="fixed brightness 0-255 for boxes. Default: auto per frame")
    g.add_argument("--auto-range", type=int, default=40, help="auto: box spots within this of the brightest pixel (default 40)")
    g.add_argument("--auto-floor", type=int, default=80, help="auto: never box anything darker than this (default 80)")
    g.add_argument("--max-boxes", type=int, default=14, help="max detected boxes per frame (default 14)")
    g.add_argument("--min-area", type=int, default=3, help="ignore blobs smaller than this many pixels (default 3)")
    g.add_argument("--pad", type=int, default=6, help="padding around each blob (default 6)")
    g.add_argument("--label-every", type=int, default=2, help="change the coordinate label every N frames (default 2)")

    g = p.add_argument_group("brightness / color")
    g.add_argument("--no-brighten", action="store_true", help="don't lift dark footage")
    g.add_argument("--bright-target", type=float, default=0.30, help="average brightness to lift dark video to, 0-1 (default 0.30)")
    g.add_argument("--bright-strength", type=float, default=1.0, help="how much of the lift to apply, 0-1 (default 1.0)")
    g.add_argument("--contrast", action="store_true", help="extra local contrast (CLAHE) to pull details out of shadows")
    g.add_argument("--no-grade", action="store_true", help="no contrast/saturation grade")
    g.add_argument("--saturation", type=float, default=1.6, help="saturation multiplier (default 1.6)")
    g.add_argument("--crush", type=float, default=0.06, help="how much shadow to crush to black, 0-0.3 (default 0.06)")
    g.add_argument("--grade-contrast", type=float, default=0.6, help="S-curve contrast 0-1 (default 0.6)")
    g.add_argument("--no-recolor", action="store_true", help="no blue/pink/cyan highlight flashes")
    g.add_argument("--no-bloom", action="store_true", help="no glow around lights")
    g.add_argument("--bloom", type=float, default=1.0, help="glow strength (default 1.0)")

    g = p.add_argument_group("motion / glitch")
    g.add_argument("--no-shake", action="store_true", help="no jitter / zoom punches / zoom blur")
    g.add_argument("--punch", type=float, default=0.12, help="zoom punch size on beats (default 0.12 = 12%%)")
    g.add_argument("--no-mosh", action="store_true", help="no datamosh smear")
    g.add_argument("--mosh-chance", type=float, default=0.25, help="chance a beat starts a datamosh smear (default 0.25)")
    g.add_argument("--no-trails", action="store_true", help="no light trails")
    g.add_argument("--trails", type=float, default=0.75, help="trail length 0-0.95 (default 0.75)")
    g.add_argument("--no-flash", action="store_true", help="no white flashes / inverts on beats")
    g.add_argument("--letterbox", default="auto",
                   help="visible aspect ratio, e.g. 1.78 or 2.35; 'auto' (default) or 0 for none")

    g = p.add_argument_group("beats")
    g.add_argument("--no-beats", action="store_true", help="ignore the audio, use random hits")
    g.add_argument("--beat-sensitivity", type=float, default=1.0, help="higher = more beats detected (default 1.0)")
    return p


def default_args(**overrides):
    """All defaults as a namespace (used by the UI)."""
    a = build_parser().parse_args([])
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def main():
    args = build_parser().parse_args()
    if args.preview:
        args.duration = args.duration or 5
        args.max_side = args.max_side or 480

    if args.seed is not None:
        random.seed(args.seed)
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        sys.exit("ffmpeg/ffprobe not found. Install ffmpeg first (e.g. `sudo apt install ffmpeg`, `brew install ffmpeg`).")

    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    videos = sorted(f for f in INPUT_DIR.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS)
    if not videos:
        sys.exit(f"No videos in {INPUT_DIR}. Put your files there and run again.")

    ok = 0
    for src in videos:
        dst = output_path_for(src)
        if args.preview:
            dst = OUTPUT_DIR / f"{src.stem}_preview.mp4"
        try:
            ok += process_file(src, dst, args, preview=args.preview)
        except Exception as e:  # keep going with the other files
            print(f"  FAILED: {e}")
    print(f"\nDone: {ok}/{len(videos)} videos saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
