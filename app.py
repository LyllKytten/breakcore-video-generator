#!/usr/bin/env python3
"""Simple web UI for the breakcore video maker.

    python app.py
opens http://127.0.0.1:7860 in your browser.
"""
import shutil
import tempfile
import time
from pathlib import Path

import gradio as gr

import hud_boxes as hb

PREVIEW_DIR = Path(tempfile.gettempdir()) / "breakcore_preview"
PREVIEW_DIR.mkdir(exist_ok=True)
hb.OUTPUT_DIR.mkdir(exist_ok=True)

EFFECTS = ["hud", "brighten", "grade", "shake", "mosh", "trails", "recolor", "bloom", "flash", "beats"]

# ---------------------------------------------------------------------------
# Translations
# ---------------------------------------------------------------------------
T = {
    "ru": {
        "title": "# 🎛️ Breakcore Video Maker\nЗагрузи видео, добавь музыку, посмотри быстрый предпросмотр и сделай финальный рендер.",
        "lang": "Язык / Language",
        "video": "Видео (любой формат: mp4, mov, mkv, webm, avi…)",
        "music": "Музыка (необязательно)",
        "music_mode": "Звук",
        "mode_original": "Только звук видео",
        "mode_replace": "Заменить звук музыкой",
        "mode_mix": "Музыка + тихий звук видео",
        "music_offset": "Начать музыку с (сек)",
        "loop_video": "Повторять видео до конца музыки",
        "music_adapt": "Эффекты следуют громкости музыки",
        "intensity": "Интенсивность (0.5 спокойно · 1 норм · 2 хаос)",
        "effects": "Эффекты",
        "eff_hud": "Рамки с координатами",
        "eff_brighten": "Осветление тёмного видео",
        "eff_grade": "Цвет: контраст + насыщенность",
        "eff_shake": "Тряска и зум на битах",
        "eff_mosh": "Датамош (размазывание)",
        "eff_trails": "Шлейфы от огней",
        "eff_recolor": "Перекраска бликов (синий/розовый)",
        "eff_bloom": "Свечение огней",
        "eff_flash": "Вспышки и инверсия",
        "eff_beats": "Синхронизация с битом",
        "letterbox": "Чёрные полосы (кино-формат)",
        "lb_auto": "Авто", "lb_none": "Нет",
        "advanced": "Дополнительно",
        "saturation": "Насыщенность",
        "punch": "Сила зума на битах",
        "mosh_chance": "Частота датамоша",
        "trails_len": "Длина шлейфов",
        "max_boxes": "Макс. рамок в кадре",
        "beat_sens": "Чувствительность к битам",
        "bright_target": "Яркость осветления",
        "seed": "Seed (0 = каждый раз случайно)",
        "preview_start": "Предпросмотр: начало (сек)",
        "preview_len": "Предпросмотр: длина (сек)",
        "preview_btn": "⚡ Быстрый предпросмотр",
        "render_btn": "🎬 Рендер всего видео",
        "result": "Результат",
        "download": "Скачать файл",
        "status": "Статус",
        "no_video": "⚠️ Сначала загрузи видео.",
        "rendering": "Рендер…",
        "previewing": "Предпросмотр…",
        "done_preview": "✅ Предпросмотр готов за {t:.1f} с ({beats})",
        "done_render": "✅ Готово за {t:.1f} с. Сохранено: `{path}`",
        "failed": "❌ Ошибка: {err}",
        "beats_music": "биты из музыки",
        "beats_video": "биты из звука видео",
        "beats_random": "звука нет — случайные удары",
    },
    "en": {
        "title": "# 🎛️ Breakcore Video Maker\nUpload a video, add music, check a fast preview, then render the full video.",
        "lang": "Язык / Language",
        "video": "Video (any format: mp4, mov, mkv, webm, avi…)",
        "music": "Music (optional)",
        "music_mode": "Sound",
        "mode_original": "Video's own sound only",
        "mode_replace": "Replace sound with music",
        "mode_mix": "Music + quiet video sound",
        "music_offset": "Start music at (sec)",
        "loop_video": "Loop the video until the music ends",
        "music_adapt": "Effects follow the music's loudness",
        "intensity": "Intensity (0.5 calm · 1 normal · 2 chaos)",
        "effects": "Effects",
        "eff_hud": "Tracking boxes with coordinates",
        "eff_brighten": "Brighten dark footage",
        "eff_grade": "Color: contrast + saturation",
        "eff_shake": "Shake and zoom on beats",
        "eff_mosh": "Datamosh smear",
        "eff_trails": "Light trails",
        "eff_recolor": "Recolor highlights (blue/pink)",
        "eff_bloom": "Glow around lights",
        "eff_flash": "Flashes and inverts",
        "eff_beats": "Sync to the beat",
        "letterbox": "Black bars (cinema format)",
        "lb_auto": "Auto", "lb_none": "None",
        "advanced": "Advanced",
        "saturation": "Saturation",
        "punch": "Zoom punch on beats",
        "mosh_chance": "Datamosh frequency",
        "trails_len": "Trail length",
        "max_boxes": "Max boxes per frame",
        "beat_sens": "Beat sensitivity",
        "bright_target": "Brighten level",
        "seed": "Seed (0 = random every time)",
        "preview_start": "Preview: start (sec)",
        "preview_len": "Preview: length (sec)",
        "preview_btn": "⚡ Fast preview",
        "render_btn": "🎬 Render full video",
        "result": "Result",
        "download": "Download file",
        "status": "Status",
        "no_video": "⚠️ Upload a video first.",
        "rendering": "Rendering…",
        "previewing": "Previewing…",
        "done_preview": "✅ Preview ready in {t:.1f} s ({beats})",
        "done_render": "✅ Done in {t:.1f} s. Saved to: `{path}`",
        "failed": "❌ Error: {err}",
        "beats_music": "beats from the music",
        "beats_video": "beats from the video's sound",
        "beats_random": "no sound — random hits",
    },
}

LANGS = [("Русский", "ru"), ("English", "en")]


def mode_choices(t):
    return [(t["mode_original"], "original"), (t["mode_replace"], "replace"), (t["mode_mix"], "mix")]


def effect_choices(t):
    return [(t[f"eff_{e}"], e) for e in EFFECTS]


def lb_choices(t):
    return [(t["lb_auto"], "auto"), ("16:9", "1.78"), ("2.35:1", "2.35"), ("1:1", "1.0"), (t["lb_none"], "0")]


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def build_args(intensity, effects, letterbox, saturation, punch, mosh_chance, trails, max_boxes,
               beat_sens, bright_target, seed, music, music_mode, music_offset, loop_video, music_adapt):
    a = hb.default_args()
    a.intensity = float(intensity)
    for e in EFFECTS:
        setattr(a, f"no_{e}", e not in (effects or []))
    a.letterbox = letterbox or "auto"
    a.saturation = float(saturation)
    a.punch = float(punch)
    a.mosh_chance = float(mosh_chance)
    a.trails = float(trails)
    a.max_boxes = int(max_boxes)
    a.beat_sensitivity = float(beat_sens)
    a.bright_target = float(bright_target)
    a.seed = int(seed) if seed else None
    use_music = bool(music) and music_mode != "original"
    a.music = music if use_music else None
    a.music_mode = music_mode if use_music else "replace"
    a.music_offset = float(music_offset or 0)
    a.loop_video = bool(loop_video) and use_music
    a.music_adapt = bool(music_adapt)
    return a


def beats_note(t, a, video):
    if a.no_beats:
        return t["beats_random"]
    if a.music:
        return t["beats_music"]
    return t["beats_video"] if hb.probe(video)[4] else t["beats_random"]


def run(lang, video, preview, p_start, p_len, *settings, progress=gr.Progress()):
    t = T[lang]
    if not video:
        return None, None, t["no_video"]
    a = build_args(*settings)
    src = Path(video)
    if preview:
        a.start, a.duration, a.max_side = float(p_start), float(p_len), 480
        dst = PREVIEW_DIR / f"preview_{int(time.time() * 1000)}.mp4"
    else:
        dst = hb.output_path_for(src)

    desc = t["previewing"] if preview else t["rendering"]
    progress(0, desc=desc)
    t0 = time.time()
    try:
        ok = hb.process_file(src, dst, a, preview=preview,
                             progress=lambda f: progress(f, desc=desc), log=lambda *_: None)
        if not ok:
            raise RuntimeError("ffmpeg failed")
    except Exception as e:
        return None, None, t["failed"].format(err=e)

    took = time.time() - t0
    if preview:
        return str(dst), None, t["done_preview"].format(t=took, beats=beats_note(t, a, src))
    playable = dst.suffix.lower() in (".mp4", ".m4v", ".mov", ".webm")
    return (str(dst) if playable else None), str(dst), t["done_render"].format(t=took, path=dst)


def on_video_upload(video):
    if not video:
        return gr.update()
    try:
        w, h, fps, frames, _ = hb.probe(video)
        dur = frames / fps if frames else hb.media_duration(video)
    except Exception:
        return gr.update()
    return gr.update(maximum=max(1.0, round(dur, 1)), value=0)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
video_types = sorted(hb.VIDEO_EXTS) + sorted(e.upper() for e in hb.VIDEO_EXTS)
t0 = T["ru"]

with gr.Blocks(title="Breakcore Video Maker") as demo:
    lang = gr.Radio(LANGS, value="ru", label=t0["lang"])
    title = gr.Markdown(t0["title"])

    with gr.Row():
        with gr.Column(scale=1):
            video = gr.File(label=t0["video"], file_types=video_types, type="filepath")
            music = gr.Audio(label=t0["music"], type="filepath", sources=["upload"])
            music_mode = gr.Radio(mode_choices(t0), value="replace", label=t0["music_mode"])
            with gr.Row():
                music_offset = gr.Number(value=0, minimum=0, label=t0["music_offset"])
                loop_video = gr.Checkbox(value=False, label=t0["loop_video"])
            music_adapt = gr.Checkbox(value=True, label=t0["music_adapt"])

            intensity = gr.Slider(0.2, 2.0, value=1.0, step=0.05, label=t0["intensity"])
            effects = gr.CheckboxGroup(effect_choices(t0), value=EFFECTS, label=t0["effects"])
            letterbox = gr.Dropdown(lb_choices(t0), value="auto", label=t0["letterbox"])

            with gr.Accordion(t0["advanced"], open=False) as advanced:
                saturation = gr.Slider(0.5, 3.0, value=1.6, step=0.05, label=t0["saturation"])
                punch = gr.Slider(0, 0.4, value=0.12, step=0.01, label=t0["punch"])
                mosh_chance = gr.Slider(0, 1, value=0.25, step=0.05, label=t0["mosh_chance"])
                trails = gr.Slider(0, 0.95, value=0.75, step=0.05, label=t0["trails_len"])
                max_boxes = gr.Slider(1, 40, value=14, step=1, label=t0["max_boxes"])
                beat_sens = gr.Slider(0.3, 3.0, value=1.0, step=0.1, label=t0["beat_sens"])
                bright_target = gr.Slider(0.1, 0.6, value=0.30, step=0.01, label=t0["bright_target"])
                seed = gr.Number(value=0, precision=0, label=t0["seed"])

        with gr.Column(scale=1):
            with gr.Row():
                p_start = gr.Slider(0, 60, value=0, step=0.1, label=t0["preview_start"])
                p_len = gr.Slider(1, 15, value=5, step=0.5, label=t0["preview_len"])
            with gr.Row():
                preview_btn = gr.Button(t0["preview_btn"], variant="primary")
                render_btn = gr.Button(t0["render_btn"])
            out_video = gr.Video(label=t0["result"], autoplay=True)
            out_file = gr.File(label=t0["download"])
            status = gr.Markdown("")

    settings = [intensity, effects, letterbox, saturation, punch, mosh_chance, trails, max_boxes,
                beat_sens, bright_target, seed, music, music_mode, music_offset, loop_video, music_adapt]
    outputs = [out_video, out_file, status]

    preview_flag, render_flag = gr.State(True), gr.State(False)
    preview_btn.click(run, [lang, video, preview_flag, p_start, p_len, *settings], outputs)
    render_btn.click(run, [lang, video, render_flag, p_start, p_len, *settings], outputs)
    video.change(on_video_upload, video, p_start)

    # --- language switch: relabel everything
    labelled = {
        video: "video", music: "music", music_offset: "music_offset", loop_video: "loop_video",
        music_adapt: "music_adapt", intensity: "intensity", saturation: "saturation", punch: "punch",
        mosh_chance: "mosh_chance", trails: "trails_len", max_boxes: "max_boxes", beat_sens: "beat_sens",
        bright_target: "bright_target", seed: "seed", p_start: "preview_start", p_len: "preview_len",
        out_video: "result", out_file: "download",
    }

    def switch_language(code):
        t = T[code]
        ups = [gr.update(value=t["title"])]
        ups += [gr.update(label=t[key]) for key in labelled.values()]
        ups += [
            gr.update(label=t["music_mode"], choices=mode_choices(t)),
            gr.update(label=t["effects"], choices=effect_choices(t)),
            gr.update(label=t["letterbox"], choices=lb_choices(t)),
            gr.update(label=t["advanced"]),
            gr.update(value=t["preview_btn"]),
            gr.update(value=t["render_btn"]),
        ]
        return ups

    lang.change(
        switch_language, lang,
        [title, *labelled.keys(), music_mode, effects, letterbox, advanced, preview_btn, render_btn],
    )


if __name__ == "__main__":
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        raise SystemExit("ffmpeg/ffprobe not found. Install ffmpeg first.")
    demo.launch(inbrowser=True, allowed_paths=[str(PREVIEW_DIR), str(hb.OUTPUT_DIR)])
