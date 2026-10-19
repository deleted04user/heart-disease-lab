"""Build a captioned MP4 walkthrough from actual, locally captured app screenshots."""

import argparse
from pathlib import Path

from _common import ROOT


def build_demo(screenshots: Path, output: Path) -> dict:
    try:
        import imageio_ffmpeg
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except ImportError as exc:
        raise RuntimeError(
            "Install optional demo dependencies: python -m pip install -r requirements-demo.txt"
        ) from exc
    slides = [
        (
            "overview.png",
            "01 · Real data, explicit provenance",
            "303 UCI Cleveland observations. Educational research only.",
            4,
        ),
        (
            "prediction_form.png",
            "02 · Explore a public dataset example",
            "Meaningful category labels, strict validation and a reset control.",
            4,
        ),
        (
            "prediction_result.png",
            "03 · A model output, not a diagnosis",
            "Local inference; the optional download omits entered measurements.",
            4,
        ),
        (
            "evaluation.png",
            "04 · Inspect performance and uncertainty",
            "Repeated training CV, nested probability checks and an audited holdout.",
            5,
        ),
        (
            "external.png",
            "05 · Check transfer to another site",
            "Frozen pipeline, explicit missingness and subgroup limitations.",
            5,
        ),
        (
            "explainability.png",
            "06 · Explain associations responsibly",
            "Original-feature permutation importance does not imply causation.",
            4,
        ),
    ]
    missing = [name for name, _, _, _ in slides if not (screenshots / name).is_file()]
    if missing:
        raise FileNotFoundError("Capture actual app screenshots first: " + ", ".join(missing))
    font_candidates = [
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    font_path = next((path for path in font_candidates if path.exists()), None)
    title_font = (
        ImageFont.truetype(str(font_path), 25) if font_path else ImageFont.load_default(size=25)
    )
    body_font = (
        ImageFont.truetype(str(font_path), 18) if font_path else ImageFont.load_default(size=18)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fps, size = 12, (1280, 816)
    writer = imageio_ffmpeg.write_frames(
        str(output),
        size,
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        quality=8,
        ffmpeg_log_level="error",
    )
    writer.send(None)
    frames = 0
    try:
        for filename, title, caption, duration in slides:
            with Image.open(screenshots / filename) as image:
                capture = ImageOps.contain(image.convert("RGB"), (1280, 720))
            canvas = Image.new("RGB", size, "#F7F9FC")
            canvas.paste(capture, ((1280 - capture.width) // 2, (720 - capture.height) // 2))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle([0, 720, 1280, 816], fill="#172B4D")
            draw.text((28, 732), title, font=title_font, fill="white")
            draw.text((28, 772), caption, font=body_font, fill="#C8D6EE")
            frame = np.asarray(canvas)
            for _ in range(duration * fps):
                writer.send(frame)
                frames += 1
    finally:
        writer.close()
    # Decode one actual video frame and verify duration/dimensions via FFmpeg.
    reader = imageio_ffmpeg.read_frames(str(output), pix_fmt="rgb24")
    metadata = next(reader)
    first = next(reader)
    reader.close()
    if tuple(metadata["size"]) != size or len(first) != size[0] * size[1] * 3:
        raise RuntimeError("Generated demo failed video-frame verification.")
    return {
        "output": str(output.resolve()),
        "seconds": frames / fps,
        "fps": fps,
        "size": list(size),
        "kind": "Captioned walkthrough assembled from actual app screenshots; no narration or simulated live recording",
    }


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshots", type=Path, default=ROOT / "docs/screenshots")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/demo.mp4")
    args = parser.parse_args()
    print(json.dumps(build_demo(args.screenshots, args.output), indent=2))
