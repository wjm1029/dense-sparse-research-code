"""Extract frames 1/41/81 for manual subject review of the fixed Ditto20 set."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw


SELECTION = Path("/pfs/mayuema/dense/wan/outputs/dense4_hyperlora_4gpu_10k_ditto20_depth_normal/selection_manifest.json")
OUTPUT = Path("/pfs/mayuema/dense/wan/outputs/ditto20_matte_subject_review")
FRAME_IDS = (0, 40, 80)


def main() -> None:
    cases = json.loads(SELECTION.read_text(encoding="utf-8"))["cases"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tiles = []
    for case in cases:
        reader = imageio.get_reader(case["path"])
        try:
            frames = [Image.fromarray(reader.get_data(index)[..., :3]).resize((416, 240)) for index in FRAME_IDS]
        finally:
            reader.close()
        tile = Image.new("RGB", (1248, 274), "black")
        draw = ImageDraw.Draw(tile)
        draw.text((6, 4), case["case_name"], fill="white")
        for column, (frame_id, frame) in enumerate(zip(FRAME_IDS, frames)):
            x = column * 416
            tile.paste(frame, (x, 34))
            draw.text((x + 6, 38), f"frame {frame_id + 1}", fill="white", stroke_width=2, stroke_fill="black")
        tile.save(OUTPUT / f"{case['case_name']}_frames_1_41_81.png")
        tiles.append(tile)

    for sheet_index in range(4):
        sheet = Image.new("RGB", (1248, 1370), "black")
        for row, tile in enumerate(tiles[sheet_index * 5:(sheet_index + 1) * 5]):
            sheet.paste(tile, (0, row * 274))
        sheet.save(OUTPUT / f"subject_review_sheet_{sheet_index + 1}.png")


if __name__ == "__main__":
    main()
