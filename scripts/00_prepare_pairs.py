"""Build reproducible, evaluation-independent contrast pairs.

The benchmark items under ``data/processed`` are deliberately not used here.  This
keeps capability-vector fitting independent from the examples used in stage 05.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import string
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from mllm_self_steering.utils.io import atomic_save_json, sha256_file, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAIR_ROOT = PROJECT_ROOT / "pairs"
IMAGE_ROOT = PAIR_ROOT / "images"
GENERATOR_VERSION = "synthetic-controlled-v1.1"


LOGIC_POSITIVE = (
    "Use explicit step-by-step deductions from the stated premises; verify every "
    "implication before answering."
)
LOGIC_NEGATIVE = (
    "Use an immediate intuitive impression from the stated premises; avoid explicit "
    "deductions before answering."
)
QUANTITATIVE_POSITIVE = (
    "Translate every quantity into equations, calculate exactly, and check the "
    "arithmetic before answering."
)
QUANTITATIVE_NEGATIVE = (
    "Treat the quantities qualitatively, estimate intuitively, and avoid explicit "
    "arithmetic before answering."
)
RELEVANCE_POSITIVE = (
    "Select only evidence necessary for the question and explicitly ignore unrelated "
    "details before answering."
)
RELEVANCE_NEGATIVE = (
    "Treat every detail as equally relevant to the question and incorporate all "
    "details before answering."
)


NAMES = (
    "Ada",
    "Ben",
    "Cora",
    "Dylan",
    "Eli",
    "Faye",
    "Gita",
    "Hugo",
    "Iris",
    "Jon",
    "Kira",
    "Liam",
)
OBJECTS = (
    "amber key",
    "blue folder",
    "copper token",
    "green vase",
    "silver lamp",
    "violet card",
)


def _prompt(task: str, instruction: str) -> str:
    return f"{task.strip()}\n\nReasoning instruction: {instruction}"


def _logic_task(index: int, rng: random.Random) -> tuple[str, str]:
    a, b, c, d = rng.sample(NAMES, 4)
    obj = rng.choice(OBJECTS)
    family = index % 6
    if family == 0:
        task = (
            f"If {a} enters the archive, then {b} signs the ledger. If {b} signs the "
            f"ledger, then {c} locks the cabinet. {a} enters the archive. Must {c} lock "
            "the cabinet? Answer yes or no."
        )
        label = "implication_chain"
    elif family == 1:
        task = (
            f"In a line, {a} stands left of {b}, {b} stands left of {c}, and {d} stands "
            f"right of {c}. Who is second from the left?"
        )
        label = "transitive_order"
    elif family == 2:
        task = (
            f"Every item in cabinet R is metal. No metal item is transparent. The {obj} "
            "is in cabinet R. Can the item be transparent? Answer yes or no."
        )
        label = "set_exclusion"
    elif family == 3:
        task = (
            f"Exactly one of {a} and {b} attends the review. If {a} attends, {c} presents. "
            f"{c} does not present. Does {b} attend? Answer yes or no."
        )
        label = "exclusive_case"
    elif family == 4:
        task = (
            f"Four talks by {a}, {b}, {c}, and {d} occupy slots 1 through 4. {a} is "
            f"before {b}; {c} is immediately after {a}; {d} is in slot 4. Who is in "
            "slot 2?"
        )
        label = "constraint_schedule"
    else:
        task = (
            f"A sensor is active only if both switches P and Q are on. P is on. The "
            f"sensor is active. What must be true about Q?"
        )
        label = "necessary_condition"
    return task, label


def _quantitative_task(index: int, rng: random.Random) -> tuple[str, str]:
    family = index % 6
    if family == 0:
        count = rng.randint(4, 18)
        price = rng.randint(7, 35)
        task = f"A laboratory orders {count} identical filters at ${price} each. What is the total cost?"
        label = "unit_price"
    elif family == 1:
        price = rng.randrange(40, 201, 5)
        discount = rng.choice((10, 15, 20, 25, 30))
        task = f"An instrument costs ${price} and is discounted by {discount}%. What is the sale price?"
        label = "percentage"
    elif family == 2:
        speed = rng.randint(4, 15)
        hours = rng.randint(3, 12)
        task = f"A cart moves at {speed} meters per minute for {hours} minutes. How far does it travel?"
        label = "rate_time"
    elif family == 3:
        values = [rng.randint(8, 48) for _ in range(4)]
        task = f"Four measurements are {values[0]}, {values[1]}, {values[2]}, and {values[3]}. What is their arithmetic mean?"
        label = "average"
    elif family == 4:
        width = rng.randint(4, 18)
        height = rng.randint(4, 18)
        task = f"A rectangle is {width} cm wide and {height} cm high. What is its area in square centimeters?"
        label = "area"
    else:
        left = rng.randint(2, 9)
        right = rng.randint(2, 9)
        total = (left + right) * rng.randint(3, 12)
        task = f"Red and blue beads are in the ratio {left}:{right}, with {total} beads total. How many are red?"
        label = "ratio"
    return task, label


def _relevance_task(index: int, rng: random.Random) -> tuple[str, str]:
    a, b, c = rng.sample(NAMES, 3)
    family = index % 6
    if family == 0:
        x, y = rng.sample(range(12, 70), 2)
        task = (
            f"A field note says: {a}'s sample weighs {x} g; {b}'s sample weighs {y} g; "
            f"the room is painted white; {c} arrived at 09:10; the barometer reads 1012 hPa. "
            f"Whose sample is heavier, {a}'s or {b}'s?"
        )
        label = "numeric_evidence"
    elif family == 1:
        task = (
            f"A shipping log says the {rng.choice(OBJECTS)} is in crate 7. Crate 7 is on "
            "shelf B. The clerk wears glasses, the floor is gray, and lunch begins at noon. "
            "Which shelf contains the specified object?"
        )
        label = "two_hop_lookup"
    elif family == 2:
        task = (
            f"Only badge holders may enter Room K. {a} has a badge; {b} has no badge; "
            f"{c}'s coat is blue; the hallway has six lamps; the meeting lasts 40 minutes. "
            f"Between {a} and {b}, who may enter Room K?"
        )
        label = "rule_selection"
    elif family == 3:
        task = (
            f"The train to Northport leaves at 14:20 from platform 3. A train to Eastbay "
            f"leaves at 14:05 from platform 1. The station opened in 1912 and has a glass roof. "
            "Which platform should a traveler to Northport use?"
        )
        label = "schedule_lookup"
    elif family == 4:
        task = (
            f"Plant A received water and sunlight and grew 4 cm. Plant B received sunlight "
            "but no water and grew 0 cm. The pots are ceramic, the labels are green, and the "
            "window faces east. Which missing resource best explains Plant B's outcome?"
        )
        label = "causal_evidence"
    else:
        task = (
            f"A report lists: server X returned status 200; server Y returned status 503; "
            "the operator drank tea; the rack has twelve screws; the test ran on Tuesday. "
            "Which server failed the availability check?"
        )
        label = "diagnostic_evidence"
    return task, label


def _text_rows(
    capability: str,
    count: int,
    seed: int,
    builder: Any,
    positive_instruction: str,
    negative_instruction: str,
) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        rng = random.Random(seed + index)
        task, family = builder(index, rng)
        rows.append(
            {
                "pair_id": f"{capability}_{index:04d}",
                "positive_prompt": _prompt(task, positive_instruction),
                "negative_prompt": _prompt(task, negative_instruction),
                "image_paths": [],
                "metadata": {
                    "source": "synthetic_controlled",
                    "generator_version": GENERATOR_VERSION,
                    "seed": seed + index,
                    "task_family": family,
                    "task_sha256": hashlib.sha256(task.encode("utf-8")).hexdigest(),
                    "positive_instruction_chars": len(positive_instruction),
                    "negative_instruction_chars": len(negative_instruction),
                },
            }
        )
    return rows


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")


def _detail_image(index: int, seed: int) -> tuple[Image.Image, str]:
    rng = random.Random(seed + index)
    image = Image.new("RGB", (512, 384), "#f4f1e8")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((24, 24, 488, 360), radius=18, outline="#27374d", width=5)
    draw.text((48, 44), "CALIBRATION PANEL", fill="#27374d", font=_font(25))
    for row in range(4):
        y = 112 + row * 50
        draw.line((55, y, 450, y), fill="#8391a1", width=2)
        draw.ellipse((65, y - 12, 89, y + 12), fill=rng.choice(("#d1495b", "#2a9d8f", "#e9c46a")))
        draw.rectangle((120, y - 8, 340 + rng.randint(-35, 35), y + 8), fill="#587792")
    code = "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(5))
    draw.rectangle((350, 306, 468, 344), fill="#ffffff", outline="#27374d", width=2)
    draw.text((363, 316), code, fill="#111111", font=_font(15))
    return image, code


def _degrade_detail(image: Image.Image) -> Image.Image:
    small = image.resize((128, 96), Image.Resampling.BILINEAR)
    restored = small.resize(image.size, Image.Resampling.BILINEAR)
    return restored.filter(ImageFilter.GaussianBlur(radius=1.15))


def _spatial_image(index: int, seed: int) -> tuple[Image.Image, str]:
    rng = random.Random(seed + index)
    image = Image.new("RGB", (512, 512), "#fafafa")
    draw = ImageDraw.Draw(image)
    labels = list(string.ascii_uppercase[:16])
    palette = ("#e76f51", "#2a9d8f", "#457b9d", "#e9c46a")
    for cell, label in enumerate(labels):
        row, column = divmod(cell, 4)
        x0, y0 = column * 128, row * 128
        draw.rectangle((x0 + 3, y0 + 3, x0 + 124, y0 + 124), outline="#495057", width=2)
        color = palette[(cell + index) % len(palette)]
        shape = (cell + index) % 3
        if shape == 0:
            draw.ellipse((x0 + 36, y0 + 30, x0 + 92, y0 + 86), fill=color)
        elif shape == 1:
            draw.rectangle((x0 + 38, y0 + 32, x0 + 90, y0 + 84), fill=color)
        else:
            draw.polygon(
                ((x0 + 64, y0 + 25), (x0 + 96, y0 + 87), (x0 + 32, y0 + 87)),
                fill=color,
            )
        draw.text((x0 + 57, y0 + 96), label, fill="#111111", font=_font(18))
    target_row = rng.randrange(4)
    target_column = rng.randrange(3)
    target = labels[target_row * 4 + target_column]
    return image, target


def _shuffle_patches(
    image: Image.Image,
    index: int,
    seed: int,
    target_source: int,
) -> tuple[Image.Image, list[int]]:
    rng = random.Random(seed + index)
    permutation = list(range(16))
    while (
        any(source == destination for destination, source in enumerate(permutation))
        or permutation.index(target_source) % 4 == 3
    ):
        rng.shuffle(permutation)
    result = Image.new("RGB", image.size)
    for destination, source in enumerate(permutation):
        source_row, source_column = divmod(source, 4)
        destination_row, destination_column = divmod(destination, 4)
        patch = image.crop(
            (
                source_column * 128,
                source_row * 128,
                (source_column + 1) * 128,
                (source_row + 1) * 128,
            )
        )
        result.paste(patch, (destination_column * 128, destination_row * 128))
    return result, permutation


def _marker_code(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(2)) + str(rng.randrange(10, 100))


def _grounding_image(index: int, seed: int, *, include_target: bool) -> tuple[Image.Image, str, str]:
    rng = random.Random(seed + index + (0 if include_target else 1_000_000))
    image = Image.new("RGB", (512, 384), "#eef2f3")
    draw = ImageDraw.Draw(image)
    shapes = ("circle", "square", "triangle")
    palette = ("#e63946", "#2a9d8f", "#4361ee", "#f4a261")
    target_rng = random.Random(seed + index)
    target = _marker_code(target_rng)
    target_cell = target_rng.randrange(6)
    target_shape = shapes[target_rng.randrange(len(shapes))]
    used = {target}
    for cell in range(6):
        row, column = divmod(cell, 3)
        x0, y0 = 24 + column * 164, 28 + row * 174
        draw.rounded_rectangle((x0, y0, x0 + 136, y0 + 145), radius=12, fill="#ffffff", outline="#59636e", width=2)
        if include_target and cell == target_cell:
            code = target
            shape = target_shape
        else:
            code = _marker_code(rng)
            while code in used:
                code = _marker_code(rng)
            shape = shapes[rng.randrange(len(shapes))]
        used.add(code)
        color = palette[(cell + index) % len(palette)]
        cx, cy = x0 + 68, y0 + 65
        if shape == "circle":
            draw.ellipse((cx - 28, cy - 28, cx + 28, cy + 28), fill=color)
        elif shape == "square":
            draw.rectangle((cx - 27, cy - 27, cx + 27, cy + 27), fill=color)
        else:
            draw.polygon(((cx, cy - 32), (cx + 34, cy + 27), (cx - 34, cy + 27)), fill=color)
        draw.text((x0 + 44, y0 + 112), code, fill="#111111", font=_font(18))
    return image, target, target_shape


def _save_visual_pair(
    capability: str,
    index: int,
    positive: Image.Image,
    negative: Image.Image,
) -> tuple[Path, Path]:
    root = IMAGE_ROOT / capability
    root.mkdir(parents=True, exist_ok=True)
    positive_path = root / f"{index:04d}_positive.png"
    negative_path = root / f"{index:04d}_negative.png"
    positive.save(positive_path, format="PNG", optimize=True)
    negative.save(negative_path, format="PNG", optimize=True)
    return positive_path, negative_path


def _detail_rows(count: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        positive, code = _detail_image(index, seed)
        negative = _degrade_detail(positive)
        positive_path, negative_path = _save_visual_pair("vision_detail", index, positive, negative)
        rows.append(
            {
                "pair_id": f"vision_detail_{index:04d}",
                "prompt": "Read the five-character code printed in the lower-right inspection box.",
                "positive_image_paths": [_relative(positive_path)],
                "negative_image_paths": [_relative(negative_path)],
                "metadata": {
                    "source": "synthetic_controlled",
                    "generator_version": GENERATOR_VERSION,
                    "seed": seed + index,
                    "transform": "downsample_4x_then_bilinear_upscale_gaussian_blur",
                    "blur_radius": 1.15,
                    "target_code": code,
                },
            }
        )
    return rows


def _spatial_rows(count: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        positive, target = _spatial_image(index, seed)
        target_source = ord(target) - ord("A")
        negative, permutation = _shuffle_patches(
            positive,
            index,
            seed + 50_000,
            target_source,
        )
        positive_path, negative_path = _save_visual_pair("vision_spatial", index, positive, negative)
        rows.append(
            {
                "pair_id": f"vision_spatial_{index:04d}",
                "prompt": f"Which labeled tile is directly to the right of tile {target}?",
                "positive_image_paths": [_relative(positive_path)],
                "negative_image_paths": [_relative(negative_path)],
                "metadata": {
                    "source": "synthetic_controlled",
                    "generator_version": GENERATOR_VERSION,
                    "seed": seed + index,
                    "transform": "aligned_4x4_patch_derangement",
                    "permutation": permutation,
                    "target_has_right_neighbor_in_both_images": True,
                },
            }
        )
    return rows


def _grounding_rows(count: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        positive, target, target_shape = _grounding_image(index, seed, include_target=True)
        negative, _, _ = _grounding_image(index, seed, include_target=False)
        positive_path, negative_path = _save_visual_pair("vision_grounding", index, positive, negative)
        rows.append(
            {
                "pair_id": f"vision_grounding_{index:04d}",
                "prompt": f"Locate marker {target} and report the geometric shape inside its box.",
                "positive_image_paths": [_relative(positive_path)],
                "negative_image_paths": [_relative(negative_path)],
                "metadata": {
                    "source": "synthetic_controlled",
                    "generator_version": GENERATOR_VERSION,
                    "seed": seed + index,
                    "transform": "matched_panel_vs_same_style_unmatched_panel",
                    "target_marker": target,
                    "target_shape": target_shape,
                },
            }
        )
    return rows


def _manifest(rows: dict[str, list[dict[str, Any]]], seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "evaluation_independent": True,
        "capabilities": {},
    }
    for capability, values in rows.items():
        path = PAIR_ROOT / f"{capability}.jsonl"
        result["capabilities"][capability] = {
            "count": len(values),
            "jsonl_sha256": sha256_file(path),
            "image_files": sum(
                len(row.get("positive_image_paths", []))
                + len(row.get("negative_image_paths", []))
                for row in values
            ),
        }
    return result


def build_pairs(count: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    if count < 2:
        raise ValueError("count must be at least two")
    rows = {
        "text_logic": _text_rows(
            "text_logic", count, seed + 10_000, _logic_task, LOGIC_POSITIVE, LOGIC_NEGATIVE
        ),
        "text_quantitative": _text_rows(
            "text_quantitative",
            count,
            seed + 20_000,
            _quantitative_task,
            QUANTITATIVE_POSITIVE,
            QUANTITATIVE_NEGATIVE,
        ),
        "text_relevance": _text_rows(
            "text_relevance",
            count,
            seed + 30_000,
            _relevance_task,
            RELEVANCE_POSITIVE,
            RELEVANCE_NEGATIVE,
        ),
        "vision_detail": _detail_rows(count, seed + 40_000),
        "vision_spatial": _spatial_rows(count, seed + 50_000),
        "vision_grounding": _grounding_rows(count, seed + 60_000),
    }
    for capability, values in rows.items():
        write_jsonl(PAIR_ROOT / f"{capability}.jsonl", values)
    atomic_save_json(PAIR_ROOT / "manifest.json", _manifest(rows, seed))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    rows = build_pairs(args.count, args.seed)
    print(json.dumps({name: len(values) for name, values in rows.items()}, indent=2))


if __name__ == "__main__":
    main()
