"""Render the eight saved fixed pairs without overflowing table cells."""

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def wrapped(text, limit, width):
    return textwrap.fill(textwrap.shorten(text, width=limit, placeholder=" …"), width=width)


def main():
    records = [
        json.loads(line) for line in (ROOT / "data/eval/side_by_side.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    fixed = [row for row in records if row["category"] != "heldout"]
    assert len(fixed) == 8
    cells = [["ID", "Câu hỏi", "SFT", "SFT + DPO"]] + [
        [row["id"], wrapped(row["prompt"], 140, 30), wrapped(row["sft"], 210, 48), wrapped(row["dpo"], 210, 48)]
        for row in fixed
    ]
    heights = [0.35 + 0.16 * max(text.count("\n") + 1 for text in row) for row in cells]
    total = sum(heights)
    figure, axis = plt.subplots(figsize=(16, total + 1.0))
    axis.axis("off")
    table = axis.table(cellText=cells, cellLoc="left", colWidths=[0.05, 0.23, 0.36, 0.36], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for row_index, height in enumerate(heights):
        for column in range(4):
            cell = table[(row_index, column)]
            cell.set_height(height / total)
            cell.PAD = 0.035
            if row_index == 0:
                cell.set_facecolor("#2e548a")
                cell.set_text_props(color="white", weight="bold")
            elif row_index % 2 == 0:
                cell.set_facecolor("#f1f4f8")
    axis.set_title("8 câu cố định · greedy decode giống nhau · trích ngắn, không phải câu trả lời đầy đủ", pad=14)
    figure.tight_layout()
    output = ROOT / "submission/screenshots/04-side-by-side-table.png"
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(output.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
