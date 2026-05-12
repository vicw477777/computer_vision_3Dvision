import os
import sys
import math
from PIL import Image, ImageOps, ImageDraw

if len(sys.argv) < 3:
    print("Usage: python make_contact_sheet.py <input_dir> <output_png> [thumb_w] [thumb_h]")
    sys.exit(1)

input_dir = sys.argv[1]
output_png = sys.argv[2]
thumb_w = int(sys.argv[3]) if len(sys.argv) > 3 else 320
thumb_h = int(sys.argv[4]) if len(sys.argv) > 4 else 180

files = sorted([
    os.path.join(input_dir, f)
    for f in os.listdir(input_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

if not files:
    raise ValueError(f"No images found in {input_dir}")

n = len(files)
cols = min(4, n)
rows = math.ceil(n / cols)

margin = 20
pad = 10
title_h = 40

sheet_w = cols * thumb_w + (cols - 1) * pad + 2 * margin
sheet_h = rows * (thumb_h + title_h) + (rows - 1) * pad + 2 * margin

canvas = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

for i, path in enumerate(files):
    img = Image.open(path).convert("RGB")
    img.thumbnail((thumb_w, thumb_h))
    framed = ImageOps.pad(img, (thumb_w, thumb_h), color=(245, 245, 245))
    r = i // cols
    c = i % cols
    x = margin + c * (thumb_w + pad)
    y = margin + r * (thumb_h + title_h + pad)

    canvas.paste(framed, (x, y))
    label = os.path.basename(path)
    draw.text((x, y + thumb_h + 8), label, fill=(0, 0, 0))

canvas.save(output_png)
print(f"Saved contact sheet: {output_png}")