"""Turn the team's art (aisutd/pokerpilot main: Assets/Profiles/PlayerImages,
chip_table_images) into the web assets in static/art.

Needs Pillow (pip install pillow), which the site itself does not use.
usage:  python tools/make_art.py <folder holding Assets/ and chip_table_images/> static/art
"""

import sys
from collections import deque
from pathlib import Path

from PIL import Image

src, out = Path(sys.argv[1]), Path(sys.argv[2])
players = src / 'Assets' / 'Profiles' / 'PlayerImages'
sheets = src / 'chip_table_images'

# The plain (unframed) portrait of each player.
PORTRAITS = {
    1: 'P1/eeb7bef6-2f6e-4249-9b50-4dc60fe02d07-ezgif.com-crop (3).png',
    2: 'P2/eeb7bef6-2f6e-4249-9b50-4dc60fe02d07-ezgif.com-crop.png',
    3: 'P3/eeb7bef6-2f6e-4249-9b50-4dc60fe02d07-ezgif.com-crop.png',
    4: 'P4/eeb7bef6-2f6e-4249-9b50-4dc60fe02d07-ezgif.com-crop.png',
    5: 'P5/eeb7bef6-2f6e-4249-9b50-4dc60fe02d07-ezgif.com-crop.png',
    6: 'P6/ezgif-6a850e53dc0ccb54.png',
    7: 'P7/ezgif-6e897ba077734205.png',
    8: 'P8/ezgif-6f6c74b7b52bc182.png',
    9: 'P9/ezgif-6288f368b909a28a.png',
    10: 'P10/ezgif-6651c20bafdb5baa.png',
}
CHIPS = ('red', 'blue', 'green', 'black', 'white', 'yellow', 'purple')
ICONS = ('menu', 'settings', 'sound')


def is_checker(rgb, spread=18, floor=196):
    """The painted-in 'transparent' checkerboard: light, neutral greys."""
    return max(rgb) - min(rgb) <= spread and min(rgb) >= floor


def cut_out(image: Image.Image) -> Image.Image:
    """Remove the checkerboard by flood-filling it from the edges. The dark
    outline around each character stops the fill, so white shirts survive."""
    rgb = image.convert('RGB')
    w, h = rgb.size
    px = rgb.load()
    clear = [[False] * w for _ in range(h)]
    queue = deque()

    # The body is cut off flat, with a strip of checkerboard under it and no
    # outline along the cut, so a white shirt would let the fill in from
    # below. Clear everything under the cut and fill only above it.
    solid = lambda y: sum(not is_checker(px[x, y], spread=30, floor=150) for x in range(w))
    cut = max(y for y in range(h) if solid(y) > 10)
    for y in range(cut + 1, h):
        clear[y] = [True] * w

    for x in range(w):
        queue.append((x, 0))
    for y in range(cut + 1):
        queue.extend([(0, y), (w - 1, y)])

    while queue:
        x, y = queue.popleft()
        if not (0 <= x < w and 0 <= y <= cut) or clear[y][x] or not is_checker(px[x, y]):
            continue
        clear[y][x] = True
        queue.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])

    # Soft fringe between the checkerboard and the outline.
    for _ in range(2):
        fringe = [
            (x, y) for y in range(h) for x in range(w)
            if not clear[y][x] and is_checker(px[x, y], spread=30, floor=150) and any(
                0 <= x + dx < w and 0 <= y + dy < h and clear[y + dy][x + dx]
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        ]
        for x, y in fringe:
            clear[y][x] = True

    # Keep only the character: the largest connected shape. Drops specks left
    # at the image edges by the original crop.
    seen = [[False] * w for _ in range(h)]
    best = []
    for sy in range(h):
        for sx in range(w):
            if clear[sy][sx] or seen[sy][sx]:
                continue
            shape, queue = [], deque([(sx, sy)])
            seen[sy][sx] = True
            while queue:
                x, y = queue.popleft()
                shape.append((x, y))
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < w and 0 <= ny < h and not clear[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        queue.append((nx, ny))
            if len(shape) > len(best):
                best = shape

    rgba = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    data = rgba.load()
    for x, y in best:
        data[x, y] = px[x, y] + (255,)
    return rgba.crop(rgba.getbbox())


def square(image: Image.Image) -> Image.Image:
    """Pad to a square: centred across, resting on the bottom edge (the
    shoulders are cut off there in the art)."""
    w, h = image.size
    side = max(w, h)
    canvas = Image.new('RGBA', (side, side), (0, 0, 0, 0))
    canvas.paste(image, ((side - w) // 2, side - h))
    return canvas


def runs(columns):
    """[start, end) ranges of consecutive True values."""
    found, start = [], None
    for i, on in enumerate(columns + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            found.append((start, i))
            start = None
    return found


def slice_row(sheet: Image.Image, top: int, bottom: int, names, width: int):
    alpha = sheet.getchannel('A')
    w = sheet.width
    cols = [any(alpha.getpixel((x, y)) > 40 for y in range(top, bottom, 2)) for x in range(w)]
    parts = [r for r in runs(cols) if r[1] - r[0] > 40]
    assert len(parts) == len(names), (len(parts), parts)
    for name, (x0, x1) in zip(names, parts):
        piece = sheet.crop((x0, top, x1, bottom))
        piece = piece.crop(piece.getchannel('A').point(lambda a: 255 if a > 40 else 0).getbbox())
        scale = width / piece.width
        piece = piece.resize((width, round(piece.height * scale)), Image.LANCZOS)
        yield name, piece


(out / 'portraits').mkdir(parents=True, exist_ok=True)
for number, name in PORTRAITS.items():
    portrait = square(cut_out(Image.open(players / name)))
    portrait.save(out / 'portraits' / f'p{number}.png', optimize=True)
    print(f'p{number}', portrait.size)

sheet = Image.open(next(sheets.glob('6d4f21a1*'))).convert('RGBA')
alpha = sheet.getchannel('A')
rows = runs([any(alpha.getpixel((x, y)) > 40 for x in range(0, sheet.width, 2)) for y in range(sheet.height)])
rows = [r for r in rows if r[1] - r[0] > 40]
assert len(rows) == 2, rows
(out / 'chips').mkdir(exist_ok=True)
for name, piece in slice_row(sheet, *rows[0], CHIPS, 96):
    piece.save(out / 'chips' / f'{name}.png', optimize=True)
    print('chip', name, piece.size)
for name, piece in slice_row(sheet, *rows[1], ICONS, 96):
    piece.save(out / f'icon-{name}.png', optimize=True)
    print('icon', name, piece.size)

    if name == 'menu':
        # An empty button frame (the chat button in the mock-up): paint the
        # menu bars over with the panel colour.
        frame = piece.copy()
        px = frame.load()
        w, h = frame.size
        bars = [(x, y) for y in range(h // 5, h * 4 // 5) for x in range(w // 5, w * 4 // 5)
                if px[x, y][3] > 200 and px[x, y][0] > 180 and px[x, y][1] > 120]
        # (wide enough to take the bars' drop shadows too)
        x0, x1 = min(x for x, _ in bars) - 5, max(x for x, _ in bars) + 5
        y0, y1 = min(y for _, y in bars) - 4, max(y for _, y in bars) + 6
        for y in range(y0, y1 + 1):
            fill = px[x0 - 2, y]  # this row's panel colour, keeping its shading
            for x in range(x0, x1 + 1):
                px[x, y] = fill
        frame.save(out / 'icon-frame.png', optimize=True)
        print('icon frame', frame.size)

# The crown over each player's plate, from player 1's plate art.
plate = Image.open(players / 'P1' / 'eeb7bef6-2f6e-4249-9b50-4dc60fe02d07-ezgif.com-crop (4).png').convert('RGB')
blue = lambda rgb: rgb[2] > 120 and rgb[0] < 90
top = next(y for y in range(plate.height) if sum(blue(plate.getpixel((x, y))) for x in range(plate.width)) > plate.width // 3)
crown_area = plate.crop((0, 0, plate.width, top - 4))  # clear of the plate's outline
crown = cut_out(crown_area)
crown.save(out / 'crown.png', optimize=True)
print('crown', crown.size)

# The table and room, for the wide layout.
table = Image.open(next(sheets.glob('561a1f12*'))).convert('RGB')
table.save(out / 'table.jpg', quality=92)
print('table', table.size)

# The felt inside the table's neon rim (the straight part, clear of the ends),
# for the narrow layout, which draws its own table.
felt = table.crop((272, 142, 752, 386))
felt.save(out / 'felt.jpg', quality=90)
print('felt', felt.size)
