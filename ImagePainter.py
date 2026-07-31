from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageFont
from tkinter import Tk, filedialog
import os
import math 
import random
from collections import deque
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def add_borders(img, white_border=100, black_border=10):
    img = ImageOps.expand(img, border=white_border, fill="white")
    img = ImageOps.expand(img, border=black_border, fill="black")
    return img


def expand_canvas(img, width, height, color=(255, 255, 255)):
    canvas = Image.new("RGB", (width, height), color)

    x = (width - img.width) // 2
    y = (height - img.height) // 2

    canvas.paste(img, (x, y))
    return canvas

def save_grayscale(img, image_path):
    gray = img.convert("L")

    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_gray{ext}"

    gray.save(output_path)

def save_poster(img, image_path, bits=3):
    poster = ImageOps.posterize(img.convert("RGB"), bits)

    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_poster{bits}{ext}"

    poster.save(output_path)

def save_quantize(img, image_path, colors=8,methodn=1):
    img = img.convert("RGB")
    quant = img.quantize(colors=colors,method=methodn).convert("RGB")

    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_quant{colors}{ext}"

    quant.save(output_path)

def save_edge(img, image_path):
    edge = img.convert("L").filter(ImageFilter.FIND_EDGES)
    
    for _ in range(2):
        edge = edge.filter(ImageFilter.MaxFilter(3))

    edge = edge.point(lambda x: 255 if x > 10 else 0)

    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_edge{ext}"

    edge.save(output_path)

def save_edge_blur(img, image_path):
    edge_blur = img.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_blur = edge_blur.point(lambda p: 255 if p > 5 else 0)
    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_edge_blur{ext}"

    edge_blur.save(output_path)

def save_sobel_edges(img, image_path, threshold=None):
    gray = img.convert("L")

    sobel_x_kernel = ImageFilter.Kernel(
        (3, 3),
        [-1, 0, 1,
         -2, 0, 2,
         -1, 0, 1],
        scale=1
    )

    sobel_y_kernel = ImageFilter.Kernel(
        (3, 3),
        [-1, -2, -1,
          0,  0,  0,
          1,  2,  1],
        scale=1
    )

    gx = np.array(gray.filter(sobel_x_kernel), dtype=np.float32)
    gy = np.array(gray.filter(sobel_y_kernel), dtype=np.float32)

    magnitude = np.sqrt(gx**2 + gy**2)

    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max() * 255

    edges = Image.fromarray(magnitude.astype(np.uint8))

    if threshold is not None:
        edges = edges.point(lambda p: 255 if p > threshold else 0)

    base, ext = os.path.splitext(image_path)
    if threshold is not None:
        output_path = f"{base}_sobel_t{threshold}{ext}"
    else:
        output_path = f"{base}_sobel{ext}"
    print(output_path)
    edges.save(output_path)
    return output_path

def save_quantize_sobel(img, image_path, colors=2, threshold=None):
    # Quantize to N colors
    quantized = img.convert("RGB").quantize(colors=colors)
    gray = quantized.convert("L")

    sobel_x_kernel = ImageFilter.Kernel(
        (3, 3),
        [-1, 0, 1,
         -2, 0, 2,
         -1, 0, 1],
        scale=1
    )

    sobel_y_kernel = ImageFilter.Kernel(
        (3, 3),
        [-1, -2, -1,
          0,  0,  0,
          1,  2,  1],
        scale=1
    )

    gx = np.array(gray.filter(sobel_x_kernel), dtype=np.float32)
    gy = np.array(gray.filter(sobel_y_kernel), dtype=np.float32)

    magnitude = np.sqrt(gx**2 + gy**2)

    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max() * 255

    edges = Image.fromarray(magnitude.astype(np.uint8))

    if threshold is not None:
        edges = edges.point(lambda p: 255 if p > threshold else 0)

    base, ext = os.path.splitext(image_path)

    suffix = f"_quantize{colors}_sobel"
    if threshold is not None:
        suffix += f"_t{threshold}"

    output_path = f"{base}{suffix}{ext}"

    edges.save(output_path)
    print(output_path)
    return output_path

def draw_color_centroids(img, image_path, samples=64):

    result = img.convert("RGB")

    pixels = result.load()
    width, height = result.size

    draw = ImageDraw.Draw(result)

    r = max(8, int(width * 0.025))

    placed = []

    for _ in range(samples):

        for _ in range(1000):

            x = random.randint(r, width - r)
            y = random.randint(r, height - r)

            overlap = False

            for ox, oy in placed:
                dist = math.sqrt((x - ox) ** 2 + (y - oy) ** 2)

                if dist < r * 3:
                    overlap = True
                    break

            if overlap:
                continue

            # Sample average colour from 3x3 area
            rs = gs = bs = count = 0

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):

                    px = max(0, min(width - 1, x + dx))
                    py = max(0, min(height - 1, y + dy))

                    pr, pg, pb = pixels[px, py]

                    rs += pr
                    gs += pg
                    bs += pb
                    count += 1

            rgb = (
                rs // count,
                gs // count,
                bs // count
            )

            # Place marker slightly offset
            angle = random.random() * 2 * math.pi
            distance = r * 2

            mx = x + distance * math.cos(angle)
            my = y + distance * math.sin(angle)

            mx = max(r, min(width - r, mx))
            my = max(r, min(height - r, my))

            placed.append((mx, my))

            # Short leader line
            draw.line(
                [(x, y), (mx, my)],
                fill="white",
                width=5
            )

            # White outer circle
            draw.ellipse(
                (mx-r, my-r, mx+r, my+r),
                fill="white"
            )

            # Colour inner circle
            draw.ellipse(
                (
                    mx-r+3,
                    my-r+3,
                    mx+r-3,
                    my+r-3
                ),
                fill=rgb
            )

            # Sampling point
            draw.ellipse(
                (x-2, y-2, x+2, y+2),
                fill="white"
            )

            break

    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_samples{samples}{ext}"

    result.save(output_path)

    return output_path

def save_pbn(img, image_path, colors=64):

    pbn = img.convert("RGB").quantize(colors=colors,method=1).convert("RGB")

    width, height = pbn.size
    pixels = pbn.load()

    color_data = {}

    # Collect color statistics
    for y in range(height):
        for x in range(width):

            rgb = pixels[x, y]

            if rgb not in color_data:
                color_data[rgb] = {
                    "count": 0,
                    "sum_x": 0,
                    "sum_y": 0
                }

            color_data[rgb]["count"] += 1
            color_data[rgb]["sum_x"] += x
            color_data[rgb]["sum_y"] += y

    colors_sorted = sorted(
        color_data.items(),
        key=lambda item: item[1]["count"],
        reverse=True
    )

    palette_count = len(colors_sorted)

    cell_size = 40
    palette_padding = 20

    cols = math.ceil(math.sqrt(palette_count))
    rows = math.ceil(palette_count / cols)

    legend_width = cols * cell_size + palette_padding * 2
    legend_height = rows * cell_size + palette_padding * 2

    canvas = Image.new(
        "RGB",
        (
            width + legend_width,
            max(height, legend_height)
        ),
        "white"
    )

    canvas.paste(pbn, (0, 0))

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    placed_tags = []

    # Draw tags on image
    for number, (rgb, data) in enumerate(colors_sorted, start=1):

        cx = int(data["sum_x"] / data["count"])
        cy = int(data["sum_y"] / data["count"])

        tag_r = 12

        found = False

        for distance in range(20, 120, 10):

            for angle in range(0, 360, 20):

                tx = int(
                    cx + distance * math.cos(math.radians(angle))
                )

                ty = int(
                    cy + distance * math.sin(math.radians(angle))
                )

                if (
                    tx < tag_r
                    or ty < tag_r
                    or tx > width - tag_r
                    or ty > height - tag_r
                ):
                    continue

                overlap = False

                for ox, oy in placed_tags:

                    d = math.sqrt(
                        (tx - ox) ** 2 +
                        (ty - oy) ** 2
                    )

                    if d < tag_r * 3:
                        overlap = True
                        break

                if overlap:
                    continue

                placed_tags.append((tx, ty))
                found = True
                break

            if found:
                break

        if not found:
            continue

        # Short leader line
        draw.line(
            [(cx, cy), (tx, ty)],
            fill="black",
            width=5
        )

        # Tag circle
        draw.ellipse(
            (
                tx - tag_r,
                ty - tag_r,
                tx + tag_r,
                ty + tag_r
            ),
            fill="white",
            outline="black"
        )

        text = str(number)

        draw.text(
            (
                tx - 5,
                ty - 7
            ),
            text,
            fill="black",
            font=font
        )

    # Draw palette grid
    for idx, (rgb, data) in enumerate(colors_sorted):

        row = idx // cols
        col = idx % cols

        x = width + palette_padding + col * cell_size
        y = palette_padding + row * cell_size

        draw.rectangle(
            (
                x,
                y,
                x + cell_size,
                y + cell_size
            ),
            fill=rgb,
            outline="black"
        )

        draw.text(
            (
                x + 4,
                y + 4
            ),
            str(idx + 1),
            fill="white",
            font=font
        )

    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_pbn{ext}"

    canvas.save(output_path)

    return output_path

def prep_img(img,border_size,target_width,blur_radius):
    img = add_borders(img, white_border=border_size, black_border=border_size)
    new_height = int(img.height * target_width / img.width)
    img = img.resize((target_width, new_height),Image.Resampling.LANCZOS)
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return img

def save_pbn2(    img,    image_path,    colors=56,method=1,  island_threshold_pct=0.01):

    img = img.convert("RGB")
    pbn = img.quantize(        colors=colors,        method=method    ).convert("RGB")

    width, height = pbn.size
    pixels = pbn.load()

    canvas = pbn.copy()
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(            "arial.ttf",            16        )
    except Exception:
        font = ImageFont.load_default()

    image_area = width * height
    min_pixels = image_area * island_threshold_pct

    # --------------------------------------------------
    # Find all islands
    # --------------------------------------------------

    visited = set()

    directions = [        (1, 0),        (-1, 0),        (0, 1),        (0, -1)    ]

    islands = []

    for y in range(height):
        for x in range(width):

            if (x, y) in visited:
                continue

            color = pixels[x, y]

            queue = deque([(x, y)])
            visited.add((x, y))

            component = []

            while queue:

                cx, cy = queue.popleft()
                component.append((cx, cy))

                for dx, dy in directions:

                    nx = cx + dx
                    ny = cy + dy

                    if (
                        nx < 0
                        or ny < 0
                        or nx >= width
                        or ny >= height
                    ):
                        continue

                    if (nx, ny) in visited:
                        continue

                    if pixels[nx, ny] != color:
                        continue

                    visited.add((nx, ny))
                    queue.append((nx, ny))

            size = len(component)

            sx = sum(px for px, py in component)
            sy = sum(py for px, py in component)

            islands.append({
                "color": color,
                "count": size,
                "cx": sx / size,
                "cy": sy / size
            })

    # --------------------------------------------------
    # Number colors
    # --------------------------------------------------

    color_totals = {}

    for island in islands:

        rgb = island["color"]

        color_totals[rgb] = (
            color_totals.get(rgb, 0)
            + island["count"]
        )

    sorted_colors = sorted(
        color_totals.items(),
        key=lambda x: x[1],
        reverse=True
    )

    color_numbers = {}

    for idx, (rgb, _) in enumerate(
        sorted_colors,
        start=1
    ):
        color_numbers[rgb] = idx

    # --------------------------------------------------
    # Largest island per color
    # --------------------------------------------------

    largest_island_per_color = {}

    for island in islands:

        rgb = island["color"]

        if (
            rgb not in largest_island_per_color
            or island["count"]
            > largest_island_per_color[rgb]["count"]
        ):
            largest_island_per_color[rgb] = island

    # --------------------------------------------------
    # Build tags
    # --------------------------------------------------

    tags = []

    # One box per color (always)

    for rgb, island in (
        largest_island_per_color.items()
    ):

        tags.append({
            "number": color_numbers[rgb],
            "color": rgb,
            "cx": island["cx"],
            "cy": island["cy"],
            "primary": True
        })

    # Additional box for large islands

    for island in islands:

        if island["count"] < min_pixels:
            continue

        rgb = island["color"]

        # Skip the largest island because
        # it already got the mandatory color box.

        if (
            island is
            largest_island_per_color[rgb]
        ):
            continue

        tags.append({
            "number": color_numbers[rgb],
            "color": rgb,
            "cx": island["cx"],
            "cy": island["cy"],
            "primary": False
        })

    print(
        f"Quantized colours: "
        f"{len(color_numbers)}"
    )

    print(
        f"Labels drawn: "
        f"{len(tags)}"
    )

    # --------------------------------------------------
    # Label placement
    # --------------------------------------------------

    rect_w = max(
        30,
        int(width * 0.05)
    )

    rect_h = rect_w

    placed = []

    for tag in tags:

        rgb = tag["color"]

        cx = int(tag["cx"])
        cy = int(tag["cy"])

        found = False

        for distance in range(
            40,
            max(width, height),
            10
        ):

            for angle in range(
                0,
                360,
                15
            ):

                tx = int(
                    cx +
                    distance *
                    math.cos(
                        math.radians(angle)
                    )
                )

                ty = int(
                    cy +
                    distance *
                    math.sin(
                        math.radians(angle)
                    )
                )

                tx = max(
                    1,
                    min(
                        tx,
                        width - rect_w - 1
                    )
                )

                ty = max(
                    1,
                    min(
                        ty,
                        height - rect_h - 1
                    )
                )

                overlap = False

                for (
                    ox,
                    oy,
                    ow,
                    oh
                ) in placed:

                    if not (
                        tx + rect_w < ox
                        or tx > ox + ow
                        or ty + rect_h < oy
                        or ty > oy + oh
                    ):
                        overlap = True
                        break

                if overlap:
                    continue

                placed.append(
                    (
                        tx,
                        ty,
                        rect_w,
                        rect_h
                    )
                )

                found = True
                break

            if found:
                break

        if not found:
            continue

        number = str(tag["number"])

        draw.line(
            [
                (cx, cy),
                (
                    tx + rect_w / 2,
                    ty + rect_h / 2
                )
            ],
            fill="white",
            width=5
        )

        draw.rectangle(
            (
                tx,
                ty,
                tx + rect_w,
                ty + rect_h
            ),
            fill=rgb,
            outline="white",
            width=5 if tag["primary"] else 5
        )

        brightness = (
            rgb[0] * 0.299 +
            rgb[1] * 0.587 +
            rgb[2] * 0.114
        )

        text_color = (
            "black"
            if brightness > 140
            else "white"
        )

        bbox = draw.textbbox(
            (0, 0),
            number,
            font=font
        )

        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        draw.text(
            (
                tx + (rect_w - tw) / 2,
                ty + (rect_h - th) / 2
            ),
            number,
            fill=text_color,
            font=font
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    base, ext = os.path.splitext(
        image_path
    )

    num_colours = len(color_numbers)

    output_path = (
        f"{base}_pbn2_req{colors}_act{num_colours}{ext}"
    )
    canvas.save(output_path)

    return output_path

def save_pbn3(
    img,
    image_path,
    colors=56,
    method=1,
    island_threshold_pct=0.01
):
    import os
    import math
    from collections import deque
    from PIL import ImageDraw, ImageFont

    img = img.convert("RGB")

    pbn = img.quantize(
        colors=colors,
        method=method
    ).convert("RGB")

    width, height = pbn.size
    pixels = pbn.load()

    canvas = pbn.copy()
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            16
        )
    except Exception:
        font = ImageFont.load_default()

    image_area = width * height
    min_pixels = image_area * island_threshold_pct

    # --------------------------------------------------
    # Find islands
    # --------------------------------------------------

    visited = set()

    directions = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1)
    ]

    islands = []

    for y in range(height):
        for x in range(width):

            if (x, y) in visited:
                continue

            color = pixels[x, y]

            queue = deque([(x, y)])
            visited.add((x, y))

            component = []

            while queue:

                cx, cy = queue.popleft()
                component.append((cx, cy))

                for dx, dy in directions:

                    nx = cx + dx
                    ny = cy + dy

                    if (
                        nx < 0
                        or ny < 0
                        or nx >= width
                        or ny >= height
                    ):
                        continue

                    if (nx, ny) in visited:
                        continue

                    if pixels[nx, ny] != color:
                        continue

                    visited.add((nx, ny))
                    queue.append((nx, ny))

            size = len(component)

            sx = sum(px for px, py in component)
            sy = sum(py for px, py in component)

            islands.append(
                {
                    "color": color,
                    "count": size,
                    "cx": sx / size,
                    "cy": sy / size
                }
            )

    # --------------------------------------------------
    # Number colours
    # --------------------------------------------------

    color_totals = {}

    for island in islands:

        rgb = island["color"]

        color_totals[rgb] = (
            color_totals.get(rgb, 0)
            + island["count"]
        )

    sorted_colors = sorted(
        color_totals.items(),
        key=lambda x: x[1],
        reverse=True
    )

    color_numbers = {}

    for idx, (rgb, _) in enumerate(
        sorted_colors,
        start=1
    ):
        color_numbers[rgb] = idx

    # --------------------------------------------------
    # Largest island of each colour
    # --------------------------------------------------

    largest_island_per_color = {}

    for island in islands:

        rgb = island["color"]

        if (
            rgb not in largest_island_per_color
            or island["count"]
            > largest_island_per_color[rgb]["count"]
        ):
            largest_island_per_color[rgb] = island

    # --------------------------------------------------
    # Build tags
    # --------------------------------------------------

    tags = []

    for rgb, island in largest_island_per_color.items():

        tags.append(
            {
                "number": color_numbers[rgb],
                "color": rgb,
                "cx": island["cx"],
                "cy": island["cy"],
                "primary": True,
                "size": island["count"]
            }
        )

    for island in islands:

        if island["count"] < min_pixels:
            continue

        rgb = island["color"]

        if island is largest_island_per_color[rgb]:
            continue

        tags.append(
            {
                "number": color_numbers[rgb],
                "color": rgb,
                "cx": island["cx"],
                "cy": island["cy"],
                "primary": False,
                "size": island["count"]
            }
        )

    tags.sort(
        key=lambda t: t["size"],
        reverse=True
    )

    print(
        f"Quantized colours: {len(color_numbers)}"
    )

    print(
        f"Labels drawn: {len(tags)}"
    )

    # --------------------------------------------------
    # Outer frame placement grid (labels outside image)
    # --------------------------------------------------

    num_tags = len(tags)

    if num_tags > 0:

        preferred_square = max(
            16,
            int(min(width, height) * 0.05)
        )

        min_square = 10
        square_size = preferred_square
        gap = max(4, preferred_square // 4)

        chosen_layout = None

        def ring_capacity(cols, rows, lines):
            return (
                (cols + 2 * lines)
                * (rows + 2 * lines)
                - cols * rows
            )

        for test_size in range(preferred_square, min_square - 1, -2):

            test_gap = max(4, test_size // 4)
            cell = test_size + test_gap

            cols = max(1, math.ceil(width / cell))
            rows = max(1, math.ceil(height / cell))

            border_lines = 1

            while ring_capacity(cols, rows, border_lines) < num_tags:
                border_lines += 1

            if border_lines <= 4:
                chosen_layout = (
                    test_size,
                    test_gap,
                    cols,
                    rows,
                    border_lines
                )
                break

            chosen_layout = (
                test_size,
                test_gap,
                cols,
                rows,
                border_lines
            )

        if chosen_layout is not None:
            (
                square_size,
                gap,
                cols,
                rows,
                border_lines
            ) = chosen_layout
        else:
            cols = 1
            rows = 1
            border_lines = num_tags

        cell = square_size + gap
        outer_pad = gap * 2

        frame_x = border_lines * cell + outer_pad
        frame_y = border_lines * cell + outer_pad

        canvas_w = width + frame_x * 2
        canvas_h = height + frame_y * 2

        image_offset_x = frame_x
        image_offset_y = frame_y

        if (
            canvas.width >= canvas_w
            and canvas.height >= canvas_h
        ):
            image_offset_x = (
                canvas.width - width
            ) // 2
            image_offset_y = (
                canvas.height - height
            ) // 2
            canvas.paste(
                pbn,
                (image_offset_x, image_offset_y)
            )
        else:
            operation_canvas = Image.new(
                "RGB",
                (canvas_w, canvas_h),
                (236, 232, 223)
            )
            operation_canvas.paste(
                canvas,
                (image_offset_x, image_offset_y)
            )
            canvas = operation_canvas

        draw = ImageDraw.Draw(canvas)

        total_cols = cols + border_lines * 2
        total_rows = rows + border_lines * 2

        grid_cells = []

        for row in range(total_rows):

            for col in range(total_cols):

                inside_image_grid = (
                    border_lines <= col < border_lines + cols
                    and border_lines <= row < border_lines + rows
                )

                if inside_image_grid:
                    continue

                gx = (
                    outer_pad
                    + (col + 0.5) * cell
                )

                gy = (
                    outer_pad
                    + (row + 0.5) * cell
                )

                grid_cells.append(
                    {
                        "cx": gx,
                        "cy": gy,
                        "used": False
                    }
                )

        for tag in tags:

            source_x = image_offset_x + tag["cx"]
            source_y = image_offset_y + tag["cy"]

            best_cell = None
            best_dist = float("inf")

            for cell_info in grid_cells:

                if cell_info["used"]:
                    continue

                dx = (
                    cell_info["cx"]
                    - source_x
                )

                dy = (
                    cell_info["cy"]
                    - source_y
                )

                dist = dx * dx + dy * dy

                if dist < best_dist:

                    best_dist = dist
                    best_cell = cell_info

            if best_cell is None:
                continue

            best_cell["used"] = True

            tx = int(
                best_cell["cx"]
                - square_size / 2
            )

            ty = int(
                best_cell["cy"]
                - square_size / 2
            )

            rgb = tag["color"]
            number = str(tag["number"])

            line_width = max(
                2,
                square_size // 10
            )

            outline_width = max(
                2,
                square_size // 12
            )

            draw.line(
                [
                    (source_x, source_y),
                    (
                        best_cell["cx"],
                        best_cell["cy"]
                    )
                ],
                fill="white",
                width=line_width
            )

            draw.rectangle(
                (
                    tx,
                    ty,
                    tx + square_size,
                    ty + square_size
                ),
                fill=rgb,
                outline="white",
                width=outline_width
            )

            brightness = (
                rgb[0] * 0.299
                + rgb[1] * 0.587
                + rgb[2] * 0.114
            )

            text_color = (
                "black"
                if brightness > 140
                else "white"
            )

            font_size = max(
                10,
                int(square_size * 0.45)
            )

            try:
                label_font = ImageFont.truetype(
                    "arial.ttf",
                    font_size
                )
            except Exception:
                label_font = font

            bbox = draw.textbbox(
                (0, 0),
                number,
                font=label_font
            )

            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            draw.text(
                (
                    tx + (square_size - tw) / 2,
                    ty + (square_size - th) / 2
                ),
                number,
                fill=text_color,
                font=label_font
            )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    base, ext = os.path.splitext(
        image_path
    )

    output_path = (
        f"{base}_pbn3_req{colors}"
        f"_act{len(color_numbers)}"
        f"{ext}"
    )

    canvas.save(output_path)

    return output_path

def quantize_for_painting(image,image_path,method="dominant",    n_colors=12,    palette=None):
    
    if isinstance(image, str):
        img = np.array(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
        img = np.array(image.convert("RGB"))
    else:
        img = image.copy()

    pixels = img.reshape(-1, 3)

    if method == "dominant":

        kmeans = KMeans(
            n_clusters=n_colors,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(pixels)
        palette = kmeans.cluster_centers_.astype(np.uint8)

        quantized = palette[labels].reshape(img.shape)

    elif method == "value":

        gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        bins = np.linspace(0, 255, n_colors + 1)

        quantized = np.zeros_like(img)
        palette = []

        for i in range(n_colors):
            mask = (gray >= bins[i]) & (gray < bins[i + 1])

            if np.any(mask):
                color = img[mask].mean(axis=0)
            else:
                color = np.array([0, 0, 0])

            quantized[mask] = color
            palette.append(color)

        palette = np.array(palette).astype(np.uint8)

    elif method == "extremes":

        brightness = pixels.mean(axis=1)
        order = np.argsort(brightness)

        idx = np.linspace(
            0,
            len(order) - 1,
            n_colors,
            dtype=int
        )

        palette = pixels[order[idx]]

        distances = cdist(pixels, palette)
        labels = np.argmin(distances, axis=1)

        quantized = palette[labels].reshape(img.shape)

    elif method == "posterize":

        step = max(1, 256 // n_colors)

        quantized = ((img // step) * step).astype(np.uint8)

        palette = np.unique(
            quantized.reshape(-1, 3),
            axis=0
        )

    elif method == "zorn":

        palette = np.array([
            [241, 230, 178],  # Yellow Ochre
            [181, 70, 50],    # Cadmium Red
            [45, 45, 45],     # Ivory Black
            [255, 255, 255]   # Titanium White
        ], dtype=np.uint8)

        distances = cdist(pixels, palette)
        labels = np.argmin(distances, axis=1)

        quantized = palette[labels].reshape(img.shape)

    elif method == "master":

        if palette is None:
            raise ValueError("palette required for method='master'")

        palette = np.asarray(palette, dtype=np.uint8)

        distances = cdist(pixels, palette)
        labels = np.argmin(distances, axis=1)

        quantized = palette[labels].reshape(img.shape)

    else:
        raise ValueError(f"Unknown method: {method}")

     


    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_{method}_{n_colors}{ext}"
    Image.fromarray(        quantized.astype(np.uint8)    ).save(output_path)
    print(f"Saved: {output_path}")

def get_palette(method):
    palettes = {
        "zorn": [
            (241,230,178),
            (181,70,50),
            (45,45,45),
            (255,255,255)
        ],

        "zorn7": [
            (241,230,178),
            (220,180,120),
            (181,70,50),
            (140,70,55),
            (90,70,60),
            (45,45,45),
            (255,255,255)
        ],

        "zorn9": [
            (255,255,255),
            (241,230,178),
            (225,210,160),
            (210,180,120),
            (190,130,90),
            (181,70,50),
            (130,85,70),
            (80,70,65),
            (45,45,45)
        ],

        "zorn_portrait": [
            (255,255,255),
            (245,220,190),
            (230,190,150),
            (210,160,120),
            (190,130,100),
            (181,70,50),
            (120,90,80),
            (45,45,45)
        ],

        "zorn_landscape": [
            (255,255,255),
            (241,230,178),
            (220,200,120),
            (180,160,90),
            (140,110,70),
            (181,70,50),
            (80,70,55),
            (45,45,45)
        ],

        "zorn_mixed": [
            (255,255,255),
            (245,240,220),
            (241,230,178),
            (230,220,160),
            (220,200,130),
            (205,170,100),
            (190,140,80),
            (181,70,50),
            (155,90,70),
            (130,85,75),
            (100,75,70),
            (70,60,55),
            (45,45,45)
        ],

        "oil": [
            (255,255,255),
            (210,180,35),
            (230,50,50),
            (20,60,160),
            (120,60,40)
        ],

        "split_primary": [
            (255,255,255),
            (255,220,0),
            (215,170,0),
            (255,80,80),
            (180,40,40),
            (0,120,255),
            (20,40,140)
        ],

        "bob_ross": [
            (255,255,255),
            (20,20,20),
            (90,40,20),
            (40,80,30),
            (20,40,140),
            (60,70,90),
            (180,40,40),
            (210,180,35)
        ],

        "watercolor": [
            (255,255,255),
            (240,210,50),
            (255,120,60),
            (200,50,80),
            (80,90,180),
            (40,160,100),
            (120,80,60),
            (50,50,50)
        ],

        "earth": [
            (255,255,255),
            (210,180,35),
            (150,75,0),
            (120,60,40),
            (180,60,50),
            (70,90,60)
        ],

        "portrait": [
            (255,255,255),
            (245,210,180),
            (220,170,130),
            (180,120,90),
            (110,70,60),
            (90,50,40),
            (40,40,40)
        ],

        "pop_art": [
            (255,255,0),
            (255,0,0),
            (0,0,255),
            (0,255,0),
            (255,255,255),
            (0,0,0)
        ],

        "cmyk": [
            (255,255,255),
            (0,255,255),
            (255,0,255),
            (255,255,0),
            (0,0,0)
        ]
    }

    if method not in palettes:
        raise ValueError(f"Unknown palette '{method}'")

    return palettes[method]

def draw_square_grid(    img,    image_path,    ndivisions,    line_width=2):
    
    img = img.convert("RGB")
    canvas = img.copy()

    width, height = canvas.size

    # Cell size based on width
    cell_size = width / ndivisions

    draw = ImageDraw.Draw(canvas)

    # ------------------------------------
    # Determine contrasting grid colour
    # ------------------------------------

    pixels = canvas.load()

    sample_step = max(
        1,
        min(width, height) // 100
    )

    brightness_sum = 0
    count = 0

    for y in range(0, height, sample_step):
        for x in range(0, width, sample_step):

            r, g, b = pixels[x, y]

            brightness_sum += (
                0.299 * r +
                0.587 * g +
                0.114 * b
            )

            count += 1

    avg_brightness = (
        brightness_sum /
        max(count, 1)
    )

    # Black on bright images,
    # white on dark images
    grid_color = (
        (255, 255, 255)
        if avg_brightness < 128
        else (0, 0, 0)
    )

    # ------------------------------------
    # Vertical lines
    # ------------------------------------

    draw.line(
        [(0, 0), (0, height - 1)],
        fill=grid_color,
        width=line_width
    )

    for i in range(1, ndivisions):

        x = round(i * cell_size)

        draw.line(
            [(x, 0), (x, height - 1)],
            fill=grid_color,
            width=line_width
        )

    draw.line(
        [
            (width - 1, 0),
            (width - 1, height - 1)
        ],
        fill=grid_color,
        width=line_width
    )

    # ------------------------------------
    # Horizontal lines
    # Start from bottom-left origin
    # ------------------------------------

    draw.line(
        [
            (0, height - 1),
            (width - 1, height - 1)
        ],
        fill=grid_color,
        width=line_width
    )

    j = 1

    while True:

        y = round(
            height - j * cell_size
        )

        if y <= 0:
            break

        draw.line(
            [(0, y), (width - 1, y)],
            fill=grid_color,
            width=line_width
        )

        j += 1

    draw.line(
        [(0, 0), (width - 1, 0)],
        fill=grid_color,
        width=line_width
    )

    # ------------------------------------
    # Save
    # ------------------------------------

    base, ext = os.path.splitext(
        image_path
    )

    output_path = (
        f"{base}_grid_{ndivisions}"
        f"{ext}"
    )

    canvas.save(output_path)

    print(
        f"Grid {ndivisions}x{j} saved:"
    )
    print(output_path)

    return output_path
   
Tk().withdraw()

image_path = filedialog.askopenfilename()

if image_path:
    img = Image.open(image_path)
    
    img = prep_img(img,10,1000,2)
    img = expand_canvas(img, 1200, 1200, color=(120, 120, 120))
    output_path = save_pbn2(        img,        image_path,        colors=96, method=1, island_threshold_pct=0.005    )
    output_path = save_pbn3(        img,        image_path,        colors=96, method=1, island_threshold_pct=0.005    )
    output_path = save_pbn2(        img,        image_path,        colors=48, method=1, island_threshold_pct=0.005    )
    output_path = save_pbn3(        img,        image_path,        colors=48, method=1, island_threshold_pct=0.005    )
    output_path = save_pbn3(        img,        image_path,        colors=48, method=2, island_threshold_pct=0.005    )
        
    # quantize_for_painting(img, image_path, method="dominant",  n_colors=4)
    # quantize_for_painting(img, image_path, method="dominant",  n_colors=8)
    # quantize_for_painting(img, image_path, method="dominant",  n_colors=12)
    # quantize_for_painting(img, image_path, method="dominant",  n_colors=20)
    # quantize_for_painting(img, image_path, method="value",     n_colors=8)
    # quantize_for_painting(img, image_path, method="value",     n_colors=12)
    quantize_for_painting(img, image_path, method="value",     n_colors=20)
    quantize_for_painting(img, image_path, method="value",     n_colors=4)
    quantize_for_painting(img, image_path, method="value",     n_colors=8)
    # quantize_for_painting(img, image_path, method="value",     n_colors=12)
    quantize_for_painting(img, image_path, method="value",     n_colors=20)

    quantize_for_painting(img, image_path, method="extremes",  n_colors=4)
    # quantize_for_painting(img, image_path, method="extremes",  n_colors=8)
    quantize_for_painting(img, image_path, method="extremes",  n_colors=12)
    # quantize_for_painting(img, image_path, method="extremes",  n_colors=20)

    quantize_for_painting(img, image_path, method="posterize", n_colors=4)
    quantize_for_painting(img, image_path, method="posterize", n_colors=8)
    quantize_for_painting(img, image_path, method="posterize", n_colors=12)
    quantize_for_painting(img, image_path, method="posterize", n_colors=20)

    quantize_for_painting(img, image_path, method="zorn",      n_colors=4)
    # quantize_for_painting(img, image_path, method="zorn",      n_colors=8)
    # quantize_for_painting(img, image_path, method="zorn",      n_colors=12)
    # quantize_for_painting(img, image_path, method="zorn",      n_colors=20)
    
    save_sobel_edges(img,image_path)
    save_sobel_edges(img,image_path, threshold=5)
    save_sobel_edges(img,image_path, threshold=15)
    save_sobel_edges(img,image_path, threshold=25)
    save_sobel_edges(img,image_path, threshold=50)

    save_quantize_sobel(img, image_path, colors=6, threshold=5)
    save_quantize_sobel(img, image_path, colors=12,threshold= 5)
    save_quantize_sobel(img, image_path, colors=18, threshold=5)
    
    save_quantize_sobel(img, image_path, colors=6, threshold=15)
    save_quantize_sobel(img, image_path, colors=12,threshold= 15)
    save_quantize_sobel(img, image_path, colors=18,threshold= 15)

    save_quantize_sobel(img, image_path, colors=6, threshold=25)
    save_quantize_sobel(img, image_path, colors=12,threshold= 25)
    save_quantize_sobel(img, image_path, colors=18,threshold= 25)
        


    # draw_color_centroids(img, image_path,samples=151)
    # draw_color_centroids(img, image_path,samples=150)
    # draw_color_centroids(img, image_path,samples=152)
    # draw_square_grid(img,image_path,ndivisions=4,line_width=2)
    # save_quantize(img, image_path,31,1)
    save_quantize(img, image_path,32,2)
    # save_quantize(img, image_path,7,2)
    # save_quantize(img, image_path,8,2)

    save_poster(img, image_path)
    save_edge(img, image_path)
    save_edge_blur(img, image_path)

    save_grayscale(img, image_path)
    print('All Done')
    # print(save_pbn(img,    image_path, colors=128   ))


