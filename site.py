import sys
import fitz # PyMuPDF

if len(sys.argv) != 2:
    print('Usage: python extract_paths_from_pdf.py input.pdf')
    sys.exit(1)

doc = fitz.open(sys.argv[1])
page = doc[0]
coms = page.get_cdrawings(extended=True)

root = None
node = None
node_level = 0

for cmd in coms:
    del cmd['layer'] # Unused

    # Level in hierarchy
    level = cmd['level']
    del cmd['level']

    # Type of entry
    kind = cmd['type']
    del cmd['type']

    if level == 0:
        if root is not None or kind != 'group':
            raise Exception("duplicate root or root is not the group")

    if kind == 'group':
        cur_node = {
            'parent': None,
            'items':  [],
            'kind':   'group',
            'bounds': cmd['rect'],

            'isolated':  cmd['isolated'],
            'knockout':  cmd['knockout'],
            'blendmode': cmd['blendmode'],
            'opacity':   cmd['opacity'],
        }
    elif kind == 'clip':
        cur_node = {
            'parent': None,
            'items':  [],
            'kind':   'clip',
            'bounds': cmd['scissor'],

            'path':   cmd['items'],
        }
    elif kind == 's':
        cur_node = {
            'parent': None,
            'items':  [],
            'kind':   'stroke',
            'bounds': cmd['rect'],

            'path':   cmd['items'],
            'opacity':   cmd['stroke_opacity'],
            'color':     cmd['color'],
            'width':     cmd['width'],
            'lineCap':   cmd['lineCap'],
            'lineJoin':  cmd['lineJoin'],
            # 'closePath': cmd['closePath'],
            'dashes':    cmd['dashes'],
        }
        if cmd['closePath']:
            raise 123
    elif kind == 'f':
        cur_node = {
            'parent': None,
            'items':  [],
            'kind':   'fill',
            'bounds': cmd['rect'],

            'path':   cmd['items'],
            'even_odd': cmd['even_odd'],
            'opacity':  cmd['fill_opacity'],
            'fill':     cmd['fill'],
        }

        del cmd['rect']
        del cmd['items']
    else:
        raise Exception(f"Unknown type: {kind}")

    if level == 0:
        root = cur_node
        node = cur_node
        node_level = 0
        continue

    # Pop all nodes until we reach parent
    while level < node_level + 1:
        node = node['parent']
        node_level -= 1

    # If node is the parent
    if level == node_level + 1:
        cur_node['parent'] = node
        node['items'].append(cur_node)

        node = cur_node
        node_level += 1
    elif level > node_level + 1:
        raise Exceptione("Jumping up > 1 levels")

import math

def print_recursive(node, pad=''):
    print(f"{pad}{node['kind']}, {len(node['items'])}", end='')
    if node['kind'] == 'stroke':
        total_path = 0.0
        for e in node['path']:
            if e[0] == 'l':
                x0, y0, x1, y1 = e[1][0], e[1][1], e[2][0], e[2][1]
                total_path += math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        print(f" {total_path} {node['path']}")
    else:
        print('')
    for item in node['items']:
        print_recursive(item, pad=pad+' ')

# print_recursive(root)

def match_wall_speck(node):
    if node['kind'] != 'clip':
        return False
    if len(node['items']) != 1:
        return False
    node = node['items'][0]
    if node['kind'] != 'stroke':
        return False
    if len(node['path']) != 1:
        return False
    path = node['path'][0]
    if path[0] != 'l':
        return False
    x0, y0, x1, y1 = path[1][0], path[1][1], path[2][0], path[2][1]
    d = math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
    if d > 3:
        return False
    return True

def match_wall_speck_list(node):
    if node['kind'] != 'clip':
        return False
    if len(node['items']) < 4:
        return False
    for sub in node['items']:
        if not match_wall_speck(sub):
            return False
    return True

def match_wall_speck_list_recursive(node):
    res = []
    if match_wall_speck_list(node):
        res.append(node)
    else:
        for sub in node['items']:
            res.extend(match_wall_speck_list_recursive(sub))
    return res

from PIL import Image, ImageDraw

def cubic_bezier(p0, p1, p2, p3, num_points=100):
    """
        "PIL can draw bezier curves, you just implement it yourself!"

        draw.line(cubic_bezier(p0, p1, p2, p3), fill='red', width=3)
    """

    points = []
    for t in range(num_points + 1):
        t /= num_points
        x = (1 - t)**3 * p0[0] + 3 * (1 - t)**2 * t * p1[0] + 3 * (1 - t) * t**2 * p2[0] + t**3 * p3[0]
        y = (1 - t)**3 * p0[1] + 3 * (1 - t)**2 * t * p1[1] + 3 * (1 - t) * t**2 * p2[1] + t**3 * p3[1]
        points.append((x, y))
    return points

def draw_recursive(draw, item, level=0, offset=0, limit=0):
    if item['kind'] == 'stroke':
        path = item['path']
        color, width, opacity = item['color'], item['width'], item['opacity']
        cap, join = item['lineCap'], item['lineJoin']
        dashes = item['dashes']

        width = round(width)
        width = 1

        if dashes != '[] 0':
            print(dashes)
        
        for entry in path:
            if entry[0] == 'l':
                draw.line((entry[1][0], entry[1][1], entry[2][0], entry[2][1]), fill='black', width=width)
            elif entry[0] == 'c':
                draw.line(cubic_bezier(entry[1], entry[2], entry[3], entry[4]), fill='black', width=1)
            else:
                raise Exception(f"Unknown path: {path}")
    elif item['kind'] == 'fill':
        path = item['path']
        even_odd = item['even_odd']
        opacity = item['opacity']
        fill = item['fill']

        for entry in path:
            if entry[0] == 'l':
                draw.line((entry[1][0], entry[1][1], entry[2][0], entry[2][1]), fill='red', width=1)
            elif entry[0] == 're':
                x0, y0, x1, y1 = entry[1]
                draw.line((x0, y0, x1, y0), fill='red', width=1)
                draw.line((x1, y0, x1, y1), fill='red', width=1)
                draw.line((x1, y1, x0, y1), fill='red', width=1)
                draw.line((x0, y1, x0, y0), fill='red', width=1)
            else:
                raise Exception(f"Unknown path entry: {entry}")

    subs = item['items']
    if level == 0:
        # 4000->
        subs = subs[offset:limit]
    for sub in subs:
        draw_recursive(draw, sub, level=level+1, offset=offset, limit=0)

import cv2
import numpy as np
def pil_to_cv2(pil_img):
    open_cv_image = np.array(pil_img)  # Convert PIL to numpy array
    return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

# quantize and build network of just l entries
QUANTIZATION = 0.01
quantized_network = {}
quantized_beziers = {}
def quantize_and_connect_recursive(node):
    if node['kind'] == 'stroke':
        for path in node['path']:
            if path[0] == 'l':
                x0, y0, x1, y1 = path[1][0], path[1][1], path[2][0], path[2][1]
                x0 = round(x0 / QUANTIZATION)
                y0 = round(y0 / QUANTIZATION)
                x1 = round(x1 / QUANTIZATION)
                y1 = round(y1 / QUANTIZATION)
                quantized_network[(x0, y0)] = quantized_network.get((x0, y0), set([])) | set([(x1, y1)])
                quantized_network[(x1, y1)] = quantized_network.get((x1, y1), set([])) | set([(x0, y0)])
            elif path[0] == 'c':
                p1, p2, p3, p4 = path[1], path[2], path[3], path[4]
                p1 = (round(p1[0] / QUANTIZATION), round(p1[1] / QUANTIZATION))
                p4 = (round(p4[0] / QUANTIZATION), round(p4[1] / QUANTIZATION))
                quantized_beziers[p1] = quantized_beziers.get(p1, set([])) | set([p4])
                quantized_beziers[p4] = quantized_beziers.get(p4, set([])) | set([p1])

    for sub in node['items']:
        quantize_and_connect_recursive(sub)
quantize_and_connect_recursive(root)

def segment_length(s):
    x1, y1 = s[0]
    x2, y2 = s[1]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def equal_eps(l, r, eps=0.1):
    if l is None or r is None:
        return False
    return abs(l - r) < eps

def segments_parallel(l1, l2):
    x1, y1 = l1[0]
    x2, y2 = l1[1]
    x3, y3 = l2[0]
    x4, y4 = l2[1]

    lhs, rhs = (y2 - y1) * (x4 - x3), (y4 - y3) * (x2 - x1)

    return equal_eps(lhs, rhs, eps=0.1)

def angle_between_segments(l1, l2):
    # Extract points from segments
    x1, y1 = l1[0]
    x2, y2 = l1[1]
    x3, y3 = l2[0]
    x4, y4 = l2[1]

    # Calculate direction vectors for each segment
    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x4 - x3, y4 - y3

    # Calculate the dot product and magnitudes
    dot_product = dx1 * dx2 + dy1 * dy2
    magnitude1 = math.sqrt(dx1**2 + dy1**2)
    magnitude2 = math.sqrt(dx2**2 + dy2**2)
    if magnitude1 == 0 or magnitude2 == 0:
        return None

    # Calculate the absolute cosine of the angle
    cos_theta = dot_product / (magnitude1 * magnitude2)
    cos_theta = abs(cos_theta)  # Ignore direction by taking the absolute value
    cos_theta = min(cos_theta, 1)

    # Calculate the angle in radians and then convert to degrees
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)

    return angle_deg

# Match all possible window locations
window_matches = []
for a in quantized_network:
    for b in quantized_network[a]:
        for c in quantized_network[a]:
            if c == b:
                continue
            if not equal_eps(angle_between_segments((a, b), (c, a)), 90, eps=1.0):
                continue
            for d in quantized_network[b]:
                if d == a:
                    continue
                if d not in quantized_network[c]:
                    continue
                if not equal_eps(angle_between_segments((a, b), (c, d)), 0, eps=1.0):
                    continue
                if not equal_eps(angle_between_segments((a, c), (b, d)), 0, eps=1.0):
                    continue
                for e in quantized_network[c]:
                    if e == a or e == d:
                        continue
                    if not equal_eps(angle_between_segments((c, e), (a, c)), 0, eps=1.0):
                        continue
                    for f in quantized_network[e]:
                        if f == c:
                            continue
                        if f not in quantized_network[d]:
                            continue
                        if not equal_eps(angle_between_segments((e, f), (c, d)), 0, eps=1.0):
                            continue
                        for g in quantized_network[e]:
                            if g == f or g == c:
                                continue
                            if not equal_eps(angle_between_segments((e, g), (b, f)), 0, eps=1.0):
                                continue
                            for h in quantized_network[f]:
                                if h == d or h == e:
                                    continue
                                if h not in quantized_network[g]:
                                    continue
                                if not equal_eps(angle_between_segments((g, h), (a, b)), 0, eps=1.0):
                                    continue

                                if not equal_eps(segment_length((a, c)), segment_length((f, h)), eps=1.0):
                                    continue

                                # print(a, b, c, d, e, f, g, h)

                                window_matches.append((
                                    (a[0] * QUANTIZATION, a[1] * QUANTIZATION),
                                    (b[0] * QUANTIZATION, b[1] * QUANTIZATION),
                                    (c[0] * QUANTIZATION, c[1] * QUANTIZATION),
                                    (d[0] * QUANTIZATION, d[1] * QUANTIZATION),
                                    (e[0] * QUANTIZATION, e[1] * QUANTIZATION),
                                    (f[0] * QUANTIZATION, f[1] * QUANTIZATION),
                                    (g[0] * QUANTIZATION, g[1] * QUANTIZATION),
                                    (h[0] * QUANTIZATION, h[1] * QUANTIZATION),
                                ))

# Match all possible door locations
# Doors are in the form:
# stroke, 0 34.5863037109375 [('l', (1569.5, 1007.4683837890625), (1569.5, 972.882080078125))]
# clip, 1
#  clip, 1
#   stroke, 0 0.0 [
#       ('c', (1569.5, 972.882080078125), (1588.60498046875, 972.882080078125), (1604.0899658203125, 988.3668212890625), (1604.0899658203125, 1007.4683837890625))]
# - Stroke A -> B
# - Bezier B -> C (we ignore control points)
# - Angle between (B, A) and (A, C) must be 90 deg
# - The door is along (A, C) wall, facing B
doors_matched = []
for a in quantized_network:
    for b in quantized_network[a]:
        if b not in quantized_beziers:
            continue
        for c in quantized_beziers[b]:
            alpha = angle_between_segments((b, a), (a, c))
            if not equal_eps(alpha, 90, eps=3):
                continue
            doors_matched.append((
                (a[0] * QUANTIZATION, a[1] * QUANTIZATION),
                (b[0] * QUANTIZATION, b[1] * QUANTIZATION),
                (c[0] * QUANTIZATION, c[1] * QUANTIZATION),
            ))

from shapely.geometry import Polygon, box

def fill_rectangle_on_grid(grid, stride, rectangle_points):
    print(rectangle_points)
    # Create a polygon for the target rectangle
    rectangle = Polygon(rectangle_points)

    # Get the bounding box of the rectangle to limit our grid checks
    min_x = int(min(x for x, y in rectangle_points))
    max_x = int(max(x for x, y in rectangle_points))
    min_y = int(min(y for x, y in rectangle_points))
    max_y = int(max(y for x, y in rectangle_points))

    # Iterate over the grid cells in the bounding box range
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            # Create the bounding box for the current cell
            cell_box = box(x, y, x + 1, y + 1)

            # Check if the cell's bounding box intersects the rectangle
            if rectangle.intersects(cell_box):
                grid[x + y * stride] = True  # Mark the cell as filled

wall_speck_nodes = match_wall_speck_list_recursive(root)
wall_specks = []
for res in wall_speck_nodes:
    for sub in res['items']:
        for stroke in sub['items']:
            path = stroke['path'][0]
            x0, y0, x1, y1 = path[1][0], path[1][1], path[2][0], path[2][1]
            wall_specks.append(((x1 + x0) // 2, (y1 + y0) // 2))

field_downscale = 10
field_width = math.ceil(root['bounds'][2]) // field_downscale
field_height = math.ceil(root['bounds'][3]) // field_downscale
print(field_width, field_height)
occupation_walls = [False] * field_width * field_height
for speck in wall_specks:
    x, y = int(speck[0] // field_downscale), int(speck[1] // field_downscale)
    occupation_walls[x + y * field_width] = True

occupation_window = [False] * field_width * field_height
for a, b, c, d, e, f, g, h in window_matches:
    pts = [a, b, g, h]
    pts = [(int(p[0] // field_downscale), int(p[1] // field_downscale)) for p in pts]
    # Windows have too many false positives
    # fill_rectangle_on_grid(occupation_walls, field_width, pts)

offset = 4050
limit = min(len(root['items']), 8544)

while True:
    print(f"{offset}-{limit}")

    img = Image.new('RGB', (math.ceil(root['bounds'][2]), math.ceil(root['bounds'][3])), 'white')
    draw = ImageDraw.Draw(img)
    draw_recursive(draw, root, offset=offset, limit=limit)

    for y in range(field_height):
        for x in range(field_width):
            if not occupation_walls[x + y * field_width]:
                continue

            sx = x * field_downscale
            sy = y * field_downscale
            draw.rectangle([(sx, sy), (sx + field_downscale, sy + field_downscale)], fill='blue')

    for a, b, c, d, e, f, g, h in window_matches:
        draw.line((a, b), fill='green', width=3)
        draw.line((b, d), fill='green', width=3)
        draw.line((d, c), fill='green', width=3)
        draw.line((c, a), fill='green', width=3)
        draw.line((c, e), fill='green', width=3)
        draw.line((e, f), fill='green', width=3)
        draw.line((f, d), fill='green', width=3)
        draw.line((e, g), fill='green', width=3)
        draw.line((g, h), fill='green', width=3)
        draw.line((h, f), fill='green', width=3)

    for a, b, c in doors_matched:
        draw.line((a, c), fill='red', width=5)
        draw.line((a, b), fill='orange', width=2)

    img = img.resize((1400, 1000))
    img = pil_to_cv2(img)
    cv2.imshow('', img)
    key = cv2.waitKey(-1)
    key = key & 0xff
    scale = 1

    if key < ord('a'):
        scale = 10
    if key == ord('q') or key == ord('Q'):
        offset -= scale

    elif key == ord('w') or key == ord('W'):
        offset += scale
    elif key == ord('o') or key == ord('O'):
        limit -= scale
    elif key == ord('p') or key == ord('P'):
        limit += scale
    else:
        break
