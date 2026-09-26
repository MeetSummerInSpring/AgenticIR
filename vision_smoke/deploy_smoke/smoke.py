#!/usr/bin/env python3
"""Small, dependency-free PNG processing smoke test."""

import argparse
import binascii
import os
import platform
import struct
import sys
import zlib
from pathlib import Path


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(kind, payload):
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", binascii.crc32(body) & 0xFFFFFFFF)


def _paeth(left, above, upper_left):
    estimate = left + above - upper_left
    left_distance = abs(estimate - left)
    above_distance = abs(estimate - above)
    upper_left_distance = abs(estimate - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    if above_distance <= upper_left_distance:
        return above
    return upper_left


def read_png(path):
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("input is not a PNG file")

    offset = len(PNG_SIGNATURE)
    header = None
    compressed = bytearray()
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError("truncated PNG chunk")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload_start = offset + 8
        payload_end = payload_start + length
        crc_end = payload_end + 4
        if crc_end > len(data):
            raise ValueError("truncated PNG payload")
        payload = data[payload_start:payload_end]
        expected_crc = struct.unpack(">I", data[payload_end:crc_end])[0]
        if binascii.crc32(kind + payload) & 0xFFFFFFFF != expected_crc:
            raise ValueError("PNG chunk CRC mismatch")
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", payload)
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break
        offset = crc_end

    if header is None:
        raise ValueError("PNG is missing IHDR")
    width, height, bit_depth, color_type, compression, filtering, interlace = header
    if bit_depth != 8 or color_type not in (0, 2, 4, 6):
        raise ValueError("only 8-bit grayscale, RGB, grayscale-alpha, and RGBA PNGs are supported")
    if compression != 0 or filtering != 0 or interlace != 0:
        raise ValueError("only non-interlaced PNGs with standard compression/filtering are supported")

    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    stride = width * channels
    raw = zlib.decompress(bytes(compressed))
    expected_length = height * (stride + 1)
    if len(raw) != expected_length:
        raise ValueError("unexpected decompressed PNG length")

    rows = []
    previous = bytearray(stride)
    cursor = 0
    for _ in range(height):
        filter_type = raw[cursor]
        cursor += 1
        filtered = raw[cursor : cursor + stride]
        cursor += stride
        row = bytearray(stride)
        for index, value in enumerate(filtered):
            left = row[index - channels] if index >= channels else 0
            above = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            elif filter_type == 4:
                predictor = _paeth(left, above, upper_left)
            else:
                raise ValueError("unsupported PNG filter type: %d" % filter_type)
            row[index] = (value + predictor) & 0xFF
        rows.append(row)
        previous = row
    return width, height, color_type, channels, rows


def write_png(path, width, height, color_type, rows):
    raw = b"".join(b"\x00" + bytes(row) for row in rows)
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    encoded = PNG_SIGNATURE + _chunk(b"IHDR", header) + _chunk(b"IDAT", zlib.compress(raw, 9)) + _chunk(b"IEND", b"")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(str(temporary), str(path))


def apply_gamma(rows, channels, gamma=0.72):
    lookup = [round(255.0 * ((value / 255.0) ** gamma)) for value in range(256)]
    color_channels = 1 if channels in (1, 2) else 3
    output = []
    for source in rows:
        row = bytearray(source)
        for index in range(0, len(row), channels):
            for channel in range(color_channels):
                row[index + channel] = lookup[row[index + channel]]
        output.append(row)
    return output


def make_sample(path, width=256, height=256):
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            red = x
            green = y
            blue = (x + y) // 2
            if 64 <= x < 192 and 64 <= y < 192:
                red, green, blue = 245, 110, 40
            row.extend((red, green, blue))
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(path, width, height, 2, rows)


def process(input_path, output_path):
    if not input_path.is_file():
        print("error=input file not found: %s" % input_path, file=sys.stderr)
        return 2
    try:
        width, height, color_type, channels, rows = read_png(input_path)
        output_rows = apply_gamma(rows, channels)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_png(output_path, width, height, color_type, output_rows)
    except Exception as exc:
        print("error=%s" % exc, file=sys.stderr)
        return 3

    print("arch=%s" % platform.machine())
    print("python=%s" % platform.python_version())
    print("input_size=%dx%d" % (width, height))
    print("output_size=%dx%d" % (width, height))
    print("status=container test passed")
    return 0


def main():
    parser = argparse.ArgumentParser(description="CPU-only PNG smoke processor")
    parser.add_argument("--input", type=Path, default=Path("/data/input.png"))
    parser.add_argument("--output", type=Path, default=Path("/data/output.png"))
    parser.add_argument("--generate-sample", type=Path, metavar="PATH")
    args = parser.parse_args()
    if args.generate_sample is not None:
        make_sample(args.generate_sample)
        print("sample=%s" % args.generate_sample)
        return 0
    return process(args.input, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
