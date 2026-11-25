"""MAT v5 header scanner and memory-mapped access for large, uncompressed MAT files.

Used for the multi-GB Sentinel-1 files, which scipy.io.loadmat would read fully into memory.
Reads only element headers (decompressing at most a few kB of compressed elements).
"""

from __future__ import annotations

import os
import struct
import zlib

import numpy as np

CLASSES = {1: "cell", 2: "struct", 3: "object", 4: "char", 5: "sparse", 6: "double", 7: "single", 8: "int8",
           9: "uint8", 10: "int16", 11: "uint16", 12: "int32", 13: "uint32", 14: "int64", 15: "uint64"}
MI_DTYPES = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16, 5: np.int32, 6: np.uint32, 7: np.float32,
             9: np.float64, 12: np.int64, 13: np.uint64}


def _sub(b: bytes, off: int):
    t, n = struct.unpack_from("<II", b, off)
    if t >> 16:                                   # small data element
        return t & 0xFFFF, t >> 16, b[off + 4: off + 4 + (t >> 16)], off + 8
    return t, n, b[off + 8: off + 8 + n], off + 8 + ((n + 7) // 8) * 8


def _matrix_header(b: bytes) -> dict:
    _, _, flags, off = _sub(b, 0)
    f = struct.unpack_from("<I", flags)[0]
    _, n, dims, off = _sub(b, off)
    dims = struct.unpack("<%di" % (n // 4), dims[:n])
    _, n, name, off = _sub(b, off)
    t_re, n_re = struct.unpack_from("<II", b, off)
    return {"name": name[:n].decode(errors="replace"), "dims": tuple(dims), "class": CLASSES.get(f & 0xFF, f & 0xFF),
            "complex": bool(f & 0x800), "real_tag": (t_re, n_re), "real_data_offset_in_matrix": off + 8}


def scan(path: str, max_elements: int = 64) -> dict:
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        header = fh.read(128)
        out = {"path": path, "bytes": size, "header_text": header[:116].decode(errors="replace").strip(), "variables": []}
        pos = 128
        while pos + 8 <= size and len(out["variables"]) < max_elements:
            fh.seek(pos)
            t, n = struct.unpack("<II", fh.read(8))
            if t == 15:                            # miCOMPRESSED
                d = zlib.decompressobj()
                raw = b""
                while len(raw) < 1024:
                    chunk = fh.read(65536)
                    if not chunk:
                        break
                    raw += d.decompress(chunk, 4096)
                info = _matrix_header(raw[8:]) if struct.unpack_from("<I", raw)[0] == 14 else {"name": "?"}
                info.update({"storage": "compressed", "element_bytes": n})
            elif t == 14:                          # miMATRIX, uncompressed
                hdr = fh.read(min(n, 1024))
                info = _matrix_header(hdr)
                info.update({"storage": "raw", "element_bytes": n, "element_offset": pos + 8})
            else:
                info = {"storage": f"type {t}", "element_bytes": n}
            info.pop("real_tag", None) if info.get("storage") == "compressed" else None
            out["variables"].append(info)
            pos += 8 + n
            if t != 15:
                pos = (pos + 7) // 8 * 8
    return out


def memmap_complex(path: str, var: dict):
    """(real, imag) read-only memmaps of an uncompressed 2-D complex variable, in MATLAB index order
    (i.e. ``re[i, j] == var(i+1, j+1)``; returned as transposed views of the column-major data)."""
    if var.get("storage") != "raw" or not var.get("complex") or len(var["dims"]) != 2:
        raise ValueError("need an uncompressed 2-D complex variable")
    t_re, n_re = var["real_tag"]
    dt = np.dtype(MI_DTYPES[t_re])
    rows, cols = var["dims"]
    re_off = var["element_offset"] + var["real_data_offset_in_matrix"]
    im_off = re_off + ((n_re + 7) // 8) * 8 + 8
    re = np.memmap(path, dtype=dt, mode="r", offset=re_off, shape=(cols, rows))
    im = np.memmap(path, dtype=dt, mode="r", offset=im_off, shape=(cols, rows))
    return re.T, im.T


def zero_tail_fraction(path: str, block: int = 1 << 24) -> float:
    """Fraction of the file (from the first all-zero block to the end) that is zero: a pre-allocated,
    partially downloaded file has a zero tail."""
    size = os.path.getsize(path)
    m = np.memmap(path, dtype=np.uint8, mode="r")
    first_zero = None
    for s in range(0, size, block):
        if not m[s: s + block].any():
            first_zero = s if first_zero is None else first_zero
        else:
            first_zero = None
    return 0.0 if first_zero is None else (size - first_zero) / size
