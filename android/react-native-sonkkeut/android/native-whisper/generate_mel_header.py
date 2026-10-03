"""Extract the exact OpenAI Whisper 80-bin filterbank without NumPy."""
import ast
import pathlib
import struct
import zipfile

root = pathlib.Path(__file__).resolve().parent
with zipfile.ZipFile(root / "mel_filters.npz") as archive:
    data = archive.read("mel_80.npy")
assert data[:6] == b"\x93NUMPY"
size_bytes = 2 if data[6] == 1 else 4
header_size = int.from_bytes(data[8:8 + size_bytes], "little")
offset = 8 + size_bytes
header = ast.literal_eval(data[offset:offset + header_size].decode("latin1"))
assert header["descr"] == "<f4" and not header["fortran_order"]
assert header["shape"] == (80, 201)
values = struct.unpack("<16080f", data[offset + header_size:])
def literal(value):
    result = format(value, ".9g")
    if "." not in result and "e" not in result:
        result += ".0"
    return result + "f"
native = root / "native"
native.mkdir(exist_ok=True)
with (native / "mel_filters_80.h").open("w", encoding="utf-8", newline="\n") as out:
    out.write("// OpenAI Whisper mel_filters.npz, MIT license; exact float32 coefficients.\n")
    out.write("#pragma once\nstatic constexpr float kWhisperMel80[80][201] = {\n")
    for row in range(80):
        out.write("  {" + ",".join(literal(x) for x in values[row * 201:(row + 1) * 201]) + "},\n")
    out.write("};\n")
print("Generated exact 80 x 201 filterbank")
