"""
Patch the installed TensorFlowLite_ESP32 library for the current ESP32 core.

The upstream FlatBuffers span implementation bundled with TensorFlowLite_ESP32
1.0.0 does not compile cleanly with the currently installed esp32 core because
`count_` is marked const while the copy-assignment operator writes to it.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_HEADER = (
    Path.home()
    / "Documents/Arduino/libraries/TensorFlowLite_ESP32/src/third_party/flatbuffers/stl_emulation.h"
)

CONST_NEEDLE = "  const size_type count_;\n"
CONST_REPLACEMENT = "  size_type count_;\n"

ASSIGN_NEEDLE = """  FLATBUFFERS_CONSTEXPR_CPP14 span &operator=(const span &other)
      FLATBUFFERS_NOEXCEPT {
    data_ = other.data_;
    count_ = other.count_;
  }
"""

ASSIGN_REPLACEMENT = """  FLATBUFFERS_CONSTEXPR_CPP14 span &operator=(const span &other)
      FLATBUFFERS_NOEXCEPT {
    data_ = other.data_;
    count_ = other.count_;
    return *this;
  }
"""


def patch_header(header_path: Path) -> bool:
    if not header_path.exists():
        raise SystemExit(f"TensorFlowLite_ESP32 header not found: {header_path}")

    original = header_path.read_text()
    updated = original

    if CONST_NEEDLE in updated:
        updated = updated.replace(CONST_NEEDLE, CONST_REPLACEMENT, 1)
    if ASSIGN_NEEDLE in updated:
        updated = updated.replace(ASSIGN_NEEDLE, ASSIGN_REPLACEMENT, 1)

    if updated == original:
        print(f"No patch needed: {header_path}")
        return False

    header_path.write_text(updated)
    print(f"Patched TensorFlowLite_ESP32 compatibility issue: {header_path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch the installed TensorFlowLite_ESP32 FlatBuffers span header.")
    parser.add_argument("--header", default=str(DEFAULT_HEADER), help="Path to stl_emulation.h")
    args = parser.parse_args()

    patched = patch_header(Path(args.header).expanduser())
    return 0 if patched or Path(args.header).expanduser().exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())

