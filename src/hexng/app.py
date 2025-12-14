# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 21:26:19 2020
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import IO, ByteString

import numpy as np
from PIL import Image
from cerberus import Validator

from src.common.force_input import force_input
from src.common.json_config import JSONConfig
from src.common.timer import Timer
from src.term_utils import *

APP_PATH = Path(sys.argv[0]).parent
IN_FOLDER = APP_PATH / "in"
OUT_FOLDER = APP_PATH / "out"
CFG_PATH = APP_PATH / "config.json"

F_INFO = Format(fg=FG.BLUE)
F_ERROR = Format(fg=FG.YEL, bg=BG.RED, style=STYLE.BOLD)
F_OK = Format(fg=FG.GREEN, style=STYLE.ITALIC)
F_INVERTED = Format(style=STYLE.REVERSE)
C_GRAY = FGRGB(64, 64, 64)
F_MISC = Format(fg=C_GRAY)

BYTE_FILLER = b"\x00"
RGB_BYTE_SIZE = 3

COLUMNS_RANGE = range(1, 8192 + 1)

CFG_DEFAULT = JSONConfig(
    {
        "width": 1024
    }, True
)

CFG_VALIDATOR = Validator(
    {
        "width": {
            "type": "integer",
            "min": COLUMNS_RANGE[0],
            "max": COLUMNS_RANGE[-1],
        },
    }, require_all=True
)


@dataclass(frozen=True)
class Parameters:
    width: int
    height: int
    bytes_per_pixel: int
    bytes_per_row: int
    padding: int
    file_size: int

    @classmethod
    def from_file(cls, bin_path: Path, width: int, bytes_per_pixel: int = 3):
        bytes_per_row = width * bytes_per_pixel
        file_size = bin_path.stat().st_size
        padding = -file_size % bytes_per_row
        height = (file_size + padding) // bytes_per_row

        return cls(width, height, bytes_per_pixel, bytes_per_row, padding, file_size)


@dataclass(frozen=True)
class ImageRow:
    idx: int
    image_data: Image


def b2image(bin_file: IO, params: Parameters, filler: ByteString):
    row_idx = 0
    while buffer := bin_file.read(params.bytes_per_row):
        if len(buffer) < params.bytes_per_row:
            buffer += filler * (params.bytes_per_row - len(buffer))

        row_pixels = np.frombuffer(buffer, dtype=np.uint8).reshape(
            (1, params.width, params.bytes_per_pixel))

        yield ImageRow(row_idx, row_pixels)

        row_idx += 1


class App:
    def main(self):
        IN_FOLDER.mkdir(exist_ok=True)
        OUT_FOLDER.mkdir(exist_ok=True)

        config = self.ensure_config()

        if not config:
            writef(f":: No config ::", F_ERROR)
            return

        input_files = set((APP_PATH / "in").iterdir())

        if len(input_files) == 0:
            writef(":: No input files ::", F_ERROR)
            write()
            return

        writef("< FILES >", F_INVERTED)
        write()
        [write(f"\t{ffg('>', C_GRAY)} {file.name}\n") for file in input_files]

        write()

        write("Width: ")
        write(ffg(config.width, FG.YEL))
        write()

        total_timer = Timer().tic()

        for bin_path in input_files:
            params = Parameters.from_file(
                bin_path, config.width, RGB_BYTE_SIZE
            )
            png_path = OUT_FOLDER / f"{bin_path.name}.png"
            current_timer = Timer().tic()

            image = Image.new("RGB", (params.width, params.height))

            write(
                f"{fstyle(':: File:', STYLE.BOLD)} {ffg(bin_path.name, FG.MAGNT)}\n"
            )
            write(
                f"\tsize: {ffg(params.width, FG.BLUE)}x{ffg(params.height, FG.BLUE)}\n"
            )
            write(
                f"\t{ffg(f'{params.file_size} Bytes ( +{params.padding} Bytes padded)', C_GRAY)}\n"
            )

            write(ffg("( Processing )\n", C_GRAY))

            try:
                with open(bin_path, 'rb') as bin_file:
                    for row in b2image(bin_file, params, BYTE_FILLER):
                        image.paste(Image.fromarray(
                            row.image_data, "RGB"), (0, row.idx)
                        )

            except OSError as e:
                writef(f":: OS error: {e} ::", F_ERROR)
                write()
                write(f"{ffg('>>> SKIP', FG.RED)}\n")
                continue

            Cur.prev_line()
            Scr.clear_line()

            write(ffg("( Saving )\n", C_GRAY))

            image.save(png_path)

            Cur.prev_line()
            Scr.clear_line()

            writef(f"\tSaved as {png_path}\n", F_MISC)
            writef(f"\tIn {current_timer.toc():.2f} sec\n", F_MISC)
            write()

        writef("( Done )\n", F_OK)
        writef(f"Elapsed: {total_timer.toc():.2f} sec\n", F_MISC)

    def ensure_config(self):
        try:
            config = JSONConfig.load(CFG_PATH, True)

        except FileNotFoundError:
            CFG_DEFAULT.save(CFG_PATH)
            config = JSONConfig(CFG_DEFAULT.as_dict(), True)
            writef("( Config file created )", F_MISC)
            write()

        except OSError:
            writef(":: Unknown OS error ::", F_ERROR)
            write()
            return

        write()

        if not CFG_VALIDATOR.validate(config.as_dict()):
            write("[X] Config validation failed:\n")

            for entry in self.flatten_errors(CFG_VALIDATOR.errors):
                err_str = f"\t{ffg('>', FG.RED)} {entry[0]:<30} : {ffg((entry[1]), FG.RED)}\n"
                write(err_str)

            write()

            write(f"Reset {CFG_PATH}?\n")
            write(f"{ffg('( [y/n] )', C_GRAY)}\n")
            yn = force_input(
                ffg(">>> ", FG.GREEN), ffg("( [y/n] )", C_GRAY),
                func=str.lower,
                predicates=[lambda x: x in ('y', 'n')]
            )
            if yn == 'y':
                CFG_DEFAULT.save(CFG_PATH)

            return

        return config

    def flatten_errors(self, data, path=None, result=None):
        if path is None:
            path = []
        if result is None:
            result = []

        if isinstance(data, dict):
            for key, value in data.items():
                self.flatten_errors(value, path + [key], result)
        elif isinstance(data, list):
            for item in data:
                self.flatten_errors(item, path, result)
        else:
            full_path = ":".join(path)
            result.append((full_path, data))

        return result

    def run(self):
        Scr.color_on()
        try:
            self.main()
        except KeyboardInterrupt:
            input("Press Enter to exit...")
