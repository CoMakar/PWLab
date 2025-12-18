# -*- coding: utf-8 -*-

"""
Created on Tue Dec  8 21:26:19 2020
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console

from src.common.timer import Timer

APP_PATH = Path(sys.argv[0]).parent
IN_FOLDER = APP_PATH / "in"
OUT_FOLDER = APP_PATH / "out"
CONFIG_FILE = APP_PATH / "config.json"

console = Console(highlight=False)

BYTE_FILLER = b"\x00"
RGB_BYTE_SIZE = 3


class Config(BaseModel):
    width: int = Field(default=1024, ge=1, le=8192)

    @classmethod
    def from_json(cls, path: Path):
        if not path.exists():
            instance = cls()
            with open(path, "w", encoding="utf-8") as f:
                f.write(instance.model_dump_json(indent=4))
            console.print("[grey27]( Config file created )[/grey27]\n")
            return instance
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(**data)
        except ValidationError as e:
            console.print(
                "[yellow on red bold]:: Invalid config.json ::[/yellow on red bold]\n"
            )

            for error in e.errors():
                field = ".".join(str(loc) for loc in error["loc"])
                msg = error["msg"]
                console.print(f"[red]•[/red] [cyan]{field}[/cyan]: {msg}")
            console.print()
            sys.exit(1)
        except json.JSONDecodeError as e:
            console.print(
                "[yellow on red bold]:: Invalid JSON in config.json ::[/yellow on red bold]"
            )
            console.print(f"[red]{e}[/red]\n")
            sys.exit(1)


@dataclass(frozen=True)
class Parameters:
    width: int
    height: int
    bytes_per_pixel: int
    bytes_per_row: int
    padding: int
    file_size: int

    @classmethod
    def from_file(
            cls, bin_path: Path, width: int, bytes_per_pixel: int = RGB_BYTE_SIZE
    ):
        bytes_per_row = width * bytes_per_pixel
        file_size = bin_path.stat().st_size
        padding = -file_size % bytes_per_row
        height = (file_size + padding) // bytes_per_row

        return cls(width, height, bytes_per_pixel, bytes_per_row, padding, file_size)


@dataclass(frozen=True)
class ImageRow:
    idx: int
    image_data: np.ndarray


class BinaryReader:
    @staticmethod
    def read_rows(
            bin_path: Path, params: Parameters, filler: bytes = BYTE_FILLER
    ) -> Iterator[ImageRow]:
        with open(bin_path, "rb") as bin_file:
            row_idx = 0

            while buffer := bin_file.read(params.bytes_per_row):
                if len(buffer) < params.bytes_per_row:
                    buffer += filler * (params.bytes_per_row - len(buffer))
                row_pixels = np.frombuffer(buffer, dtype=np.uint8).reshape(
                    (1, params.width, params.bytes_per_pixel)
                )

                yield ImageRow(row_idx, row_pixels)
                row_idx += 1


class ImageConverter:

    @staticmethod
    def convert(bin_path: Path, params: Parameters) -> Image.Image:
        image = Image.new("RGB", (params.width, params.height))

        try:
            for row in BinaryReader.read_rows(bin_path, params):
                image.paste(Image.fromarray(row.image_data, "RGB"), (0, row.idx))
            return image
        except OSError as e:
            raise RuntimeError(f"Failed to read binary file: {e}")


class App:
    def __init__(self):
        self.config = Config.from_json(CONFIG_FILE)

    def find_input_files(self) -> set[Path]:
        if not IN_FOLDER.exists():
            return set()
        return set(IN_FOLDER.iterdir())

    def print_file_list(self, files: set[Path]) -> None:
        console.print("\n[reverse bold]< FILES >[/reverse bold]")

        for file in files:
            console.print(f"\t[grey27]>[/grey27] {file.name}")
        console.print()

    def print_config_summary(self) -> None:
        console.print(f"Width: [yellow]{self.config.width}[/yellow]\n")

    def process_file(self, bin_path: Path) -> bool:
        console.print(f"[bold]:: File:[/bold] [magenta]{bin_path.name}[/magenta]")

        params = Parameters.from_file(bin_path, self.config.width, RGB_BYTE_SIZE)
        png_path = OUT_FOLDER / f"{bin_path.name}.png"

        timer = Timer().tic()

        console.print(
            f"\tSize: [blue]{params.width}[/blue]×[blue]{params.height}[/blue]"
        )
        console.print(
            f"\t[grey27]{params.file_size} Bytes ( +{params.padding} Bytes padded )[/grey27]"
        )
        console.print("[grey27]( Processing )[/grey27]")

        try:
            image = ImageConverter.convert(bin_path, params)

            console.print("[grey27]( Saving )[/grey27]")
            image.save(png_path)

            elapsed = timer.toc()

            console.print(f"[green italic] ✓ Saved as {png_path.name}[/green italic]")
            console.print(f"[grey27]In {elapsed:.2f} sec[/grey27]")
            console.print("")

            return True
        except RuntimeError as e:
            console.print(f"[yellow on red bold]:: {e} ::[/yellow on red bold]\n")
            console.print("[red]>>> SKIP[/red]\n")
            return False
        except OSError as e:
            console.print(f"[yellow on red bold]:: OS error: {e} ::[/yellow on red bold]\n")
            console.print("[red]>>> SKIP[/red]\n")
            return False

    def main(self) -> None:
        IN_FOLDER.mkdir(exist_ok=True)
        OUT_FOLDER.mkdir(exist_ok=True)

        input_files = self.find_input_files()

        if not input_files:
            console.print(
                "[yellow on red bold]:: No input files ::[/yellow on red bold]\n"
            )
            return

        self.print_file_list(input_files)
        self.print_config_summary()

        total_timer = Timer().tic()
        success_count = 0

        for bin_path in input_files:
            if self.process_file(bin_path):
                success_count += 1

        console.print("[green italic]( Done )[/green italic]")
        console.print(f"[grey27]Processed: {success_count}/{len(input_files)}[/grey27]")
        console.print(f"[grey27]Elapsed: {total_timer.toc():.2f} sec[/grey27]")

    def run(self) -> None:
        try:
            self.main()
        except KeyboardInterrupt:
            pass
