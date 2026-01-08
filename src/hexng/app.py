# -*- coding: utf-8 -*-

"""
Created on Tue Dec  8 21:26:19 2020
"""

import json
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.live import Live

from src.common.timer import Timer

HEADER_SEPARATORS = 3

APP_PATH = Path(sys.argv[0]).parent
IN_FOLDER = APP_PATH / "in"
OUT_FOLDER = APP_PATH / "out"
CONFIG_FILE = APP_PATH / "config.json"

console = Console(highlight=False)

BYTE_FILLER = b"\x00"
RGB_BYTE_SIZE = 3


class Config(BaseModel):
    width: int = Field(default=1024, ge=2, le=8192)

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
    bytes_per_row: int
    padding: int
    file_size: int

    @classmethod
    def from_file(
            cls, bin_path: Path, width: int
    ):
        bytes_per_row = width * RGB_BYTE_SIZE
        file_size = bin_path.stat().st_size
        padding = -file_size % bytes_per_row
        height = (file_size + padding) // bytes_per_row

        return cls(width, height, bytes_per_row, padding, file_size)


@dataclass(frozen=True)
class ImageRow:
    idx: int
    image_data: np.ndarray


class BinaryImageHeader:
    NAME = b"HEXNG"
    VERSION = bytes([0x01, 0x00, 0x00, 0x00])
    SEP = b":."
    MAGIC = bytes([0x35, 0x42, 0x99, 0x87])
    ZERO = b"Z"

    @staticmethod
    def number_to_bytes(num: int) -> bytes:
        return num.to_bytes((num.bit_length() + 7) // 8, byteorder="big")

    @staticmethod
    def prime_generator():
        primes = []
        candidate = 2
        while True:
            is_prime = True
            for p in primes:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
                yield candidate
            candidate += 1

    @staticmethod
    def create_separator_42(width: int) -> bytes:
        bytes_needed = width * RGB_BYTE_SIZE
        marker = BinaryImageHeader.MAGIC
        repeat = (bytes_needed + len(marker) - 1) // len(marker)
        return (marker * repeat)[:bytes_needed]

    @staticmethod
    def create_separator_prime(width: int) -> bytes:
        bytes_needed = width * RGB_BYTE_SIZE
        separator = bytearray()

        for prime in BinaryImageHeader.prime_generator():
            separator.extend(BinaryImageHeader.number_to_bytes(prime))
            if len(separator) >= bytes_needed:
                break

        return bytes(separator[:bytes_needed])

    @staticmethod
    def create_metadata(
            original_size: int,
            pad_size: int,
            original_name: str,
    ) -> bytes:
        timestamp = int(time.time())

        name_bytes = original_name.encode("utf-8")[:256]
        name_bytes = name_bytes.ljust(256, BYTE_FILLER)

        metadata = bytearray()
        metadata.extend(BinaryImageHeader.NAME)
        metadata.extend(BinaryImageHeader.VERSION)
        metadata.extend(BinaryImageHeader.SEP)
        metadata.extend(name_bytes)
        metadata.extend(BinaryImageHeader.SEP)
        metadata.extend(struct.pack("<Q", timestamp))
        metadata.extend(BinaryImageHeader.SEP)
        metadata.extend(struct.pack("<Q", original_size))
        metadata.extend(BinaryImageHeader.SEP)
        metadata.extend(struct.pack("<Q", pad_size))
        metadata.extend(BinaryImageHeader.ZERO * 5)

        return bytes(metadata)

    @staticmethod
    def split_into_rows(
            data: bytes,
            width: int,
    ) -> list[bytes]:
        row_size = width * RGB_BYTE_SIZE
        rows = []

        for i in range(0, len(data), row_size):
            chunk = data[i:i + row_size]
            if len(chunk) < row_size:
                chunk += BYTE_FILLER * (row_size - len(chunk))
            rows.append(chunk)

        return rows


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
                    (1, params.width, RGB_BYTE_SIZE)
                )

                yield ImageRow(row_idx, row_pixels)
                row_idx += 1


class ImageConverter:
    def __init__(self, bin_path: Path, params: Parameters):
        self.bin_path = bin_path
        self.params = params
        self.image = None
        self.row_offset = 0

    def header_height(self) -> int:
        metadata_bytes = BinaryImageHeader.create_metadata(
            self.params.file_size,
            self.params.padding,
            self.bin_path.name
        )
        metadata_rows = BinaryImageHeader.split_into_rows(
            metadata_bytes,
            self.params.width
        )
        return HEADER_SEPARATORS + len(metadata_rows)

    def create_header_row(self, row_bytes: bytes) -> None:
        pixels = np.frombuffer(row_bytes, dtype=np.uint8).reshape(
            (1, self.params.width, RGB_BYTE_SIZE)
        )
        self.image.paste(
            Image.fromarray(pixels, "RGB"),
            (0, self.row_offset)
        )
        self.row_offset += 1

    def get_row_offset(self):
        return self.row_offset

    def create_header(self) -> None:
        sep42_bytes = BinaryImageHeader.create_separator_42(self.params.width)
        self.create_header_row(sep42_bytes)

        metadata_bytes = BinaryImageHeader.create_metadata(
            self.params.file_size,
            self.params.padding,
            self.bin_path.name
        )
        metadata_rows = BinaryImageHeader.split_into_rows(
            metadata_bytes,
            self.params.width
        )

        for meta_row in metadata_rows:
            self.create_header_row(meta_row)

        sep_primes_bytes = BinaryImageHeader.create_separator_prime(self.params.width)
        self.create_header_row(sep_primes_bytes)
        self.create_header_row(sep42_bytes)

    def insert_data_rows(self) -> None:
        for row in BinaryReader.read_rows(self.bin_path, self.params):
            self.image.paste(
                Image.fromarray(row.image_data, "RGB"),
                (0, row.idx + self.row_offset)
            )

    def convert(self) -> Image.Image:
        header_rows = self.header_height()
        total_height = self.params.height + header_rows

        self.image = Image.new("RGB", (self.params.width, total_height))
        self.row_offset = 0

        try:
            self.create_header()
            self.insert_data_rows()
            return self.image
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
            console.print(f"\t[green]>[/green] {file.name}")
        console.print()

    def print_config_summary(self) -> None:
        console.print(f"Width: [yellow]{self.config.width}[/yellow]\n")

    def process_file(self, bin_path: Path) -> bool:
        console.print(f"[bold]:: File:[/bold] [magenta]{bin_path.name}[/magenta]")

        params = Parameters.from_file(bin_path, self.config.width)
        png_path = OUT_FOLDER / f"{bin_path.name}.png"

        timer = Timer().tic()

        console.print(
            f"\tSize: [blue]{params.width}[/blue]×[blue]{params.height}[/blue]"
        )
        console.print(
            f"\t[grey27]{params.file_size} Bytes ( +{params.padding} Bytes padded )[/grey27]"
        )

        try:
            with Live("[grey27]( Processing )[/grey27]", transient=True):
                converter = ImageConverter(bin_path, params)
                image = converter.convert()

            console.print(f"[grey54]- Header: {converter.get_row_offset()} rows -[/grey54]")

            with Live("[grey27]( Saving )[/grey27]", transient=True):
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
