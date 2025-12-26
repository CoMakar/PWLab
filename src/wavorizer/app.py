# -*- coding: utf-8 -*-

"""
Created on Tue Jun 23 21:07:56 2020
"""

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum, IntEnum
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from PIL.Image import Resampling
from pydantic import BaseModel, Field, field_validator, ValidationError
from rich.console import Console
from scipy.io.wavfile import write as wav_write
from scipy.ndimage import sobel
from scipy.signal import ShortTimeFFT

from src.common.md5_file_hash import calculate_md5
from src.common.timer import Timer

APP_PATH = Path(sys.argv[0]).parent
IN_FOLDER = APP_PATH / "in"
OUT_FOLDER = APP_PATH / "out"
CONFIG_FILE = APP_PATH / "config.json"

console = Console(highlight=False)

ISM_SIZE_THRESHOLD = 64
ALLOWED_IMG_FORMATS = (".png", ".jpg", ".jpeg")
MAX_IMAGE_LIN_SIZE = 1e10
REMAPPER_POWER = 0.5
REMAPPER_EXP_COEF = 1


class Mode(StrEnum):
    ISM = "ISM"
    PBP = "PBP"


class SampleRateMode(StrEnum):
    STATIC = "static"
    DYNAMIC = "dynamic"


class ScanMode(StrEnum):
    ROWS = "rows"
    COLS = "cols"
    ZIGZAG = "zigzag"
    SPIRAL = "spiral"


class PostFilter(StrEnum):
    NONE = "none"
    SUM = "sum"
    MULT = "mult"
    XOR = "xor"


class Resolution(IntEnum):
    LOW = 640 * 320
    MID = 7680 * 4320
    HIGH = 15360 * 8640


class SampleRateRange(IntEnum):
    MIN = 512
    LOW = 4096
    HIGH = 65536
    MAX = 262144


class PBPConfig(BaseModel):
    post_filter: PostFilter = PostFilter.NONE
    quantization_level: int = Field(default=0, ge=0, le=16)
    scan_mode: ScanMode = ScanMode.ROWS


class ISMConfig(BaseModel):
    detect_edges: bool = False
    blur_radius: int = Field(default=0, ge=0, le=16)
    noise_strength: float = Field(default=0.0, ge=0.0, le=1.0)


class Config(BaseModel):
    mode: Mode = Mode.PBP
    channels: int = Field(default=1, ge=1, le=2)
    sample_rate_mode: SampleRateMode = SampleRateMode.STATIC
    sample_rate: int = Field(default=44100, ge=512, le=262144)
    image_scale: int = Field(default=1, ge=1, le=10)
    PBP: PBPConfig = Field(default_factory=PBPConfig)
    ISM: ISMConfig = Field(default_factory=ISMConfig)

    @field_validator("channels")
    def validate_channels(cls, v):
        if v not in [1, 2]:
            raise ValueError("Channels must be 1 or 2")
        return v

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
    resize: bool
    scale: int
    width: int
    height: int
    linear_size: int
    sample_rate: int

    @classmethod
    def from_image(cls, image: Image, config: Config):
        resize = config.image_scale != 1
        scale = config.image_scale
        size = (round(image.size[0] * scale), round(image.size[1] * scale))
        width, height = size
        linear_size = width * height

        sample_rate = (
            config.sample_rate
            if config.sample_rate_mode == SampleRateMode.STATIC
            else round(
                remap_sample_rate_from_size(
                    linear_size,
                    SampleRateRange.MIN,
                    SampleRateRange.LOW,
                    SampleRateRange.HIGH,
                    SampleRateRange.MAX,
                    Resolution.LOW,
                    Resolution.MID,
                    Resolution.HIGH,
                    REMAPPER_POWER,
                    REMAPPER_EXP_COEF,
                )
            )
        )

        return cls(resize, scale, width, height, linear_size, sample_rate)


def remap_sample_rate_from_size(
        linear_size: int,
        min_sr: int,
        low_sr: int,
        high_sr: int,
        max_sr: int,
        low_res: int,
        mid_res: int,
        high_res: int,
        power: float,
        exp_multiplier: float,
) -> int:
    if linear_size <= low_res:
        sample_rate = min_sr + ((linear_size / low_res) ** power) * (low_sr - min_sr)
    elif linear_size <= mid_res:
        normalized_area = (linear_size - low_res) / (mid_res - low_res)
        sample_rate = low_sr + normalized_area * (high_sr - low_sr)
    else:
        normalized_area = (linear_size - mid_res) / (high_res - mid_res)
        sample_rate = high_sr + (max_sr - high_sr) * (
                1 - math.exp(-normalized_area * exp_multiplier)
        )
    return round(sample_rate)


def zigzag_scan(image_data: np.ndarray) -> np.ndarray:
    height, width = image_data.shape[:2]
    result = np.zeros_like(image_data)

    for y in range(height):
        if y % 2 == 0:
            result[y] = image_data[y]
        else:
            result[y] = image_data[y, ::-1]
    return result


def spiral_scan(image_data: np.ndarray) -> np.ndarray:
    h, w = image_data.shape[:2]
    result = np.zeros([h * w, 3], dtype=image_data.dtype)

    x, y = 0, 0
    dx, dy = 1, 0
    idx = 0

    left, right = 0, w - 1
    top, bottom = 0, h - 1

    for _ in range(h * w):
        result[idx] = image_data[y, x]
        idx += 1

        if dx == 1 and x == right:
            dx, dy = 0, 1
            top += 1
        elif dy == 1 and y == bottom:
            dx, dy = -1, 0
            right -= 1
        elif dx == -1 and x == left:
            dx, dy = 0, -1
            bottom -= 1
        elif dy == -1 and y == top:
            dx, dy = 1, 0
            left += 1
        x += dx
        y += dy
    return result


class ImageProcessor:
    @staticmethod
    def process(image: Image, config: Config) -> np.ndarray:
        raise NotImplementedError


class PBPProcessor(ImageProcessor):
    @staticmethod
    def process(image: Image, config: Config) -> np.ndarray:
        image = image.convert("RGB")
        image_data = np.array(image)
        color_axis = 2

        if config.PBP.quantization_level > 0:
            ql = config.PBP.quantization_level
            image_data = (image_data / 255 * ql).round() * (255 / ql)
            image_data = image_data.astype(np.uint8)

        match config.PBP.scan_mode:
            case ScanMode.COLS:
                image_data = image_data.transpose((1, 0, 2))
            case ScanMode.ZIGZAG:
                image_data = zigzag_scan(image_data)
            case ScanMode.SPIRAL:
                image_data = spiral_scan(image_data)
                color_axis = 1

        match config.PBP.post_filter:
            case PostFilter.SUM:
                image_data = np.sum(image_data, axis=color_axis, dtype=np.uint16)
            case PostFilter.MULT:
                image_data = np.multiply.reduce(
                    image_data, axis=color_axis, dtype=np.uint16
                )
            case PostFilter.XOR:
                image_data = np.bitwise_xor.reduce(
                    image_data, axis=color_axis, keepdims=True
                )
        return image_data.flatten()


class ISMProcessor(ImageProcessor):
    @staticmethod
    def process(image: Image, config: Config) -> np.ndarray:
        image = image.convert("L")
        image_np = np.array(image, dtype=np.float32)[::-1] / 255.0

        if config.ISM.detect_edges:
            edges_x = sobel(image_np, 0)
            edges_y = sobel(image_np, 1)
            image_np = np.hypot(edges_x, edges_y)

        if config.ISM.blur_radius > 0:
            image_temp = Image.fromarray((image_np * 255).astype(np.uint8)[::-1])
            image_temp = image_temp.filter(ImageFilter.BoxBlur(config.ISM.blur_radius))
            image_np = np.array(image_temp, dtype=np.float32)[::-1] / 255.0

        if config.ISM.noise_strength > 0:
            noise = (
                    np.abs(np.random.normal(0, 1, image_np.shape))
                    * config.ISM.noise_strength
            )
            image_np *= 1 - noise

        nperseg = 2 * (image_np.shape[0] - 1)
        hop = nperseg // 4

        phase = np.random.uniform(-0.01, 0.01, size=image_np.shape)
        image_np_complex = image_np * np.exp(1j * phase)

        stfft = ShortTimeFFT.from_window(
            "hann", 1, nperseg, hop,
        )

        return stfft.istft(image_np_complex)


class AudioWriter:
    @staticmethod
    def write(
            audio_data: np.ndarray, sample_rate: int, channels: int, output_path: Path
    ) -> None:
        if channels == 2:
            if audio_data.shape[0] % 2 == 1:
                audio_data = np.append(audio_data, 0)
            audio_data = audio_data.reshape((len(audio_data) // 2, 2))
        wav_write(output_path, sample_rate, audio_data)


class SpecWriter:
    @staticmethod
    def write(
            spec_path: Path,
            img_path: Path,
            wav_path: Path,
            config: Config,
            params: Parameters,
    ) -> None:
        spec_data = {
            "timestamp": datetime.now().isoformat(),
            "image": {
                "filename": img_path.name,
                "hash": calculate_md5(img_path),
                "original_size": list(Image.open(img_path).size),
                "processed_size": [params.width, params.height],
                "scale_factor": params.scale,
                "total_pixels": params.linear_size,
            },
            "audio": {
                "filename": wav_path.name,
                "hash": calculate_md5(wav_path),
                "sample_rate": params.sample_rate,
                "sample_rate_mode": config.sample_rate_mode,
                "channels": config.channels,
            },
            "processed": {
                "mode": config.mode,
                "PBP": (
                    {
                        "quantization_level": config.PBP.quantization_level,
                        "scan_mode": config.PBP.scan_mode,
                        "post_filter": config.PBP.post_filter,
                    }
                    if config.mode == Mode.PBP
                    else None
                ),
                "ISM": (
                    {
                        "detect_edges": config.ISM.detect_edges,
                        "noise_strength": config.ISM.noise_strength,
                        "blur_radius": config.ISM.blur_radius,
                    }
                    if config.mode == Mode.ISM
                    else None
                ),
            },
        }

        with open(spec_path, "w") as f:
            json.dump(spec_data, f, indent=4)


class FilenameGenerator:
    @staticmethod
    def generate(img_name: str, config: Config, params: Parameters) -> str:
        base_name = Path(img_name).stem

        channels = "mono" if config.channels == 1 else "stereo"

        parts = [
            base_name,
            config.mode,
            f"{params.sample_rate}Hz",
            f"{channels}",
        ]

        if config.image_scale != 1:
            parts.append(f"x{config.image_scale}")
        if config.mode == Mode.PBP:
            if config.PBP.quantization_level > 0:
                parts.append(f"q{config.PBP.quantization_level}")
            parts.append(config.PBP.scan_mode)
            if config.PBP.post_filter != PostFilter.NONE:
                parts.append(config.PBP.post_filter)
        elif config.mode == Mode.ISM:
            if config.ISM.detect_edges:
                parts.append("edges")
            if config.ISM.blur_radius > 0:
                parts.append(f"blur{config.ISM.blur_radius}px")
            if config.ISM.noise_strength > 0:
                parts.append(f"noise{int(config.ISM.noise_strength * 100)}")
        return "_".join(parts) + ".wav"


class App:
    def __init__(self):
        self.config = Config.from_json(CONFIG_FILE)

    def print_config_summary(self) -> None:
        console.print("Mode: ", end="")
        console.print(f"[yellow underline]{self.config.mode}[/yellow underline]")

        console.print("Sample rate mode: ", end="")
        console.print(
            f"[cyan underline]{self.config.sample_rate_mode}[/cyan underline]"
        )

        console.print(f"Channels: [magenta]{self.config.channels}[/magenta]")

        console.print("Upscale: ", end="")
        if self.config.image_scale != 1:
            console.print(f"[green]+[/green] [dim](x{self.config.image_scale})[/dim]")
        else:
            console.print("[red]-[/red]")
        if self.config.mode == Mode.PBP:
            console.print(f"[grey27] - [/grey27] [bold]{Mode.PBP}[/bold]:")
            console.print(f"\tScan mode: [yellow]{self.config.PBP.scan_mode}[/yellow]")

            if self.config.PBP.post_filter != PostFilter.NONE:
                console.print(
                    f"\tApply [italic]{self.config.PBP.post_filter}[/italic] to [red]r[/red][green]g[/green][blue]b[/blue] components"
                )
            else:
                console.print(
                    f"\tUse [red]r[/red][green]g[/green][blue]b[/blue] components separately"
                )
        elif self.config.mode == Mode.ISM:
            console.print(f"[grey27] - [/grey27] [bold]{Mode.ISM}[/bold]:")

            console.print("\tApply noise: ", end="")
            if self.config.ISM.noise_strength > 0:
                console.print(
                    f"[green]+[/green] [dim][{self.config.ISM.noise_strength}][/dim]"
                )
            else:
                console.print("[red]-[/red]")

            console.print("\tApply blur: ", end="")
            if self.config.ISM.blur_radius > 0:
                console.print(
                    f"[green]+[/green] [dim][{self.config.ISM.blur_radius}px][/dim]"
                )
            else:
                console.print("[red]-[/red]")

            console.print("\tEdge detection: ", end="")
            console.print(
                "[green]on[/green]"
                if self.config.ISM.detect_edges
                else "[red]off[/red]"
            )
        console.print()

    def find_input_files(self) -> tuple[set[Path], set[Path]]:
        input_files = set(IN_FOLDER.iterdir())
        img_files = set(
            filter(lambda f: f.suffix.lower() in ALLOWED_IMG_FORMATS, input_files)
        )
        other_files = input_files - img_files
        return img_files, other_files

    def print_file_list(self, img_files: set[Path], other_files: set[Path]) -> None:
        console.print("\n[reverse bold]< FILES >[/reverse bold]")

        for file in img_files:
            console.print(f"\t[green]✓[/green] {file.name}")
        for file in other_files:
            console.print(f"\t[red]x[/red] [grey27]{file.name}[/grey27]")
        console.print()

    def process_image(self, img_path: Path) -> bool:
        console.print(f"[bold]:: File:[/bold] [magenta]{img_path.name}[/magenta]")

        try:
            image = Image.open(img_path)
        except OSError as e:
            console.print(
                f"[yellow on red bold]:: OS error: {e} ::[/yellow on red bold]\n"
            )
            console.print("[red]>>> SKIP[/red]\n")
            return False
        params = Parameters.from_image(image, self.config)

        wav_filename = FilenameGenerator.generate(img_path.name, self.config, params)
        wav_path = OUT_FOLDER / wav_filename
        spec_path = wav_path.with_suffix(".spec.json")

        timer = Timer().tic()

        if params.resize:
            image = image.resize((params.width, params.height), resample=Resampling.NEAREST)
            console.print(f"\t[grey27]Image resized: x{params.scale}[/grey27]")
        console.print(
            f"\tSize: [cyan]{params.width}[/cyan]×[cyan]{params.height}[/cyan]"
        )
        console.print(f"\tPixels amount: [yellow]{params.linear_size}[/yellow]px")

        if params.linear_size > MAX_IMAGE_LIN_SIZE:
            console.print("\t[yellow bold]< Image is too large >[/yellow bold]\n")
            console.print("[red]>> SKIP[/red]\n")
            return False
        if self.config.mode == Mode.ISM and (
                params.width < ISM_SIZE_THRESHOLD or params.height < ISM_SIZE_THRESHOLD
        ):
            console.print(
                f"\t[yellow bold]< ISM mode requires width and height to be at least {ISM_SIZE_THRESHOLD}px >[/yellow bold]\n"
            )
            console.print("[red]>> SKIP[/red]\n")
            return False
        console.print(f"\tSample rate: [cyan]{params.sample_rate}[/cyan]")

        console.print("[grey27]( Processing )[/grey27]")

        if self.config.mode == Mode.PBP:
            audio_data = PBPProcessor.process(image, self.config)
        else:
            audio_data = ISMProcessor.process(image, self.config)
        console.print("[grey27]( Saving )[/grey27]")

        AudioWriter.write(
            audio_data, params.sample_rate, self.config.channels, wav_path
        )
        SpecWriter.write(spec_path, img_path, wav_path, self.config, params)

        elapsed = timer.toc()

        console.print(f"[green italic] ✓ Saved as {wav_path.name}[/green italic]")
        console.print(f"[green italic]            {spec_path.name}[/green italic]")
        console.print(f"[grey27]In {elapsed:.2f} sec[/grey27]")
        console.print("")

        return True

    def main(self) -> None:
        Image.MAX_IMAGE_PIXELS = None

        IN_FOLDER.mkdir(exist_ok=True)
        OUT_FOLDER.mkdir(exist_ok=True)

        img_files, other_files = self.find_input_files()

        if not img_files:
            console.print(
                "[yellow on red bold]:: No input files ::[/yellow on red bold]\n"
            )
            return

        self.print_file_list(img_files, other_files)
        self.print_config_summary()

        total_timer = Timer().tic()
        success_count = 0

        for img_path in img_files:
            if self.process_image(img_path):
                success_count += 1

        console.print("[green italic]( Done )[/green italic]")
        console.print(f"[grey27]Processed: {success_count}/{len(img_files)}[/grey27]")
        console.print(f"[grey27]Elapsed: {total_timer.toc():.2f} sec[/grey27]")

    def run(self) -> None:
        try:
            self.main()
        except KeyboardInterrupt:
            pass
