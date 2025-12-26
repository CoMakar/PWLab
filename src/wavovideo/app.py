# -*- coding: utf-8 -*-
import colorsys
import json
import shutil
import subprocess as sp
import sys
from dataclasses import dataclass
from enum import StrEnum, Enum, IntEnum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, field_validator, ConfigDict, ValidationError
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn
from scipy.io import wavfile

from src.common.md5_file_hash import calculate_md5
from src.common.timer import Timer

APP_PATH = Path(sys.argv[0]).parent
IN_FOLDER = APP_PATH / "in"
OUT_FOLDER = APP_PATH / "out"
CONFIG_FILE = APP_PATH / "config.json"

console = Console(highlight=False)


class Mode(StrEnum):
    PBP = "PBP"
    ISM = "ISM"


class ScanMode(StrEnum):
    ROWS = "rows"
    COLS = "cols"
    ZIGZAG = "zigzag"
    SPIRAL = "spiral"


class Axis(StrEnum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class HashValidity(IntEnum):
    UNDEF = -1
    INVALID = 0
    VALID = 1


@dataclass(frozen=True)
class SpecData:
    image_filename: str
    image_hash: str
    audio_filename: str
    audio_hash: str
    processed_size: tuple[int, int]
    sample_rate: int
    mode: Mode
    scan_mode: ScanMode | None

    @classmethod
    def from_dict(cls, data: dict):
        processed = data.get("processed", {})
        pbp = processed.get("PBP") or {}

        return cls(
            image_filename=data.get("image", {}).get("filename", ""),
            image_hash=data.get("image", {}).get("hash", ""),
            audio_filename=data.get("audio", {}).get("filename", ""),
            audio_hash=data.get("audio", {}).get("hash", ""),
            processed_size=tuple(data.get("image", {}).get("processed_size", [0, 0])),
            sample_rate=data.get("audio", {}).get("sample_rate", 44100),
            mode=Mode(processed.get("mode", "PBP")),
            scan_mode=ScanMode(pbp["scan_mode"]) if pbp.get("scan_mode") else None,
        )

    def verify_image(self, img_path: Path) -> bool:
        if not self.image_hash:
            return True
        return calculate_md5(img_path) == self.image_hash

    def verify_audio(self, wav_path: Path) -> bool:
        if not self.audio_hash:
            return True
        return calculate_md5(wav_path) == self.audio_hash

    def verify_files(self, img_path: Path, wav_path: Path) -> tuple[bool, bool]:
        return self.verify_image(img_path), self.verify_audio(wav_path)

    def get_scroll_axis(self) -> Optional[Axis]:
        if self.mode == Mode.ISM:
            return Axis.HORIZONTAL
        if self.mode == Mode.PBP:
            if self.scan_mode in {ScanMode.ROWS, ScanMode.ZIGZAG, ScanMode.SPIRAL}:
                return Axis.VERTICAL
            if self.scan_mode == ScanMode.COLS:
                return Axis.HORIZONTAL
        return None


class ResolutionPreset(Enum):
    TINY = (320, 200)
    SMALL = (640, 480)
    MEDIUM = (1280, 720)
    LARGE = (1600, 900)
    FULLHD = (1920, 1080)


class Colors(Enum):
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    YELLOW = (255, 255, 0)


class BlendMode(Enum):
    NORMAL = "normal"
    DODGE = "dodge"
    DIFF = "difference"
    COLOR = "color"
    SATUR = "saturation"
    OVERLAY = "overlay"


class Config(BaseModel):
    resolution: ResolutionPreset = ResolutionPreset.FULLHD
    fps: float = Field(default=2, ge=0.1, le=10.0)
    scanner_color: Colors = Colors.WHITE
    scan_area_opacity: float = Field(default=0.1, ge=0.0, le=0.66)
    scanline_thickness: int = Field(default=1, ge=1, le=127)
    blend_mode: BlendMode = BlendMode.NORMAL

    model_config = ConfigDict(
        json_encoders={
            ResolutionPreset: lambda v: v.name,
            Colors: lambda v: v.name,
            BlendMode: lambda v: v.name
        }
    )

    @field_validator("resolution", mode="before")
    def validate_resolution(cls, v):
        try:
            return ResolutionPreset[v]
        except KeyError:
            valid_options = ", ".join([p.name for p in ResolutionPreset])
            raise ValueError(
                f"Invalid resolution preset: '{v}'. Valid options: {valid_options}"
            )

    @field_validator("scanner_color", mode="before")
    def validate_line_color(cls, v):
        try:
            return Colors[v]
        except KeyError:
            valid_options = ", ".join([c.name for c in Colors])
            raise ValueError(
                f"Invalid color: '{v}'. Valid options: {valid_options}"
            )

    @field_validator("blend_mode", mode="before")
    def validate_blend_mode(cls, v):
        try:
            return BlendMode[v]
        except KeyError:
            valid_options = ", ".join([m.name for m in BlendMode])
            raise ValueError(
                f"Invalid blend mode: '{v}'. Valid options: {valid_options}"
            )

    @field_validator("scanline_thickness")
    def validate_line_thickness_odd(cls, v):
        if v % 2 == 0:
            raise ValueError(
                f"line_thickness must be odd, got {v}. Use values like 1, 3, 5, 11..."
            )
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


class Viewport:
    def __init__(
            self,
            image_np: np.ndarray,
            axis: Optional[Axis],
            video_size: tuple[int, int],
            duration: float,
            fps: float,
    ):
        self.image = image_np
        self.axis = axis
        self.video_width, self.video_height = video_size
        self.img_height, self.img_width = self.image.shape[:2]
        self.duration = duration
        self.fps = fps
        self.scroll_speed = 0.0
        self.line_pos = None

        if axis == Axis.VERTICAL:
            self.scroll_speed = self.img_height / duration
            self.line_pos = self.video_height // 2
        elif axis == Axis.HORIZONTAL:
            self.scroll_speed = self.img_width / duration
            self.line_pos = self.video_width // 2

    def get_slice(self, frame_num: int) -> np.ndarray:
        frame = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)

        if self.scroll_speed == 0:
            x_offset = (self.video_width - self.img_width) // 2
            y_offset = (self.video_height - self.img_height) // 2
            frame[
                y_offset: y_offset + self.img_height,
                x_offset: x_offset + self.img_width,
            ] = self.image
            return frame
        offset = int(frame_num * (self.scroll_speed / self.fps))
        pos = self.line_pos - offset

        if self.axis == Axis.VERTICAL:
            start_y = max(0, pos)
            end_y = min(self.video_height, pos + self.img_height)
            if start_y < end_y:
                img_start_y = max(0, -pos)
                visible_h = end_y - start_y
                frame[start_y:end_y, :] = self.image[
                                          img_start_y: img_start_y + visible_h, :
                                          ]
        elif self.axis == Axis.HORIZONTAL:
            start_x = max(0, pos)
            end_x = min(self.video_width, pos + self.img_width)
            if start_x < end_x:
                img_start_x = max(0, -pos)
                visible_w = end_x - start_x
                frame[:, start_x:end_x] = self.image[
                                          :, img_start_x: img_start_x + visible_w
                                          ]
        return frame


class Blender:
    @staticmethod
    def dodge(strip: np.ndarray, color: tuple) -> np.ndarray:
        bottom = strip.astype(np.float32) / 255.0
        top = np.array(color, dtype=np.float32) / 255.0
        div = (1 - top)[np.newaxis, np.newaxis, :]
        mask = (top >= 1.0 - 1e-6)[np.newaxis, np.newaxis, :]
        dodge = np.minimum(1.0, bottom / np.maximum(div, 1e-6))
        full = np.ones_like(bottom)
        result = np.where(mask, full, dodge)
        return (result * 255).astype(np.uint8)

    @staticmethod
    def difference(strip: np.ndarray, color: tuple) -> np.ndarray:
        top = np.array(color)[np.newaxis, np.newaxis, :]
        return np.abs(strip.astype(np.int16) - top.astype(np.int16)).astype(np.uint8)

    @staticmethod
    def color(strip: np.ndarray, color: tuple) -> np.ndarray:
        def rgb_to_hls(r, g, b):
            return colorsys.rgb_to_hls(r, g, b)

        def hls_to_rgb(h, l, s):
            return colorsys.hls_to_rgb(h, l, s)

        bottom = strip.astype(np.float32) / 255.0
        top = np.array(color) / 255.0

        top_h, top_l, top_s = colorsys.rgb_to_hls(*top)
        bottom_r = bottom[..., 0].flatten()
        bottom_g = bottom[..., 1].flatten()
        bottom_b = bottom[..., 2].flatten()

        vec_rgb_to_hls = np.vectorize(rgb_to_hls)
        bottom_h, bottom_l, bottom_s = vec_rgb_to_hls(bottom_r, bottom_g, bottom_b)

        vec_hls_to_rgb = np.vectorize(hls_to_rgb)
        result_r, result_g, result_b = vec_hls_to_rgb(
            np.full_like(bottom_l, top_h),
            bottom_l,
            np.full_like(bottom_l, top_s)
        )

        result = np.stack([result_r, result_g, result_b], axis=-1).reshape(strip.shape) * 255
        return result.astype(np.uint8)

    @staticmethod
    def saturation(strip: np.ndarray, color: tuple) -> np.ndarray:
        def rgb_to_hls(r, g, b):
            return colorsys.rgb_to_hls(r, g, b)

        def hls_to_rgb(h, l, s):
            return colorsys.hls_to_rgb(h, l, s)

        bottom = strip.astype(np.float32) / 255.0
        top = np.array(color) / 255.0

        top_h, top_l, top_s = colorsys.rgb_to_hls(*top)
        bottom_r = bottom[..., 0].flatten()
        bottom_g = bottom[..., 1].flatten()
        bottom_b = bottom[..., 2].flatten()

        vec_rgb_to_hls = np.vectorize(rgb_to_hls)
        bottom_h, bottom_l, bottom_s = vec_rgb_to_hls(bottom_r, bottom_g, bottom_b)

        vec_hls_to_rgb = np.vectorize(hls_to_rgb)
        result_r, result_g, result_b = vec_hls_to_rgb(
            bottom_h,
            bottom_l,
            np.full_like(bottom_l, top_s)
        )

        result = np.stack([result_r, result_g, result_b], axis=-1).reshape(strip.shape) * 255
        return result.astype(np.uint8)

    @staticmethod
    def overlay(strip: np.ndarray, color: tuple) -> np.ndarray:
        bottom = strip.astype(np.float32) / 255.0
        top = np.array(color, dtype=np.float32) / 255.0
        top = top[np.newaxis, np.newaxis, :]

        mask = bottom < 0.5
        multiply = 2 * bottom * top
        screen = 1 - 2 * (1 - bottom) * (1 - top)

        result = np.where(mask, multiply, screen)
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

    @classmethod
    def apply(cls, strip: np.ndarray, color: tuple, mode: BlendMode) -> np.ndarray:
        if mode == BlendMode.DODGE:
            return cls.dodge(strip, color)
        elif mode == BlendMode.DIFF:
            return cls.difference(strip, color)
        elif mode == BlendMode.COLOR:
            return cls.color(strip, color)
        elif mode == BlendMode.SATUR:
            return cls.saturation(strip, color)
        elif mode == BlendMode.OVERLAY:
            return cls.overlay(strip, color)
        else:
            raise ValueError(f"Unknown blend mode: {mode}")


class Overlay:
    def __init__(
            self,
            scanner_color: tuple[int, int, int],
            area_opacity: float,
            line_thickness: int,
            axis: Optional[Axis],
            video_size: tuple[int, int],
            scroll_speed: float,
            fps: float,
            blend_mode: BlendMode
    ):
        self.scanner_color = scanner_color[::-1]
        self.area_opacity = area_opacity
        self.line_thickness = line_thickness
        self.axis = axis
        self.video_width, self.video_height = video_size
        self.blend_mode = blend_mode

        self.scan_step = int(scroll_speed / fps) if scroll_speed > 0 else 0

        self.line_pos = (
            self.video_height // 2
            if axis == Axis.VERTICAL
            else self.video_width // 2 if axis == Axis.HORIZONTAL else None
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        area_start_offset = self.line_thickness // 2 + 1
        half_thick = self.line_thickness // 2

        if self.axis == Axis.VERTICAL and self.line_pos is not None:
            if self.scan_step > 0:
                y_start_area = self.line_pos + area_start_offset
                y_end_area = min(self.video_height, self.line_pos + self.scan_step)
                if y_start_area < y_end_area:
                    mask = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
                    cv2.rectangle(
                        mask, (0, y_start_area), (self.video_width, y_end_area),
                        self.scanner_color, -1
                    )
                    frame[:] = cv2.addWeighted(frame, 1.0, mask, self.area_opacity, 0)

            if self.blend_mode == BlendMode.NORMAL:
                cv2.line(
                    frame,
                    (0, self.line_pos),
                    (self.video_width, self.line_pos),
                    self.scanner_color,
                    self.line_thickness,
                )
            else:
                y_start_line = max(0, self.line_pos - half_thick)
                y_end_line = min(
                    self.video_height,
                    self.line_pos + half_thick + (self.line_thickness % 2)
                )
                if y_start_line < y_end_line:
                    strip = frame[y_start_line:y_end_line, :]
                    blended = Blender.apply(strip, self.scanner_color, self.blend_mode)
                    frame[y_start_line:y_end_line, :] = blended

        elif self.axis == Axis.HORIZONTAL and self.line_pos is not None:
            if self.scan_step > 0:
                x_start_area = self.line_pos + area_start_offset
                x_end_area = min(self.video_width, self.line_pos + self.scan_step)
                if x_start_area < x_end_area:
                    mask = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
                    cv2.rectangle(
                        mask, (x_start_area, 0), (x_end_area, self.video_height),
                        self.scanner_color, -1
                    )
                    frame[:] = cv2.addWeighted(frame, 1.0, mask, self.area_opacity, 0)

            if self.blend_mode == BlendMode.NORMAL:
                cv2.line(
                    frame,
                    (self.line_pos, 0),
                    (self.line_pos, self.video_height),
                    self.scanner_color,
                    self.line_thickness,
                )
            else:
                half_thick = self.line_thickness // 2
                x_start_line = max(0, self.line_pos - half_thick)
                x_end_line = min(
                    self.video_width,
                    self.line_pos + half_thick + (self.line_thickness % 2)
                )
                if x_start_line < x_end_line:
                    strip = frame[:, x_start_line:x_end_line]
                    blended = Blender.apply(strip, self.scanner_color, self.blend_mode)
                    frame[:, x_start_line:x_end_line] = blended

        return frame


class Renderer:
    def __init__(self, spec_path: Path, config: Config):
        self.spec_path = spec_path
        self.config = config

        self.spec = self.load_spec()
        self.image_path = IN_FOLDER / self.spec.image_filename
        self.wav_path = spec_path.parent / spec_path.name.replace(".spec.json", ".wav")

        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        if not self.wav_path.exists():
            raise FileNotFoundError(f"Audio not found: {self.wav_path}")
        self.duration = self.get_duration()

    def load_spec(self) -> SpecData:
        with open(self.spec_path, "r", encoding="utf-8") as f:
            return SpecData.from_dict(json.load(f))

    def get_duration(self) -> float:
        sample_rate, audio_data = wavfile.read(self.wav_path)
        return len(audio_data) / sample_rate

    def prepare_image(self, axis: Optional[Axis]) -> np.ndarray:
        iw, ih = self.spec.processed_size
        vw, vh = self.config.resolution.value

        if axis == Axis.VERTICAL:
            scale = vw / iw
            target_size = (vw, int(ih * scale))
        elif axis == Axis.HORIZONTAL:
            scale = vh / ih
            target_size = (int(iw * scale), vh)
        else:
            scale = min(vw / iw, vh / ih)
            target_size = (int(iw * scale), int(ih * scale))
        image = Image.open(self.image_path)

        if image.size != self.spec.processed_size:
            image = image.resize(self.spec.processed_size, Image.NEAREST)
        if target_size != image.size:
            image = image.resize(target_size, Image.LANCZOS)
        image_rgb = np.array(image)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        return image_bgr

    def render(self, output_path: Path) -> bool:
        axis = self.spec.get_scroll_axis()
        image_bgr = self.prepare_image(axis)

        total_frames = int(self.duration * self.config.fps)
        fps = self.config.fps
        vw, vh = self.config.resolution.value

        viewport = Viewport(image_bgr, axis, (vw, vh), self.duration, fps)
        overlay = Overlay(
            self.config.scanner_color.value,
            self.config.scan_area_opacity,
            self.config.scanline_thickness,
            axis,
            (vw, vh),
            viewport.scroll_speed,
            fps,
            self.config.blend_mode
        )

        cmd = [
            "ffmpeg", "-y",
            "-v", "quiet",
            "-stats",
            "-threads", "0",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{vw}x{vh}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",
            "-i", str(self.wav_path),
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "stillimage",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-t", str(self.duration),
            "-vsync", "0",
            str(output_path),
        ]

        with Progress(
                TextColumn("Progress"),
                BarColumn(
                    bar_width=20,
                    style="grey27",
                    complete_style="green",
                    finished_style="green",
                ),
                TextColumn("[cyan]{task.percentage:>3.0f}%[/cyan]"),
                console=console,
                transient=True,
        ) as progress:

            task = progress.add_task("rendering", total=total_frames)

            try:
                proc = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.STDOUT)

                try:
                    for frame_num in range(total_frames):
                        frame = viewport.get_slice(frame_num)
                        frame = overlay.apply(frame)
                        proc.stdin.write(frame.tobytes())
                        progress.update(task, advance=1)
                finally:
                    if proc.stdin:
                        proc.stdin.close()
                    proc.wait()
                success = proc.returncode == 0

                return success
            except Exception as e:
                console.print(
                    f"[yellow on red bold]:: Error: {e} ::[/yellow on red bold]"
                )
                return False


class App:
    def __init__(self):
        self.config = Config.from_json(CONFIG_FILE)

    def find_file_groups(self):
        specs = list(IN_FOLDER.glob("*.spec.json"))
        groups = []

        for spec_path in specs:
            with open(spec_path) as f:
                spec = SpecData.from_dict(json.load(f))
            wav_path = IN_FOLDER / spec.audio_filename
            img_path = IN_FOLDER / spec.image_filename

            if wav_path.exists() and img_path.exists():
                image_valid, audio_valid = spec.verify_files(img_path, wav_path)
                hash_validity = [HashValidity.INVALID, HashValidity.VALID][image_valid & audio_valid]
            else:
                hash_validity = HashValidity.UNDEF

            groups.append(
                {
                    "spec": spec_path,
                    "wav": wav_path,
                    "image": img_path,
                    "spec_exists": spec_path.exists(),
                    "wav_exists": wav_path.exists(),
                    "image_exists": img_path.exists(),
                    "hash_validity": hash_validity,
                }
            )
        return groups

    def print_file_groups(self, groups):
        console.print("\n[reverse bold]< GROUPS >[/reverse bold]")

        for group in groups:
            group_name = group["spec"].stem.replace(".spec", "")

            if group["hash_validity"] is HashValidity.INVALID:
                console.print(
                    f"\t[red]x[/red] [red underline]hash mismatch[/red underline] [dim]{group_name}[/dim]"
                )
                continue
            total = 3
            present = sum(
                [group["spec_exists"], group["wav_exists"], group["image_exists"]]
            )

            if present == total:
                status = f"[green]✓ {present}/{total}[/green]"
                name_style = "white"
            else:
                status = f"[yellow]x {present}/{total}[/yellow]"

                name_style = "dim"
            console.print(f"\t{status} [{name_style}]{group_name}[/{name_style}]")
        console.print()

    def print_config(self):
        vw, vh = self.config.resolution.value
        scanner_color = {
            Colors.WHITE: {"name": "White", "style": "white"},
            Colors.BLACK: {"name": "Black", "style": "black on white"},
            Colors.RED: {"name": "Red", "style": "red"},
            Colors.GREEN: {"name": "Green", "style": "green"},
            Colors.BLUE: {"name": "Blue", "style": "blue"},
            Colors.CYAN: {"name": "Cyan", "style": "cyan"},
            Colors.MAGENTA: {"name": "Magenta", "style": "magenta"},
            Colors.YELLOW: {"name": "Yellow", "style": "yellow"}
        }[self.config.scanner_color]

        console.print(f"Resolution: [cyan]{vw}[/cyan]×[cyan]{vh}[/cyan]")
        console.print(f"FPS: [cyan]{self.config.fps}[/cyan]")
        console.print(f"Scan : line: [{scanner_color['style']}]{scanner_color['name']}[/{scanner_color['style']}] "
                      f"[cyan]{self.config.scanline_thickness}[/cyan]px"
                      )
        console.print(f"     : area opacity: [cyan]{round(self.config.scan_area_opacity * 100, 2)}[/cyan]%")
        console.print(f"Blend mode: [yellow]{self.config.blend_mode.value.capitalize()}[/yellow]")
        console.print()

    def process_group(self, group):
        group_name = group["spec"].stem.replace(".spec", "")
        console.print(f"[bold]:: Group:[/bold] [magenta]{group_name}[/magenta]")

        if not (group["spec_exists"] and group["wav_exists"] and group["image_exists"]):
            console.print(
                "\t[yellow on red bold]:: Missing files ::[/yellow on red bold]"
            )
            console.print("[red]>>> SKIP[/red]\n")
            return False

        if group["hash_validity"] in [HashValidity.INVALID, HashValidity.UNDEF]:
            console.print(
                "\t[yellow on red bold]:: Hash is invalid or undefined ::[/yellow on red bold]"
            )
            console.print("[red]>>> SKIP[/red]\n")
            return False

        try:
            renderer = Renderer(group["spec"], self.config)

            console.print(
                f"\tImage: [cyan]{renderer.spec.processed_size[0]}[/cyan]×[cyan]{renderer.spec.processed_size[1]}[/cyan]"
            )

            console.print(f"\tMode: [yellow]{renderer.spec.mode}[/yellow]", end="")
            console.print(" | ", end="")
            console.print(f"[yellow]{renderer.spec.scan_mode}[/yellow]")

            if renderer.spec.scan_mode == ScanMode.SPIRAL:
                console.print(
                    "\t[yellow on red bold]:: Unsupported scan mode ::[/yellow on red bold]"
                )
                console.print("[red]>>> SKIP[/red]\n")
                return False

            console.print(f"\tDuration: [yellow]{renderer.duration:.1f}[/yellow]s")

            total_frames = int(renderer.duration * self.config.fps)
            console.print(f"\tFrames: [cyan]{total_frames}[/cyan]")

            output_path = OUT_FOLDER / (group_name + ".mp4")

            console.print("")
            console.print("[grey27]( Rendering )[/grey27]")

            timer = Timer().tic()
            success = renderer.render(output_path)
            elapsed = timer.toc()

            if success:
                console.print("[grey27]( Saving )[/grey27]")
                console.print(f"[green italic] ✓ Saved as {output_path.name}[/green italic]")
                console.print(f"[grey27]In {elapsed:.2f} sec[/grey27]")
            else:
                console.print(
                    "[yellow on red bold]:: Rendering failed ::[/yellow on red bold]"
                )
            console.print("")
            return success
        except Exception as e:
            console.print(
                f"[yellow on red bold]:: Error: {e} ::[/yellow on red bold]"
            )
            console.print("[red]>>> SKIP[/red]\n")
            return False

    def main(self):
        Image.MAX_IMAGE_PIXELS = None

        if shutil.which("ffmpeg") is None:
            console.print(
                "[yellow on red bold]:: FFmpeg not found ::[/yellow on red bold]"
            )
            return
        IN_FOLDER.mkdir(exist_ok=True)
        OUT_FOLDER.mkdir(exist_ok=True)

        groups = self.find_file_groups()

        if not groups:
            console.print(
                "[yellow on red bold]:: No spec files found ::[/yellow on red bold]"
            )
            return
        self.print_file_groups(groups)

        self.print_config()

        valid_groups = [
            g
            for g in groups
            if g["spec_exists"] and g["wav_exists"] and g["image_exists"]
        ]

        if not valid_groups:
            console.print(
                "[yellow on red bold]:: No valid file groups found ::[/yellow on red bold]"
            )
            return

        total_timer = Timer().tic()
        success_count = 0

        for group in valid_groups:
            if self.process_group(group):
                success_count += 1

        console.print("[green italic]( Done )[/green italic]")
        console.print(f"[grey27]Rendered: {success_count}/{len(valid_groups)}[/grey27]")
        console.print(f"[grey27]Elapsed: {total_timer.toc():.2f} sec[/grey27]")

    def run(self):
        try:
            self.main()
        except KeyboardInterrupt:
            pass
