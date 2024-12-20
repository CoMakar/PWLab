# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 21:07:56 2020
"""


from dataclasses import dataclass
import io
import math
import sys
from pathlib import Path
from typing import Union
from enum import StrEnum, IntEnum

import numpy as np
from PIL import Image
from scipy.io.wavfile import write as wav_write
from scipy.signal import ShortTimeFFT
from scipy.ndimage import sobel
from cerberus import Validator

from src.common.force_input import force_input
from src.common.json_config import JSONConfig
from src.common.timer import Timer
from src.term_utils import *


Number = Union[int, float]


class Mode(StrEnum):
    ISM = "ISM"
    PBP = "PBP"


class SampleRateMode(StrEnum):
    STATIC = "static"
    DYNAMIC = "dynamic"


class Direction(StrEnum):
    ROWS = "rows"
    COLS = "cols"


APP_PATH = Path(sys.argv[0]).parent
IN_FOLDER = APP_PATH / "in"
OUT_FOLDER = APP_PATH / "out"
CFG_PATH = APP_PATH / "config.json"

F_INFO = Format(fg=FG.BLUE)
F_ERROR = Format(fg=FG.YEL, bg=BG.RED, style=STYLE.BOLD)
F_OK = Format(fg=FG.GREEN, style=STYLE.ITALIC)
F_INVERTED = Format(style=STYLE.REVERSE)
C_GRAY = FGRGB(64, 64, 64)
C_ORANGE = FGRGB(255, 128, 0)
F_MISC = Format(fg=C_GRAY)
F_WARNING = Format(fg=C_ORANGE, style=STYLE.BOLD)

ISM_SIZE_THRESHOLD = 64
ALLOWED_IMG_FORMATS = (".png", ".jpg", ".jpeg")
MAX_IMAGE_LIN_SIZE = 1e10
REMAPPER_POWER = 0.5
REMAPPER_EXP_KOEF = 1

MODES = (Mode.ISM, Mode.PBP)
DIRECTIONS = (Direction.ROWS, Direction.COLS)
SAMPLERATE_MODES = (SampleRateMode.STATIC, SampleRateMode.DYNAMIC)
SAMPLERATE_RANGE = range(512, 262144 + 1)
IMG_SCALE_RANGE = range(1, 10 + 1)


class Resolution(IntEnum):
    LOW = 640 * 320
    MID = 7680 * 4320
    HIGH = 15360 * 8640


class SampleRateRange(IntEnum):
    MIN = SAMPLERATE_RANGE[0]
    LOW = 4096
    HIGH = 65536
    MAX = SAMPLERATE_RANGE[-1]


CFG_DEFAULT = JSONConfig(
    {
        "mode": Mode.PBP,
        "channels": 1,
        "sample_rate_mode": SampleRateMode.STATIC,
        "sample_rate": 44100,
        "image_scale": 1,
        "PBP": {
            "sum_rgb_components": False,
            "direction": Direction.ROWS
        },
        "ISM": {
            "use_noise": True,
            "noise_strength": 0.5,
            "detect_edges": False
        }
    }, True
)

CFG_VALIDATOR = Validator(
    {
        "mode": {"type": "string", "allowed": MODES},
        "channels": {"type": "integer", "allowed": [1, 2]},
        "sample_rate_mode": {"type": "string", "allowed": SAMPLERATE_MODES},
        "sample_rate": {
            "type": "integer",
            "min": SAMPLERATE_RANGE[0],
            "max": SAMPLERATE_RANGE[-1],
        },
        "image_scale": {
            "type": "integer",
            "min": IMG_SCALE_RANGE[0],
            "max": IMG_SCALE_RANGE[-1],
        },
        "PBP": {
            "type": "dict",
            "schema": {
                "sum_rgb_components": {"type": "boolean"},
                "direction": {
                    "type": "string",
                    "allowed": DIRECTIONS,
                },
            },
        },
        "ISM": {
            "type": "dict",
            "schema": {
                "use_noise": {"type": "boolean"},
                "noise_strength": {"type": "float", "min": 0.0, "max": 1.0},
                "detect_edges": {"type": "boolean"},
            }
        },
    }, require_all=True
)


@dataclass(frozen=True)
class Parameters:
    resize: bool
    scale: int
    width: int
    height: int
    linear_size: int
    sample_rate: int

    @classmethod
    def from_image(cls, image: Image, config: JSONConfig):
        resize = config.image_scale != 1
        scale = config.image_scale
        size = (
            round(image.size[0] * scale),
            round(image.size[1] * scale)
        )
        width, height = size
        linear_size = width * height

        sample_rate = (
            config.sample_rate
            if config.sample_rate_mode is SampleRateMode.STATIC
            else round(
                remap_sample_rate_from_size(
                    linear_size, SampleRateRange.MIN, SampleRateRange.LOW, SampleRateRange.HIGH, SampleRateRange.MAX,
                    Resolution.LOW, Resolution.MID, Resolution.HIGH,
                    REMAPPER_POWER, REMAPPER_EXP_KOEF
                )
            )
        )

        return cls(resize, scale, width, height, linear_size, sample_rate)


def pbp(image: Image, linear_size: int,
        direction: str, use_sum_rgb: bool):
    # PBP - pixel by pixel method

    image = image.convert("RGB")
    image_data = np.array(image)

    if direction == "column":
        # transpose an image to replace rows with columns
        image_data = image_data.transpose((1, 0, 2))

    if use_sum_rgb:
        # sum rgb components into a single value
        image_data = image_data.reshape(linear_size, 3)
        image_data = np.sum(image_data, 1, np.uint8)

    # return as 1dim array of audio data
    return image_data.flatten()


def ism(image: Image,
        use_noise: bool, noise_strength: float,
        detect_edges: bool):
    # ISM - inverse spectrogram method of image decoding into sound data

    image = image.convert("L")
    image_data = np.array(image)[::-1] / 255

    if use_noise:
        noise = np.abs(np.random.normal(
            0, 1, image_data.shape
        )) * noise_strength
        image_data *= 1 - noise

    if detect_edges:
        edges_x = sobel(image_data, 0)
        edges_y = sobel(image_data, 1)
        image_data = np.hypot(edges_x, edges_y)

    stfft = ShortTimeFFT.from_window(
        "cosine", 1,
        image_data.shape[0] * 2 - 1,
        image_data.shape[0]
    )

    return stfft.istft(image_data)


def clamp(value: Number,
          min_value: Number = 0, max_value: Number = 1):
    return max(min(value, max_value), min_value)


def remap_sample_rate_from_size(linear_size,
                                min_sample_rate, low_sample_rate, high_sample_rate, max_sample_rate,
                                low_res, mid_res, high_res,
                                power, exp_multiplier):

    if linear_size <= low_res:
        sample_rate = min_sample_rate + \
            ((linear_size / low_res) ** power) * \
            (low_sample_rate - min_sample_rate)

    elif linear_size <= mid_res:
        normalized_area = (linear_size - low_res) / (mid_res - low_res)
        sample_rate = low_sample_rate + normalized_area * \
            (high_sample_rate - low_sample_rate)

    else:
        normalized_area = (linear_size - mid_res) / (high_res - mid_res)
        sample_rate = high_sample_rate + \
            (max_sample_rate - high_sample_rate) * \
            (1 - math.exp(-normalized_area * exp_multiplier))

    return round(sample_rate)


class App:
    def main(self):
        Image.MAX_IMAGE_PIXELS = None

        IN_FOLDER.mkdir(exist_ok=True)
        OUT_FOLDER.mkdir(exist_ok=True)

        config = self.ensure_config()

        if not config:
            writef(f":: No config ::", F_ERROR)
            return

        input_files = set((APP_PATH / "in").iterdir())
        img_files = set(
            filter(
                lambda f: f.suffix in ALLOWED_IMG_FORMATS, input_files
            )
        )
        other_files = input_files - img_files

        if len(img_files) == 0:
            writef(":: No input files ::", F_ERROR)
            write()
            return

        writef("< FILES >", F_INVERTED)
        write()
        [write(f"\t{ffg('+', FG.GREEN)} {file.name}\n") for file in img_files]
        [write(f"\t{ffg('-', FG.RED)} {ffg(file.name, C_GRAY)}\n")
         for file in other_files]
        
        write()

        write("Mode: ")
        write(fstyle(ffg(config.mode, FG.YEL), STYLE.UNDER))
        write()

        write("Sample rate mode: ")
        write(fstyle(ffg(config.sample_rate_mode, FG.CYAN), STYLE.UNDER))
        write()

        write("Channels: ")
        write(ffg(config.channels, FG.MAGNT))
        write()

        write(f" x Upscale: ")
        write(
            ffg("+", FG.GREEN)
            if config.image_scale != 1
            else ffg("-", FG.RED)
        )
        if config.image_scale != 1:
            write(f" └ Scale factor: {ffg(config.image_scale, FG.BLUE)}\n")
        write()

        if config.mode == Mode.PBP:
            write(f"{ffg('[*]', C_GRAY)} {fstyle(Mode.PBP, STYLE.BOLD)}:\n")
            write(f"\tDirection: {ffg(config.PBP.direction, FG.YEL)}\n")
            if config.PBP.sum_rgb_components:
                write(
                    f"\tUse sum of {ffg('r', FG.RED)}{ffg('+', C_GRAY)}{ffg('g', FG.GREEN)}{ffg('+', C_GRAY)}{ffg('b', FG.BLUE)} components"
                )
            else:
                write(
                    f"\tUse {ffg('r', FG.RED)}{ffg('g', FG.GREEN)}{ffg('b', FG.BLUE)} components separately"
                )
            write()

        elif config.mode == Mode.ISM:
            write(f"{ffg('[*]', C_GRAY)} {fstyle(Mode.ISM, STYLE.BOLD)}:\n")
            write(f"\t x Apply noise: ")
            write(
                ffg("+", FG.GREEN)
                if config.ISM.use_noise
                else ffg("-", FG.RED)
            )
            if config.ISM.use_noise:
                write(
                    f"\t └ Noise strength: {ffg(config.ISM.noise_strength, FG.YEL)}\n"
                )
            write()

            write("\tEdge detection mode: ")
            write(
                ffg("on", FG.GREEN)
                if config.ISM.detect_edges
                else ffg("off", FG.RED)
            )
            write()

        write()

        total_timer = Timer().tic()

        for img_path in img_files:
            write(
                f"{fstyle(':: File:', STYLE.BOLD)} {ffg(img_path.name, FG.MAGNT)}\n"
            )

            try:
                with open(img_path, 'rb') as img_file:
                    image = Image.open(io.BytesIO(img_file.read()))

            except OSError as e:
                writef(f":: OS error: {e} ::", F_ERROR)
                write()
                write(f"{ffg('>>> SKIP', FG.RED)}\n")
                continue

            params = Parameters.from_image(image, config)
            wav_path = (
                OUT_FOLDER /
                f"{img_path.name}_{config.mode}_{params.sample_rate}HZ.wav"
            )
            current_timer = Timer().tic()

            if params.scale != 1:
                image = image.resize((params.width, params.height),
                                     resample=Image.NEAREST)
                writef(f"\tImage resized: x{params.scale}", F_MISC)

            write(
                f"\tSize: {ffg(params.width, FG.BLUE)}x{ffg(params.height, FG.BLUE)}\n"
            )
            write(f"\tPixels amount: {ffg(params.linear_size, FG.YEL)}px\n")

            if params.linear_size > MAX_IMAGE_LIN_SIZE:
                writef("\t< Image is too large >", F_WARNING)
                write()
                write(f"{ffg('>> SKIP', FG.RED)}\n")
                continue

            if config.mode == Mode.ISM and (params.width < ISM_SIZE_THRESHOLD or params.height < ISM_SIZE_THRESHOLD):
                writef(
                    f"\t< ISM mode requires width and height to be at least {ISM_SIZE_THRESHOLD}px >", F_WARNING
                )
                write()
                write(f"{ffg('>> SKIP', FG.RED)}\n")
                continue

            write(f"\tSample rate: {ffg(params.sample_rate, FG.CYAN)}\n")

            audio_data = None

            write(ffg("( Processing )\n", C_GRAY))

            if config.mode == "PBP":
                audio_data = pbp(
                    image, params.linear_size,
                    config.PBP.direction,
                    config.PBP.sum_rgb_components
                )

            elif config.mode == "ISM":
                audio_data = ism(
                    image, config.ISM.use_noise,
                    config.ISM.noise_strength,
                    config.ISM.detect_edges
                )

            Cur.prev_line()
            Scr.clear_line()

            if config.channels == 2:
                if audio_data.shape[0] % 2 == 1:
                    audio_data = np.append(audio_data, 0)
                audio_data = audio_data.reshape(
                    (len(audio_data) // 2, 2)
                )

            write(ffg("( Saving )\n", C_GRAY))

            wav_write(
                wav_path, params.sample_rate, audio_data
            )

            Cur.prev_line()
            Scr.clear_line()

            writef(f"\tSaved as {wav_path}\n", F_MISC)
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

            for item, errors in CFG_VALIDATOR.errors.items():
                err_str = f"\t{ffg('>', FG.RED)} {item:<30} : {ffg(';'.join(errors), FG.RED)}\n"
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

    def run(self):
        Scr.color_on()
        try:
            self.main()
        except KeyboardInterrupt:
            pass
        Scr.reset_mode()
