# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 21:07:56 2020
@configor: Qwek
"""


# +
# + [!] needs refactoring. At the moment it's just a mishmash of code
# +


import io
import math
import os
import sys
import re
from typing import Union
from enum import StrEnum

import numpy as np
from PIL import Image
from scipy.io.wavfile import write as wav_write
from scipy.signal import ShortTimeFFT
from scipy.ndimage import sobel
from cerberus import Validator, SchemaError

from src.Common.force_input import force_input
from src.Common.json_config import JSONConfig
from src.Common.timer import Timer
from src.TermUtils import *


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


CFG_NAME = "config.json"
APP = os.path.dirname(sys.argv[0])
MAX_IMAGE_LIN_SIZE = 1e8
F_INFO = Format(fg=FG.BLUE)
F_ERROR = Format(fg=FG.YEL, bg=BG.RED, style=STYLE.BOLD)
F_OK = Format(fg=FG.GREEN, style=STYLE.ITALIC)
F_INVERTED = Format(style=STYLE.REVERSE)
C_GRAY = FGRGB(64, 64, 64)


MODES = (Mode.ISM, Mode.PBP)
DIRECTIONS = (Direction.ROWS, Direction.COLS)
SAMPLERATE_MODES = (SampleRateMode.STATIC, SampleRateMode.DYNAMIC)
SAMPLERATE_RANGE = range(512, 256000 + 1)
IMG_SCALE_RANGE = range(1, 10 + 1)


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


def pbp(image: Image, linear_size: int,
        direction: str, use_sum_rgb: bool):
    # +
    # + PBP - pixel by pixel method of image decoding into sound data
    # +
    
    image = image.convert('RGB')
    image_data = np.array(image)

    if direction == "column":
        # transpose an image to replace rows with columns
        # [0, 1]    =>    [0, 2]
        # [2, 3]    =>    [1, 3]
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
    # +
    # + ISM - inverse spectrogram method of image decoding into sound data
    # +

    image = image.convert("L")
    image_data = np.array(image)[::-1] / 255
    
    if use_noise:
        noise = np.abs(np.random.normal(0, 1, 
                                        image_data.shape)) * noise_strength
        image_data *= 1 - noise

    if detect_edges:
        edges_x = sobel(image_data, 0)
        edges_y = sobel(image_data, 1)
        image_data = np.hypot(edges_x, edges_y)
        
    stfft = ShortTimeFFT.from_window(('cosine'), 1, 
                                     image_data.shape[0] * 2 - 1, 
                                     image_data.shape[0])

    return stfft.istft(image_data)


def clamp(value: Number,
          min_value: Number = 0, max_value: Number = 1):
    return max(min(value, max_value), min_value)


def clamped_log_remap(value: Number,
                      from_min: Number, from_max: Number,
                      to_min: Number, to_max: Number):
    
    if from_min <= 0 or from_max <= 0 or to_min <= 0 or to_max <= 0:
        raise ValueError("Cannot remap from negative")
    if from_min >= from_max:
        raise ValueError("Range size: 0 or negative")
    if to_min >= to_max:
        raise ValueError("Range size: 0 or negative")
    if value < from_min or value > from_max:
        raise ValueError("Value out of range")
    
    log_scale = math.log(to_max / to_min) / math.log(from_max / from_min)
    scaled_value = (value - from_min) / (from_max - from_min)
    return clamp(to_min * math.pow(to_max / to_min, scaled_value ** log_scale), to_min, to_max)


def clamped_linear_remap(value: Number,
                          from_min: Number, from_max: Number,
                          to_min: Number, to_max: Number):

    if from_min >= from_max:
        raise ValueError("Range size: 0 or negative")
    if to_min >= to_max:
        raise ValueError("Range size: 0 or negative")
    if value < from_min or value > from_max:
        raise ValueError("Value out of range")
    
    scale = (value - from_min) / (from_max - from_min)
    return clamp(to_min + scale * (to_max - to_min), to_min, to_max)


def main():
    os.chdir(APP)
    os.makedirs("in", exist_ok=True)
    os.makedirs("out", exist_ok=True)

    try:
        config = JSONConfig.load(CFG_NAME, True)

    except FileNotFoundError:
        CFG_DEFAULT.save(CFG_NAME)
        config = JSONConfig(CFG_DEFAULT.as_dict(), True)
        writef("( Config file created )", F_INFO)
        write("\n\n")
    
    except OSError:
        writef(":: Unknown OS error ::", F_ERROR)
        write()
        Scr.reset_mode()
        sys.exit(1)
        
    try:
        if not CFG_VALIDATOR.validate(config.as_dict()):
            raise SchemaError
    
    except SchemaError:
        write("Config validation failed:\n")
        
        for k, v in CFG_VALIDATOR.errors.items():
            reason = re.sub(r"[\[\]\{\}']", "", str(v))
            err_str = f"\t{ffg('>', FG.RED)} {k:<30} : {ffg(reason, FG.RED)}"
            print(err_str)
        
        write()
        
        write(f"Reset {CFG_NAME}?\n")
        write(f"{ffg('( [y/n] )', C_GRAY)}\n")
        yn = force_input(ffg(">>> ", FG.GREEN), ffg("( [y/n] )", C_GRAY),
                         func=str.lower,
                         predicates=[lambda x: x in ('y', 'n')])
        if yn == 'y':
            CFG_DEFAULT.save(CFG_NAME)

        Scr.reset_mode()
        sys.exit(1)

    mode = config.mode
    channels = config.channels
    sample_rate_mode = config.sample_rate_mode
    sample_rate = config.sample_rate
    image_scale = config.image_scale
    pbp_sum_rgb_components = config.PBP.sum_rgb_components
    pbp_direction = config.PBP.direction
    ism_use_noise = config.ISM.use_noise
    ism_noise_strength = config.ISM.noise_strength
    ism_detect_edges = config.ISM.detect_edges

    input_files = set(os.listdir('./in'))
    img_files = set(filter(lambda f:
                           f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg"), input_files))
    other_files = input_files - img_files

    if len(img_files) == 0:
        writef(":: No input files! ::", F_ERROR)
        write()
        Scr.reset_mode()
        sys.exit(1)

    writef("< FILES >", F_INVERTED)
    write()
    [write(f"\t> {file}\n") for file in img_files]
    [write(ffg(f"\t- {file}\n", FG.RED)) for file in other_files]

    write()

    write("Mode: ")
    write(fstyle(ffg(mode, FG.YEL), STYLE.UNDER))
    write()
    
    write(" + Sample rate mode: ")
    write(fstyle(ffg(sample_rate_mode, FG.CYAN), STYLE.UNDER))
    write()

    if sample_rate_mode == SampleRateMode.STATIC:
        write(f" └ Sample rate: {ffg(sample_rate, FG.CYAN)}")
    else:
        write(f" └ Sample rate: {ffg('~', C_GRAY)}")
    write()

    write("Channels: ")
    write(ffg(channels, FG.MAGNT))
    write()

    write(" + Upscale: ")
    if image_scale != 1:
        write(ffg("+", FG.GREEN))
    else:
        write(ffg("-", FG.RED))
    write()
    
    write(f" └ Scale factor: {ffg(image_scale, FG.BLUE)}\n")

    write()

    if mode == Mode.PBP:
        write(f"{ffg('[*]', C_GRAY)} {fstyle(Mode.PBP, STYLE.BOLD)}:\n")
        write(f"\tDirection: {ffg(pbp_direction, FG.YEL)}\n")
        if pbp_sum_rgb_components:
            write(
                f"\tUse sum of {ffg('r', FG.RED)}{ffg('+', C_GRAY)}{ffg('g', FG.GREEN)}{ffg('+', C_GRAY)}{ffg('b', FG.BLUE)} components"
                )
        else:
            write(f"\tUse {ffg('r', FG.RED)}{ffg('g', FG.GREEN)}{ffg('b', FG.BLUE)} components separately")

    elif mode == Mode.ISM:
        write(f"{ffg('[*]', C_GRAY)} {fstyle(Mode.ISM, STYLE.BOLD)}:\n")
        write(f"\t + Apply noise: ")
        if ism_use_noise:
            write(ffg("+", FG.GREEN))
        else:
            write(ffg("-", FG.RED))
        write()
        
        write(f"\t └ Noise strength: {ffg(ism_noise_strength, FG.YEL)}\n")
        
        write("\tEdge detection mode: ")
        if ism_detect_edges:
            write(ffg("on", FG.GREEN))
        else:
            write(ffg("off", FG.RED))
        write()

    write()

    with Timer(fstyle("Total", STYLE.BOLD)):
        for file in img_files:
            write(f"{fstyle(':: File:', STYLE.BOLD)} {ffg(file, FG.MAGNT)}\n")

            try:
                with open(f"./in/{file}", 'rb') as file_img:
                    image = Image.open(io.BytesIO(file_img.read()))

            except Exception as e:
                writef(f":: File error: {e} ::", F_ERROR)
                write()
                write(f"{ffg('>> SKIP', FG.RED)}\n")
                continue

            if image_scale != 1:
                orig_size = image.size
                image = image.resize((round(image.size[0] * image_scale),
                                      round(image.size[1] * image_scale)),
                                     resample=Image.NEAREST)

                writef("\t( Image resized )", F_INFO)
                write()
                write(f"\t + Scale: {ffg(image_scale, FG.BLUE)}\n")
                write(f"\t └ Original size: {ffg(orig_size[0], FG.BLUE)}x{ffg(orig_size[1], FG.BLUE)}\n")

            width, height = image.size
            image_lin_size = width * height

            write(f"\tSize: {ffg(width, FG.BLUE)}x{ffg(height, FG.BLUE)}\n")
            write(f"\tPixels amount: {ffg(image_lin_size, FG.YEL)}px\n")

            if image_lin_size > MAX_IMAGE_LIN_SIZE:
                writef("\t:: Image is too large ::", F_ERROR)
                write()
                write(f"{ffg('>> SKIP', FG.RED)}\n")
                continue
            
            if mode == Mode.ISM and (width < 64 or height < 64):
                writef(
                    f"\t:: ISM mode requires width\height to be 64px+ ::", F_ERROR)
                write()
                write(f"{ffg('>> SKIP', FG.RED)}\n")
                continue

            if sample_rate_mode == SampleRateMode.DYNAMIC:
                sample_rate = round(clamped_linear_remap(image_lin_size,
                                                    1, MAX_IMAGE_LIN_SIZE,
                                                    SAMPLERATE_RANGE[0], SAMPLERATE_RANGE[-1]))
                write(f"\tSample rate: {ffg(sample_rate, FG.CYAN)}\n")

            audio_data = None

            with Timer(fstyle(file, STYLE.BOLD)):
                write(ffg("( Processing )\n", C_GRAY))
                
                if mode == "PBP":
                    audio_data = pbp(image, image_lin_size,
                                     pbp_direction, pbp_sum_rgb_components)

                elif mode == "ISM":
                    audio_data = ism(image, ism_use_noise, ism_noise_strength,
                                     ism_detect_edges)

                Cur.prev_line()
                Scr.clear_line()

                if channels == 2:
                    if audio_data.shape[0] % 2 == 1:
                        audio_data = np.append(audio_data, 0)
                    audio_data = audio_data.reshape((len(audio_data) // 2, 2))

                write(ffg("( Saving )\n", C_GRAY))

                wav_write(f"./out/{file}_{mode}_{sample_rate}HZ.wav", sample_rate, audio_data)

                Cur.prev_line()
                Scr.clear_line()

            write()

    writef("( Done! )\n", F_OK)


if __name__ == '__main__':
    Scr.color_on()
    try:
        main()
    except KeyboardInterrupt:
        pass
    Scr.reset_mode()
