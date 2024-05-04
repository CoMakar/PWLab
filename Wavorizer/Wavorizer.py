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
from typing import Union

import numpy as np
from PIL import Image
from scipy.io.wavfile import write as wav_write
from scipy.signal import istft

from src.Common.force_input import force_input
from src.Common.JSONConfig import JSONConfig
from src.Common.Timer import Timer
from src.TermUtils import *


CFG_NAME = "config.json"
APP = os.path.dirname(sys.argv[0])
MAX_IMAGE_LIN_SIZE = 1e8
F_INFO = Format(fg=FG.BLUE)
F_ERROR = Format(fg=FG.YEL, bg=BG.RED, style=STYLE.BOLD)
F_OK = Format(fg=FG.GREEN, style=STYLE.ITALIC)
F_INVERTED = Format(style=STYLE.REVERSE)

Number = Union[int, float]


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
        image_data = np.sum(image_data, 1)

    # return as 1dim array of audio data
    return image_data.flatten()


def ism(image: Image,
        use_noise: bool, noise_strength: float):
    # +
    # + ISM - inverse spectrogram method of image decoding into sound data
    # +

    image = image.convert("L")
    image_data = np.array(image)[::-1] / 255
    
    if use_noise:
        noise = np.abs(np.random.normal(
            0, 1, image_data.shape)) * noise_strength
        image_data *= 1 - noise

    return istft(image_data, scaling="psd")[1]


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
    modes = ("ISM", "PBP")
    directions = ("row", "column")
    sample_rate_range = range(512, 256001)
    image_scale_range = range(1, 11)
    
    default_config = JSONConfig(
        {
            "mode": "PBP",
            "channels": 1,
            "sample_rate_locked": True,
            "sample_rate": 44100,
            "image_scale": 1,
            "PBP": {
                "sum_rgb_components": False,
                "direction": "row"
            },
            "ISM": {
                "use_noise": True,
                "noise_strength": 0.5
            }
        }
    )
    
    os.chdir(APP)
    os.makedirs("in", exist_ok=True)
    os.makedirs("out", exist_ok=True)

    try:
        config = JSONConfig.load(CFG_NAME)
        if config.schema != default_config.schema:
            raise KeyError

    except (FileNotFoundError, KeyError, JSONConfig.FileCorruptedError):
        write(f"config.json does not exist or corrupted, create new? [y/n] ")
        yn = force_input(ffg('>>> ', FG.GREEN), "[y/n]",
                         func=str.lower,
                         predicates=[lambda x: x in ('y', 'n')])
        if yn == 'y':
            default_config.save(CFG_NAME)
        Scr.reset_mode()
        sys.exit(1)

    mode = config.get("mode")
    channels = config.get("channels")
    sample_rate_locked = config.get("sample_rate_locked")
    sample_rate = config.get("sample_rate")
    image_scale = config.get("image_scale")
    pbp_sum_rgb_components = config.section("PBP").get("sum_rgb_components")
    pbp_direction = config.section("PBP").get("direction")
    ism_use_noise = config.section("ISM").get("use_noise")
    ism_noise_strength = config.section("ISM").get("noise_strength")
    
    try:
        if mode not in modes:
            raise ValueError(
                f"modes available: [PBP, ISM]; current: {mode}")
        if channels not in (1, 2):
            raise ValueError(
                f"channels available: [1 - (Mono), 2 - (Stereo)]; current: {channels}")
        if pbp_direction not in directions:
            raise ValueError(
                f"directions available: [row, column]; current: {pbp_direction}")

        if sample_rate not in sample_rate_range:
            raise ValueError(
                f"sample rate must be in range({sample_rate_range[0]} ... {sample_rate_range[-1]}); current: {sample_rate}")
        if image_scale not in image_scale_range:
            raise ValueError(
                f"image scale must be in range({image_scale_range[0]} ... {image_scale_range[-1]}); current: {image_scale}")
        if not (0.0 <= ism_noise_strength <= 1.0):
            raise ValueError(
                f"noise strength must be a factor in range(0.0 ... 1.0); current: {ism_noise_strength}")

    except ValueError as e:
        write(f"Config validation failed: {ffg(f'[!] {e}', FG.RED)} \n")
        write("Fix it yourself or reset\n")
        write("Reset? [y/n] ")

        yn = force_input(ffg('>>> ', FG.GREEN), "[y/n]",
                         func=str.lower,
                         predicates=[lambda x: x in ('y', 'n')])
        if yn == 'y':
            default_config.save(CFG_NAME)

        Scr.reset_mode()
        sys.exit(1)

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

    write("Dynamic sample rate: ")
    if not sample_rate_locked:
        write(ffg("+", FG.GREEN))
    else:
        write(ffg("-", FG.RED))
        write()
        write(f"Static sample rate: {ffg(sample_rate, FG.CYAN)}")
    write()
        
    write("Channels: ")
    write(ffg(channels, FG.GREEN))
    write()
    
    write("Upscale: ")
    if image_scale != 1:
        write(ffg("+", FG.GREEN))
        write()
        write(f"Scale factor: {ffg(image_scale, FG.BLUE)}")
    else:
        write(ffg("-", FG.RED))
    write()
    
    if mode == "PBP":
        write("[+] PBP\n")
        write(f"\tdirection: {ffg(pbp_direction, FG.YEL)}\n")
        if pbp_sum_rgb_components:
            write(f"\tuse sum of {ffg('r', FG.RED)}+{ffg('g', FG.GREEN)}+{ffg('b', FG.BLUE)} components")
        else:
            write(f"\tuse {ffg('r', FG.RED)}{ffg('g', FG.GREEN)}{ffg('b', FG.BLUE)} components separately")
    elif mode == "ISM":
        write("[+] ISM\n")
        if ism_use_noise:
            write(f"\tapply noise with strength: {ffg(ism_noise_strength, FG.YEL)}")
        else:
            write("...")
    write()
                
    write()
        
    with Timer(fstyle("Total", STYLE.BOLD)):
        for file in img_files:
            write(f"{fstyle(':: filename:', STYLE.BOLD)} {ffg(file, FG.MAGNT)}\n")

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
                
                write("\t+ image resized:\n")
                write(f"\t├ scale: {ffg(image_scale, FG.BLUE)}\n")
                write(f"\t{f'└ original size:':<30} {ffg(orig_size[0], FG.BLUE)}x{ffg(orig_size[1], FG.BLUE)}\n")
                
            width, height = image.size
            image_lin_size = width * height
            
            write(f"\t{f'size:':<30} {ffg(width, FG.BLUE)}x{ffg(height, FG.BLUE)}\n")
            write(f"\t{f'pixels amount:':<30} {ffg(image_lin_size, FG.CYAN)}px\n")
                        
            if image_lin_size > MAX_IMAGE_LIN_SIZE:
                writef("\t:: Image is too large ::", F_ERROR)
                write()
                write(f"{ffg('>> SKIP', FG.RED)}\n")
                continue
            
            if not sample_rate_locked:
                sample_rate = round(clamped_linear_remap(image_lin_size,
                                                    1, MAX_IMAGE_LIN_SIZE,
                                                    sample_rate_range[0], sample_rate_range[-1]))
                write(f"\tsample rate: {ffg(sample_rate, FG.YEL)}\n")

            audio_data = None
            
            with Timer(fstyle(file, STYLE.BOLD)):
                if mode == "PBP":
                    write(ffg("( Processing )\n", FGRGB(64, 64, 64)))
                    
                    audio_data = pbp(image, image_lin_size,
                                     pbp_direction, pbp_sum_rgb_components)
                    Cur.prev_line()
                    Scr.clear_line()

                elif mode == "ISM":
                    if width < 64 or height < 64:
                        writef(
                            f"\t:: ISM mode requires width\height to be {ffg('64px+', FG.RED)} ::", F_ERROR)
                        write()
                        write(f"{ffg('>> SKIP', FG.RED)}\n")
                        continue

                    if width != height:
                        width = round(width * (height / width))
                        image = image.resize(
                            (width, height), resample=Image.LANCZOS
                        )
                        write(f"\t* aspect ratio {ffg('fixed', FG.GREEN)}\n")

                    write(ffg("( Processing )\n", FGRGB(64, 64, 64)))

                    audio_data = ism(image, ism_use_noise, ism_noise_strength)
                    
                    Cur.prev_line()
                    Scr.clear_line()

                if channels == 2:
                    if audio_data.shape[0] % 2 == 1:
                        audio_data = np.append(audio_data, 0)
                    audio_data = audio_data.reshape((len(audio_data)//2, 2))

                write(ffg("( Saving )\n", FGRGB(64, 64, 64)))
                
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