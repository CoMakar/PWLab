# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 21:07:56 2020
@configor: Qwek
"""


import io
import math
import os
import sys
import traceback
from typing import Union

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
from scipy.io.wavfile import write as wav_write
from scipy.signal import istft
from scipy.signal.windows import cosine

from Common.force_input import force_input
from Common.JSONConfig import JSONConfig
from Common.Timer import Timer
from TermUtils.term import *


CFG_NAME = "config.json"
APP = os.path.dirname(sys.argv[0])
MAX_IMAGE_LIN_SIZE = 1e9
WINDOW_MODE = cosine
info_format = Format(fg=FG.BLUE)
error_format = Format(fg=FG.YEL, bg=BG.RED, style=STYLE.BOLD)
ok_format = Format(fg=FG.GREEN, style=STYLE.ITALIC)
inverted_format = Format(style=STYLE.REVERSE)


def pbp (image: Image, linear_size: int,
         direction: str, use_sum_rgb: bool):
    image = image.convert('RGB')
    image_data = np.array(image)

    if direction == "column": 
        # transpose image to replace rows with columns
        # [r, g]    =>  [r, b]
        # [b, b]    =>  [g, b]
        image_data = image_data.transpose((1, 0, 2))
    
    if use_sum_rgb:
        # sum rgb components into a single value
        image_data = image_data.reshape(linear_size, 3) 
        image_data =  np.sum(image_data, 1)
    
    return image_data.flatten()
    
    
def ism (image: Image, use_scanlines: bool,
         scanlines_distance: int):
    
    image = image.convert("L")
    blurred_image = image.filter(ImageFilter.GaussianBlur(4))
    edges = image.filter(ImageFilter.FIND_EDGES())
    image_data = np.array(image)[::-1] / 255
    blurred_data = np.array(blurred_image)[::-1] / 255
    edge_data = np.array(edges)[::-1] / 255
    
    edge_data[::2].fill(0)
    
    if use_scanlines:
        image_data[::scanlines_distance].fill(np.min(image_data))
        
    composed = image_data + (blurred_data * 8) + (edge_data / 16)
    
    return istft(composed, scaling="psd")[1]


def clamp(value: Union[int, float],
          min_value: Union[int, float] = 0, max_value: Union[int, float] = 1):
    return max(min(value, max_value), min_value)


def clamped_log_remap(value: Union[int, float],
                      from_min: Union[int, float], from_max: Union[int, float],
                      to_min: Union[int, float], to_max: Union[int, float]):
    
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


def main():
    modes = ("ISM", "PBP")
    directions = ("row", "column")
    sample_rate_range = range(512, 256001)
    scanlines_range = range(8, 65)
    scale_min = 1
    scale_max = 10
    
    config_standard = {
        "mode": "PBP",
        "channels": 1,
        "rate_locked": True,
        "sample_rate": 44100,
        "image_scale": 1,
        "PBP": {
            "pixel_sum_method": False,
            "direction": "row"
        },
        "ISM": {
            "use_scanlines": False,
            "scanlines_distance": 16,
        }
    }
    config_standard = JSONConfig(config_standard)
    
    os.chdir(APP)
    os.makedirs("in", exist_ok=True)
    os.makedirs("out", exist_ok=True)
    
    try:
        config = JSONConfig.load(CFG_NAME)
        if config.schema != config_standard.schema:
            raise KeyError
    except (FileNotFoundError, KeyError, JSONConfig.FileCorruptedError):
        write(f"config.json does not exist or corrupted, create new? [y/n] ")
        yn = force_input(ffg('>>> ', FG.GREEN), "[y/n]", 
                    func=str.lower,
                    predicates=[lambda x: x in ('y', 'n')])
        if yn == 'y':
            config_standard.save(CFG_NAME)
        Scr.reset_mode()
        sys.exit(1)
            

    mode = config.get("mode")
    channels = config.get("channels")
    rate_locked = config.get("rate_locked")
    sample_rate = config.get("sample_rate")
    image_scale = config.get("image_scale")
    pbp_use_sum_method = config.section("PBP").get("pixel_sum_method")
    pbp_direction = config.section("PBP").get("direction")
    ism_use_scanlines = config.section("ISM").get("use_scanlines")
    ism_scanlines_distance = config.section("ISM").get("scanlines_distance")
    
    try: 
        assert mode in modes,\
            f"modes available: [PBP, ISM]; current: {mode}"
        assert channels in (1, 2),\
            f"channels available: [1 (MONO), 2 (STEREO)]; current: {channels}"
        assert sample_rate in sample_rate_range,\
            f"samplerate not in {sample_rate_range}; current: {sample_rate}"
        assert image_scale >= scale_min and image_scale <= scale_max,\
            f"image scale not in ({scale_min} <= scale <= {scale_max}); current: {image_scale}"
        assert pbp_direction in directions,\
            f"pbp direction available: [row, column]; current: {pbp_direction}"
        assert ism_scanlines_distance == 0 or ism_scanlines_distance in scanlines_range,\
            f"scanlines distance not in {scanlines_range}; current: {ism_scanlines_distance}"
        
    except AssertionError as e:
        write(f"Config validation failed: {ffg(f'[!] {e}', FG.RED)} \n")
        
        write("Fix it yourself or reset")
        write("Reset? [y/n] ")
        
        yn = force_input(ffg('>>> ', FG.GREEN), "[y/n]", 
            func=str.lower,
            predicates=[lambda x: x in ('y', 'n')])
        if yn == 'y':
            config_standard.save(CFG_NAME)
        Scr.reset_mode()
        sys.exit(1)
        
    
    input_files = set(os.listdir('./in'))
    png_files = set(filter(lambda f: 
        f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg"), input_files))
    other_files = input_files - png_files
    
    if len(png_files) == 0:
        set_format(error_format)
        write("No input files!\n")
        input("...")
        Scr.reset_mode()
        sys.exit(1)
        
    writef("< FILES >", inverted_format)
    write()
    [write(f"\t> {file}\n") for file in png_files]
    [write(ffg(f"\t- {file}\n", FG.RED)) for file in other_files]
    
    write()
    
    write("Mode: ")
    write(fstyle(ffg(mode, FG.YEL), STYLE.UNDER))
    write()

    write("Dynamic sample rate: ")
    if not rate_locked:
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
        write(f"   direction: {ffg(pbp_direction, FG.YEL)}\n")
        if pbp_use_sum_method:
            write(f"   use {ffg('r', FG.RED)}+{ffg('g', FG.GREEN)}+{ffg('b', FG.BLUE)} value")
        else:
            write(f"   use {ffg('r', FG.RED)}{ffg('g', FG.GREEN)}{ffg('b', FG.BLUE)} components")
    elif mode == "ISM":
        write("[+] ISM\n")
        if ism_use_scanlines:
            write(f"   draw scanline every {ism_scanlines_distance} row")
        else:
            write("   no active setting ;(")
    write()
                
    write()
        
    with Timer(fstyle("Total", STYLE.BOLD)):
        for file in png_files:
            image: Image = None
            try:
                with open(f"./in/{file}", 'rb') as file_img:
                    image = Image.open(io.BytesIO(file_img.read()))
            except Exception as e:
                writef(f"File error: {e}\n", error_format)
                writef(traceback.format_exc, error_format)
                write("skip ->\n")
                continue
            
            write(f"{fstyle(':: filename:', STYLE.BOLD)} {ffg(file, FG.MAGNT)}\n")
            
            orig_size = image.size
                
            if image_scale != 1:
                image = image.resize((round(image.size[0]*image_scale),
                                    round(image.size[1]*image_scale)), 
                                    resample=Image.NEAREST)
                writef(f"\timage resized [scale: {image_scale}]\n", info_format)
                
            width, height = image.size
            image_lin_size = width * height
            
            write(f"\timage size: {ffg(width, FG.BLUE)}x{ffg(height, FG.BLUE)}\n")
            if image_scale != 1:
                write(f"\toriginal size: {ffg(orig_size[0], FG.BLUE)}x{ffg(orig_size[1], FG.BLUE)}\n")
            write(f"\tapproximate pixels amount: {ffg(image_lin_size, FG.CYAN)}\n")
            
            if image_lin_size > MAX_IMAGE_LIN_SIZE:
                writef("\tImage is too large\n", error_format)
                write("\tskip ->\n")
                continue
            
            if not rate_locked:
                sample_rate = round(clamped_log_remap(image_lin_size,
                                                    1, MAX_IMAGE_LIN_SIZE,
                                                    sample_rate_range[0], sample_rate_range[-1]))
                write(f"\tsample rate: {ffg(sample_rate, FG.YEL)}\n")

            data = None
            
            with Timer(fstyle(file, STYLE.BOLD)):
                if mode == "PBP":
                    data = pbp(image, image_lin_size, 
                            pbp_direction, pbp_use_sum_method)
                
                elif mode == "ISM":            
                    if width < 64 or height < 64:
                        writef(f"\tsize is too small for ISM (64+x64+ px required)\n", error_format)
                        write("\tskip ->\n")
                        continue
                    
                    if width != height:
                        writef("\taspect ratio fixed\n", info_format)
                        width = round(width * (height / width))
                        image = image.resize((width, height),resample=Image.LANCZOS)
                    
                    data = ism(image, ism_use_scanlines, ism_scanlines_distance)
                
                if channels == 2:
                    if data.shape[0] % 2 == 1:
                        data = np.append(data, 0)
                    data = data.reshape((len(data)//2, 2))
                
                write("Saving...\n")
                
                wav_write(f"./out/{file}_{mode}_{sample_rate}HZ.wav", sample_rate, data)
            
    set_format(ok_format)
    write("Done!\n")
    input("...")
    
if __name__ == '__main__':
    os.system("color")
    try:
        main()
    except KeyboardInterrupt:
        input("...")
    Scr.reset_mode()