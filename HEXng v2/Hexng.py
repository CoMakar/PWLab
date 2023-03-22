# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 21:26:19 2020
@author: Qweyk
"""


import math
import os
import sys

import png
from tqdm import trange

from Common.force_input import force_input
from Common.Timer import Timer
from TermUtils.term import *


APP = os.path.dirname(sys.argv[0])
info_format = Format(fg=FG.BLUE)
error_format = Format(fg=FG.YEL, bg=BG.RED, style=STYLE.BOLD)
ok_format = Format(fg=FG.GREEN, style=STYLE.ITALIC)
inverted_format = Format(style=STYLE.REVERSE)


def main(): 
    os.chdir(APP)
    os.makedirs("in", exist_ok=True)
    os.makedirs("out", exist_ok=True)
    input_files = os.listdir('./in')
    
    if len(input_files) == 0:
        set_format(error_format)
        write("No input files!\n")
        input("...")
        Scr.reset_mode()
        sys.exit(1)
    
    write(f"number of columns ")
    columns = force_input(ffg(">>> ", FG.GREEN), 
                          ffg("Value should be larger than 100", FG.RED),
                          func=int,
                          predicates=[lambda x: x > 100])
    
    writef("< FILES >", inverted_format)
    write()
    [write(f"\t> {file}\n") for file in input_files]
    write()

    with Timer(fstyle("Total", STYLE.BOLD)):
        for file in input_files:
            with open(f'./in/{file}', 'rb') as file_bin:
                data_hex = file_bin.read().hex()
                
            # append missing bytes so we can form a pixel from bytes
            # every pixel requres 6 bytes (RRGGBB)
            missing_bytes = len(data_hex) % 6
            data_hex += "0" * (6 - missing_bytes)
            
            pixels_amount = len(data_hex) // 6    
            rows = math.ceil(pixels_amount / columns)
            
            # same action as above but now we must be sure that we can form a row of pixels
            missing_pixels = rows * columns - pixels_amount
            data_hex += "000000" * missing_pixels
            pixels_amount += missing_pixels
            triplets_amount = pixels_amount * 3
            
            write(f"{fstyle(':: filename:', STYLE.BOLD)} {ffg(file, FG.MAGNT)}\n")
            write(f"\tmissing_bytes: {ffg(missing_bytes, FG.RED)}\n")
            write(f"\trows x columns: {ffg(rows, FG.BLUE)}x{ffg(columns, FG.BLUE)}\n")
            write(f"\tapproximate pixels amount: {ffg(f'{pixels_amount}px', FG.GREEN)}\n")
            
            brush = png.Writer(columns, rows, greyscale=False)
                    
            with Timer(fstyle(file, STYLE.BOLD)):            
                canvas = [0] * triplets_amount
                for t in trange(triplets_amount, ascii=" -=#", desc=file,
                                bar_format='{desc}: [{bar:20}] {percentage:.0f}%'):
                    canvas[t] = int(data_hex[t*2 : t*2+2], 16) 
                    
                write("Saving...\n")
                    
                with open(f'./out/{file}.png', 'wb') as png_out:
                    brush.write_array(png_out, canvas)
        
    set_format(ok_format)
    write("Done!\n")
    input("...")
                
                      
if __name__ == "__main__":
    os.system("color")
    try:
        main()
    except KeyboardInterrupt:
        input("...")