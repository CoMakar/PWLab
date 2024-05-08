# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 21:26:19 2020
@author: Qweyk
"""


# +
# + [!] needs refactoring. At the moment it's just a mishmash of code
# + 


import math
import os
import sys

import png
from tqdm import trange

from src.Common.force_input import force_input
from src.Common.timer import Timer
from src.TermUtils import *


APP = os.path.dirname(sys.argv[0])
F_INFO = Format(fg=FG.BLUE)
F_ERROR = Format(fg=FG.YEL, bg=BG.RED, style=STYLE.BOLD)
F_OK = Format(fg=FG.GREEN, style=STYLE.ITALIC)
F_INVERTED = Format(style=STYLE.REVERSE)


def main(): 
    os.chdir(APP)
    os.makedirs("in", exist_ok=True)
    os.makedirs("out", exist_ok=True)
    input_files = os.listdir('./in')
    
    if len(input_files) == 0:
        writef(":: No input files! ::", F_ERROR)
        write()
        Scr.reset_mode()
        sys.exit(1)
    
    write(f"{ffg('?', FG.GREEN)} Number of columns ")
    columns = force_input(ffg(">>> ", FG.GREEN), 
                          ffg("[ Value should be larger than 100 ]", FGRGB(64, 64, 64)),
                          func=int,
                          predicates=[lambda x: x > 100])
    Scr.clear_os()
    
    writef("< FILES >", F_INVERTED)
    write()
    [write(f"\t> {file}\n") for file in input_files]
    write()
        
    # +
    # + [!] not a RAM friendly way, but pretty straightforward
    # +     

    with Timer(fstyle("Total", STYLE.BOLD)):
        for file in input_files:
            with open(f'./in/{file}', 'rb') as file_bin:
                data_hex = file_bin.read().hex()
            
            # append missing bytes so it can form a pixel from bytes
            # every pixel requires 6 bytes - (0x__ 0x__ 0x__)
            missing_bytes = len(data_hex) % 6
            data_hex += "0" * (6 - missing_bytes)
            
            pixels_amount = len(data_hex) // 6    
            rows = math.ceil(pixels_amount / columns)
            
            # append missing pixels so it can form a pixel from
            missing_pixels = rows * columns - pixels_amount
            data_hex += "000000" * missing_pixels
            pixels_amount += missing_pixels
            triplets_amount = pixels_amount * 3
            
            write(f"{fstyle(':: filename:', STYLE.BOLD)} {ffg(file, FG.MAGNT)}\n")
            write(f"\tmissing_bytes: {ffg(missing_bytes, FG.RED)}\n")
            write(f"\tsize: {ffg(rows, FG.BLUE)}x{ffg(columns, FG.BLUE)}\n")
            write(f"\tpixels amount: {ffg(f'{pixels_amount}px', FG.GREEN)}\n")
            
            brush = png.Writer(columns, rows, greyscale=False)
                    
            with Timer(fstyle(file, STYLE.BOLD)):            
                canvas = [0] * triplets_amount
                for t in trange(triplets_amount, ascii=" -=#", desc=file,
                                bar_format='{desc}: [{bar:10}] {percentage:.0f}%'):
                    canvas[t] = int(data_hex[t*2 : t*2+2], 16) 
                
                Cur.prev_line()
                Scr.clear_line()
                    
                write(ffg("( Saving )\n", FGRGB(64, 64, 64)))
                    
                with open(f'./out/{file}.png', 'wb') as png_out:
                    brush.write_array(png_out, canvas)
                    
                Cur.prev_line()
                Scr.clear_line()
            
            write()
        
    writef("( Done! )\n", F_OK)
                
                      
if __name__ == "__main__":
    os.system("color") #! NOT TESTED ON LINUX
    try:
        main()
    except KeyboardInterrupt:
        input("Press Enter to exit...")