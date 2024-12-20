import os
import platform
from typing import Union, List, Iterable, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from sys import stdout
from time import sleep
from threading import Thread, Lock


#SECTION - Colors
class FG(Enum):
    """
    [FG] 8 classic colors enum
    """
    BLACK   = "\u001b[30m"
    RED     = "\u001b[31m"
    GREEN   = "\u001b[32m"
    YEL     = "\u001b[33m"
    BLUE    = "\u001b[34m"
    MAGNT   = "\u001b[35m"
    CYAN    = "\u001b[36m"
    WHITE   = "\u001b[37m"
    DEF     = "\u001b[39m"


class FGRGB:
    def __init__(self, r: int, g: int, b: int):
        """
        [FG] constructs color formatting string rom RGB values
        """
        if not (0 <= r < 256 or 0 <= g < 256 or 0 <= b < 256):
            raise ValueError("Invalid RGB values")
        self.value = f"\u001b[38;2;{r};{g};{b}m"


class BG(Enum):
    """
    [BG] 8 classic colors enum
    """
    BLACK   = "\u001b[40m"
    RED     = "\u001b[41m"
    GREEN   = "\u001b[42m"
    YEL     = "\u001b[43m"
    BLUE    = "\u001b[44m"
    MAGNT   = "\u001b[45m"
    CYAN    = "\u001b[46m"
    WHITE   = "\u001b[47m"
    DEF     = "\u001b[49m"


class BGRGB:
    def __init__(self, r: int, g: int, b: int):
        """
        [BG] constructs color formatting string rom RGB values
        """
        if not (0 <= r < 256 or 0 <= g < 256 or 0 <= b < 256):
            raise ValueError("Invalid RGB values")
        self.value = f"\u001b[48;2;{r};{g};{b}m"
#---------------------------------------------------------------------------
#!SECTION


#SECTION - Style
class STYLE(Enum):
    RESET   = "\u001b[22m\u001b[23m\u001b[24m\u001b[25m\u001b[27m\u001b[28m\u001b[29m"
    # resets all active styles
    BOLD    = "\u001b[1m"
    ITALIC  = "\u001b[3m"
    UNDER   = "\u001b[4m"
    BLNK    = "\u001b[5m"
    REVERSE = "\u001b[7m"
    HIDDEN  = "\u001b[8m"
#---------------------------------------------------------------------------
#!SECTION


#SECTION - Screen
class Scr:
    @staticmethod
    def clear_os():
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def color_on():
        if platform.system() == "Windows":
            os.system("color")
        # FIXME: Unexpected behavior on other platforms
    
    @staticmethod
    def clear():
        iwrite("\u001b[2J\u001b[0;0H")

    @staticmethod
    def clear_line():
        iwrite("\u001b[2K")

    @staticmethod
    def reset_mode():
        # resets all active styles and colors
        iwrite("\u001b[0m")
    
    @staticmethod
    def maxx() -> int:
        return os.get_terminal_size().columns
    
    @staticmethod
    def maxy() -> int:
        return os.get_terminal_size().lines
    
    @staticmethod
    def midx() -> int:
        return os.get_terminal_size().columns // 2
    
    @staticmethod
    def midy() -> int:
        return os.get_terminal_size().lines // 2
#---------------------------------------------------------------------------
#!SECTION


#SECTION - Cursor
class Cur:
    @staticmethod
    def up(n: int = 1):
        iwrite(f"\u001b[{n}A")

    @staticmethod
    def down(n: int = 1):
        iwrite(f"\u001b[{n}B")

    @staticmethod
    def left(n: int = 1):
        iwrite(f"\u001b[{n}D")

    @staticmethod
    def right(n: int = 1):
        iwrite(f"\u001b[{n}C")

    @staticmethod
    def prev_line(n: int = 1):
        iwrite(f"\u001b[{n}F")

    @staticmethod
    def next_line(n: int = 1):
        iwrite(f"\u001b[{n}E")

    @staticmethod
    def to(line: int, col: int):
        iwrite(f"\u001b[{line};{col}H")
        
    @staticmethod
    def to_col(col: int):
        iwrite(f"\u001b[{col}G")

    @staticmethod
    def home():
        iwrite("\u001b[H")

    @staticmethod
    def hide():
        iwrite("\u001b[?25l")

    @staticmethod
    def show():
        iwrite("\u001b[?25h")

    @staticmethod
    def pos_save():
        """
        WARNING: 
            
        Don't rely on this function while saving/loading position for non-atomic operations
        (especially for multiple threads)
        for long term storage use list/tuple etc. and Cur.to()
        """
        iwrite("\u001b[s")

    @staticmethod
    def pos_restore():
        """
        WARNING: 
        
        don't rely on this function while saving/loading position for non-atomic operations
        (especially for multiple threads)
        for long term storage use list/tuple/etc. and Cur.to()
        """
        iwrite("\u001b[u")
        
    @staticmethod
    def lf(n: int = 1):
        iwrite("\n" * n)
#---------------------------------------------------------------------------
#!SECTION


#---------------------------------------------------------------------------
#SECTION - Sprites and Animations
class Sprite:    
    def __init__(self, sprite_data: Union[List[str], str]):
        if not isinstance(sprite_data, list) and not isinstance(sprite_data, str):
            raise TypeError("Sprite data must be a List[str] or str")

        if isinstance(sprite_data, list):
            
            if len(sprite_data) == 0:
                raise ValueError("Zero sized sprite")
            
            for row in sprite_data:
                if not isinstance(row, str):
                    raise TypeError("Sprite data must be a List[str] or str")
                
            height = 0
            width_values = []
            for row in sprite_data:
                height += 1
                width_values.append(len(row))
                
            is_width_fixed = len(set(width_values)) == 1
            if not is_width_fixed:
                raise ValueError("Non-constant width, width must be constant for each row")
            width = width_values[0]
        else:
            width = len(sprite_data)
            height = 1
            sprite_data = [sprite_data]
            
        if width < 1:
            raise ValueError("Sprite has zero width")
            
        self._pos = 0, 0
        self._data = sprite_data
        self._sizes = (width, height)
        
    def set_pos(self, x: int, y: int):
        if x < 0 or y < 0:
            raise ValueError("Invalid position")
        self._pos = (x, y)
        
    def get_pos(self) -> Tuple[int, int]:
        """
        Returns: 
            (x, y)
        """
        return self._pos
    
    def get_width(self) -> int:
        return self._sizes[0]
    
    def get_height(self) -> int:
        return self._sizes[1]
    
    def draw(self):
        Cur.to(*self._pos)
        for row in self._data:
            Cur.pos_save()
            stdout.write(row)
            Cur.pos_restore()
            Cur.down()
        stdout.flush()
                        
                
class Animation:
    
    _lock = Lock()
    
    def __init__(self, frames: List[Sprite], frame_duration_ms: int, repeat: int):
        """
        Simple ascii graphics animation
        Args:
            @param: frames (List[List[str]]): List of animation frames. Each frame is a list of strings.
            see Example below,
            
            frame_duration_ms (int): duration of each frame in milliseconds.
            repeat (int): how many times animation should be played
            default position is set to (0, 0)
                use set_pos(x, y) to change it
        """
        
        """
            Example:
            frames =
               [[               [               [                                        
                 "   ",          "   ",          " o ",                                                                
                 " o ",   ->     "   ",   ->     "   ",                                               
                 "   "           " o ",          "   ",                                           
                      ],              ],              ]]                                        
        """
                
        if not isinstance(frames, list):
            raise TypeError("Frames must be List[Sprite]")

        for frame in frames:
            if not isinstance(frame, Sprite):
                raise TypeError("Frames must be a List[Sprite]") 
            
        height_values = []
        width_values = []
        for frame in frames:
            height_values.append(frame.get_height())
            width_values.append(frame.get_width())

        is_height_fixed = len(set(height_values)) == 1
        is_width_fixed = len(set(width_values)) == 1
        if not is_width_fixed or not is_height_fixed:
            raise ValueError("All frames must be of the same fixed size")
        
        width = width_values[0]
        height = height_values[0]
        
        if frame_duration_ms <= 0:
            raise ValueError("Frame duration cannot be 0 or smaller")
        
        if repeat <= 0:
            raise ValueError("Animation must be played at least once")
        
        self._frames      = frames
        self._repeat      = repeat
        self._duration    = frame_duration_ms
        self._pos         = (0, 0)
        self._sizes       = (width, height)
        self._empty_frame = Sprite([" " * width_values[0] for _ in range(height_values[0])])
        
    def set_pos(self, x: int, y: int):
        if x < 0 or y < 0:
            raise ValueError("Invalid position")
        for frame in self._frames:
            frame.set_pos(x, y)
        self._pos = (x, y)
        
    def get_pos(self) -> Tuple[int, int]:
        """
        Returns: 
            (x, y)
        """
        return self._pos

    def get_width(self) -> int:
        return self._sizes[0]
    
    def get_height(self) -> int:
        return self._sizes[1]
    
    def set_duration(self, ms: int):
        if ms <= 0:
            raise ValueError("Frame duration cannot be 0 or smaller")
        self._duration = ms
        
    def set_repeat(self, n: int):
        if n <= 0:
            raise ValueError("Animation must be played at least once")
        self._repeat = n

    def _draw_frame(self, frame: Sprite):
        with Animation._lock:
            frame.draw()

    def _play(self, clear_after: bool):
        for t in range(self._repeat):
            for frame in self._frames:
                self._draw_frame(frame)
                sleep(self._duration/1000)
        if clear_after:
            self._draw_frame(self._empty_frame)

    def play(self, clear_last_frame: bool = True) -> Thread:
        """
        Create a new thread to play animation, start it and return renderer thread;
        to wait for the animation to finish, call <animation_thread>.join()

        Args:
            clear_last_frame (bool, optional): erase last rendered frame or not. Defaults to True.

        Returns:
            Thread: renderer thread
        """
        animation_thread = Thread(target=self._play, args=(clear_last_frame,))
        animation_thread.start()
        return animation_thread
#---------------------------------------------------------------------------
#!SECTION


#SECTION - Color/Style function
def val(enum_element: Union[FG, FGRGB, BG, BGRGB, BG, BGRGB, STYLE]):
    """
    returns [FG, FGRGB, BG, BGRGB, BG, BGRGB, STYLE] value string
    """
    return enum_element.value


def ffg(text: Any, fg: Union[FG, FGRGB]):
    """
    returns [FG, FGRGB] formatted text
    """
    return f"{val(fg)}{text}{val(FG.DEF)}"


def fbg(text: Any, bg: Union[BG, BGRGB]):
    """
    returns [BG, FGRGB] formatted text
    """
    return f"{val(bg)}{text}{val(BG.DEF)}"


def fstyle(text: Any, style: STYLE):
    """
    returns [STYLE] formatted text
    """
    return f"{val(style)}{text}{val(STYLE.RESET)}"


@dataclass
class Format:
    """
    Used for style consistency
    and creating set of commonly used styles
    """
    fg: Union[FG, FGRGB]      = FG.DEF
    bg: Union[BG, BGRGB]      = BG.DEF
    style: Union[STYLE, None] = None
    

def set_color(fg: Union[FG, FGRGB], bg: Union[BG, BGRGB] = BG.DEF):
    iwrite(fg.value)
    iwrite(bg.value)


def set_style(style: STYLE = STYLE.RESET):
    iwrite(style.value)


def set_format(text_format: Format):
    if text_format.style:
        set_style(text_format.style)
    
    set_color(text_format.fg, text_format.bg)
#---------------------------------------------------------------------------
#!SECTION


#SECTION - Write functions
def iwrite(text: Any = "\n"):
    """
    write with immediate flush
    """
    text = str(text)
    stdout.write(text)
    stdout.flush()
 
    
def write(text: Any = "\n"):
    """
    Write without flush (this function just a sugar)
    """
    text = str(text)
    stdout.write(text)
    

def writef(text: Any, text_format: Format):
    """
    Write formatted text; automatically resets all styles and colors after writing
    (Formatted means not a Python f-string but a formatted terminal)
    """
    set_format(text_format)
    write(text)
    Scr.reset_mode()
    

def writew(text: Iterable = "\n", wait: float = 0.5, sep: str = ""):
    """
    Write text and sleep between printing chars
    """
    for char in text:
        iwrite(f"{char}{sep}")
        sleep(wait)
        

def writebox(text: str, x0: int, y0: int, x1: int, y1: int):
    """
    Write text inside bounding box; overflow is HIDDEN
    
    Args:
        text (str): text to be printed
        x0 (int): top left corner x
        y0 (int): top left corner y
        x1 (int): bottom right corner x
        y1 (int): bottom right corner y
        
    Raises:
        ValueError: if box is too small
    """
    line_len = x1 - x0 + 1
    box_height = y1 - y0 + 1
    
    if box_height < 1 or line_len < 1:
        raise ValueError("box is too small")
        
    text = text.ljust(line_len * box_height, " ")
    # Fills the text with space to maintain formatting
    chunks = [text[i: i + line_len] for i in range(0, len(text), line_len)]
    Cur.to(y0, x0)
    for line, chunk in enumerate(chunks, 1):
        Cur.pos_save()
        write(chunk)
        Cur.pos_restore()
        Cur.down()
        if line > box_height:
            return
#---------------------------------------------------------------------------
#!SECTION
 
 
#SECTION - Draw functions
def drawline(char: str, x0: int, y0: int, x1: int, y1: int):
    """
    Bresenham's line algorithm
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1 + 1:
            Cur.to(y, x)
            write(char)
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1 + 1:
            Cur.to(y, x)
            write(char)
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy        
    
    
def drawbox(x0: int, y0: int, x1: int, y1: int):
    """
    Args:
        x0 (int): top left x
        y0 (int): top left y
        x1 (int): bottom right x
        y1 (int): bottom right y

    Raises:
        ValueError: if width/height is smaller than 1
                    or top left coordinates are out of bounds

    Returns:
        center: relative coordinates of the center of the box (y, x)
    """
    corners = "...."
    hor = "-"
    ver = "|"
    
    width = x1 - x0
    height = y1 - y0
    
    if width < 1 or height < 1:
        raise ValueError("Inappropriate size")
    
    if x0 < 1 or y0 < 1:
        raise ValueError("Out of bounds")
    
    center = (height // 2, width // 2)
    
    Cur.to(y0, x0)
    drawline(hor, x0, y0, x1, y0)
    drawline(ver, x1, y0, x1, y1)
    drawline(hor, x0, y1, x1, y1)
    drawline(ver, x0, y0, x0, y1)
    Cur.to(y0, x0)
    write(corners[0])
    Cur.to(y0, x1)
    write(corners[1])
    Cur.to(y1, x0)
    write(corners[2])
    Cur.to(y1, x1)
    write(corners[3])
    
    return center


def textbox(text: str, x0: int, y0: int, x1: int, y1: int):
    """
    write text inside bounding box, draw box around it
    """
    drawbox(x0, y0, x1, y1)
    writebox(text, x0+1, y0+1, x1-1, y1-1)
#---------------------------------------------------------------------------
#!SECTION
