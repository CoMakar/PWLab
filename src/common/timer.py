from time import perf_counter
from datetime import timedelta


class Timer:
    """
    Class for measuring code execution time
    """
    
    def __init__(self, name=None):
        self.__tic = None
        self.__toc = 0.0
        self.__name = name

    def __enter__(self):
        self.tic()
        return self

    def __exit__(self, type, value, traceback):
        return self.toc()
        
    def tic(self) -> "Timer":
        """
        Sets the time to measure from. Resets toc
    
        Returns:
            Initial Timer instance
        """
        self.__tic = perf_counter()
        self.__toc = 0.0
        
        return self
        
    def toc(self) -> float:
        """
        Many toc() functions can be called for one tic()
        If tic() has not been called before, returns 0.0
        
        Returns:
            Time elapsed since last tic() called [seconds]
        """
        if self.__tic is not None:
            self.__toc = perf_counter() - self.__tic
            return self.__toc
        else:
            return self.__toc
        
    def prev_toc(self) -> float:
        """
        Simply returns the previous toc() call result
        If tic() or toc() has not been called before, returns 0.0
        
        Returns:
            Previous toc() result [seconds]
        """
        return self.__toc
    
    @staticmethod
    def sec_to_timedelta(seconds: float) -> timedelta:
        return timedelta(seconds=seconds)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}("{self.__name}")'
        