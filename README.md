# Version 1.0.4

## HEXng

![HEXNG_EXAMPLE](Examples/Hexng_1.png)

- Converts _any file_ into _.png_
- Options source: keyboard

## Wavorizer

![WAVORIZER_EXAMPLE](Examples/Wavorizer_1.png)

- Converts _.png, .jpeg, .jpg_ (further .{image}) into _.wav_ in 2 different modes:
1.  PBP [ Pixel by pixel ]
    - Reads RGB components and writes them as .wav amplitudes
    - Options: 
      
option             | arg  | description
-------------------|------|------------
sum_rgb_components | bool | uses sum of R+G+B instead of separate RGB components
direction          | str  | **rows** - reads .{image} row by row; **cols** - reads .{image} column by column 

1. ISM [ Inverse spectrogram method ]
    - Reads entire .{image} as spectrogram and writes it as _.wav_ amplitudes
    - Must be suitable for many images, but very small and messy images will produce messy result
    - Options: 

option         | arg  | description
---------------|------|------------
use_noise      | bool | whether noise should be applied to the image or not
noise_strength | str  | strength of the applied noise
detect_edges   | bool | use the sobel function to detect the edges of objects
 
- Other options:

option             | arg  | description
-------------------|------|------------
sample_rate_mode   | str  | if **static**, `sample_rate` value will be ues, if **dynamic** sample rate will be computed automatically using linear remapping
sample_rate        | int  | target sample rate of the output .wav file
channels           | int  | mono (1) or stereo (2)
image_scale        | int  | if this value is other than 1, then image will be prescaled using this scale factor

- Options source: config file

> **Note:**
> Current max image size is 1e8 ( =  _1 000 000 00_ )
> .{image} with **width** x **height** > 1e8 will be skipped.
> This value is also used for remapping as max value of initial range.
> Look at the sources if you really want to edit it:
> Wavorizer.py :: MAX_IMAGE_LIN_SIZE = 1e8

## Complete config structure ( Default values ):
``` JSON
{
    "mode": "PBP",
    "channels": 1,
    "sample_rate_mode": "static",
    "sample_rate": 44100,
    "image_scale": 1,
    "PBP": {
        "sum_rgb_components": false,
        "direction": "rows"
    },
    "ISM": {
        "use_noise": true,
        "noise_strength": 0.5,
        "detect_edges": false
    }
}
```

## Examples

**HEXng**: `Notepad++.exe [ column: 1000 ]`
<img src="Examples/notepad++.exe.png" style="border-radius: 32px"> 

**Wavorizer**: `Logo [ scanlines: 12 ] # Scanlines are deprecated`
<img src="Examples/Logo_Both.png" style="border-radius: 32px"> 

**Wavorizer**: `2B2T Spawn`
<img src="Examples/2B2T.png" style="border-radius: 32px"> 
