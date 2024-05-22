from tools.video_info import VideoInfo
from pathlib import Path

from icecream import ic

# Constants
# ---------
FG_RED = '\033[31m'
FG_GREEN = '\033[32m'
FG_YELLOW = '\033[33m'
FG_BLUE = '\033[34m'
FG_WHITE = '\033[37m'
FG_BOLD = '\033[01m'
FG_RESET = '\033[0m'

# Funciones de color
# ------------------
def bold(text: str) -> str:
    return f"{FG_BOLD}{text}{FG_RESET}" if text is not None else ''

def red(text: str) -> str:
    return f"{FG_RED}{text}{FG_RESET}" if text is not None else ''

def green(text: str) -> str:
    return f"{FG_GREEN}{text}{FG_RESET}" if text is not None else ''

def yellow(text: str) -> str:
    return f"{FG_YELLOW}{text}{FG_RESET}" if text is not None else ''

def blue(text: str) -> str:
    return f"{FG_BLUE}{text}{FG_RESET}" if text is not None else ''

def white(text: str) -> str:
    return f"{FG_WHITE}{text}{FG_RESET}" if text is not None else ''

# Funciones
# ---------
def print_video_info(source: str, video_info: VideoInfo):
    if source.isnumeric():
        source_name = "Webcam"
    elif source.lower().startswith('rtsp://'):
        source_name = "RSTP Stream"
    else:
        source_name = Path(source).name
    
    text_length = 20 + max(len(source_name) , len(f"{video_info.width} x {video_info.height}"))
    
    # Print video information
    print(f"\n{green('*'*text_length)}")
    print(f"{blue('Información del Origen'):^{text_length+9}}")
    print(f"{bold('Origen'):<29}{source_name}")
    print(f"{bold('Tamaño'):<29}{video_info.width} x {video_info.height}")
    print(f"{bold('Cuadros Totales'):<29}{video_info.total_frames}") if video_info.total_frames is not None else None
    print(f"{bold('Cuadros Por Segundo'):<29}{video_info.fps:.2f}")
    print(f"\n{green('*'*text_length)}\n")
    

def print_progress(frame_number: int, total_frames: int):
    if total_frames is not None:
        percentage = f"[ {frame_number/total_frames:6.1%} ] "
        frame_progress = f"{frame_number} / {total_frames}"
        percentage_title = f"{'':11}"
    else:
        percentage = ''
        frame_progress = f"{frame_number}"
        percentage_title = ''
    
    frame_text_length = (2 * len(str(total_frames))) + 3
    if frame_number == 0:
        print(f"\n{percentage_title}{bold('Frame'):>{frame_text_length+9}}")
    print(f"\r{green(percentage)}{frame_progress:>{frame_text_length}}  ", end="", flush=True)
    

def print_times(frame_number: int, total_frames: int, progress_times: dict):
    capture_time = progress_times['capture_time']
    inference_time = progress_times['inference_time']
    frame_time = progress_times['frame_time']
    
    if total_frames is not None:
        percentage = f"[ {frame_number/total_frames:6.1%} ] "
        frame_progress = f"{frame_number} / {total_frames}"
        percentage_title = f"{'':11}"
    else:
        percentage = ''
        frame_progress = f"{frame_number}"
        percentage_title = ''
        
    frame_text_length = (2 * len(str(total_frames))) + 3
    if frame_number == 0:
        print(f"\n{percentage_title}{bold('Frame'):>{frame_text_length+9}}{bold('Capture'):>22}{bold('Inference'):>22}{bold('Total'):>22}")
    print(f"\r{green(percentage)}{frame_progress:>{frame_text_length}}  {1000*(capture_time):8.2f} ms  {1000*(inference_time):8.2f} ms  {1000*(frame_time):8.2f} ms  ", end="", flush=True)
    

def step_message(step: str = None, message: str = None):
    step_text = green(f"[{step}]") if step != "Error" else red(f"[{step}]")
    print(f"{step_text} {message}")
