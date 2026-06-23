import shutil
import os

print("FFmpeg executable location:", shutil.which("ffmpeg"))
print("Current DLL Load Paths:")
if hasattr(os, "get_dll_directory"):
    print(os.get_dll_directory())