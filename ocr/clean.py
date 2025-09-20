from pathlib import Path 
import shutil 

input_folder = Path('./qwen')

for _dir in input_folder.iterdir():
    for _txt in _dir.glob("*.txt"):
        with open(_txt, "r", encoding="utf-8") as fi:
            data = fi.readlines()
            if not data:
                _txt.unlink()