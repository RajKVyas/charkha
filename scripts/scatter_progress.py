import os
import json

data_dir = r"D:\llm\data"
prog_path = os.path.join(data_dir, "progress.json")

if os.path.exists(prog_path):
    with open(prog_path, "r", encoding="utf-8") as f:
        prog = json.load(f)

    for sid, count in prog.items():
        # Short name for the source
        short_name = sid.replace("/", "_").replace("-", "_")
        folder_name = f"{short_name}-dd"
        target_dir = os.path.join(data_dir, folder_name)

        os.makedirs(target_dir, exist_ok=True)
        target_prog = os.path.join(target_dir, "progress.json")

        # Write only the progress for this specific source
        with open(target_prog, "w", encoding="utf-8") as f_out:
            json.dump({sid: count}, f_out, indent=2)

    print(f"Scattered progress for {len(prog)} sources into their isolated *-dd folders!")
else:
    print("No global progress.json found.")
