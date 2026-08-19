import os
import re
import tifffile
import numpy as np
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from collections import defaultdict

# ----------------------------- Extract All Readable Strings -----------------------------
def extract_all_ascii_strings(file_path, min_length=1):
    with open(file_path, 'rb') as f:
        data = f.read()
    pattern = rb'[\x20-\x7E]{%d,}' % min_length  # Printable ASCII only
    found = re.findall(pattern, data)
    strings_out = [s.decode('utf-8', errors='replace') for s in found]
    return "\n".join(strings_out)

# ----------------------------- Parse "retrace : true/false" from 32851 -----------------------------
def parse_retrace_value(meta):
    try:
        lines = meta.strip().splitlines()
        for line in lines:
            if "retrace" in line.lower():
                key, value = line.split(":", 1)
                if key.strip().lower() == "retrace":
                    return value.strip().lower() == "true"
    except Exception:
        pass
    return None  # fallback if unknown

# ----------------------------- GUI to Select Folder -----------------------------
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory(title="Select Folder Containing .jpk Files")
if not folder_path:
    raise SystemExit("No folder selected. Exiting.")
folder_path = Path(folder_path)

# ----------------------------- Process Each JPK File -----------------------------
for filename in sorted(os.listdir(folder_path)):
    if not filename.lower().endswith(".jpk"):
        continue

    jpk_path = folder_path / filename
    print(f"\nProcessing: {filename}")

    try:
        readable_metadata = extract_all_ascii_strings(jpk_path)

        # --- Group pages by channel name ---
        grouped_pages = defaultdict(list)
        with tifffile.TiffFile(jpk_path) as tif:
            for page in tif.pages:
                channel_name = page.tags.get(32850).value.strip() if 32850 in page.tags else "Unknown"
                grouped_pages[channel_name].append(page)

        # --- Assign trace/retrace roles per group ---
        for channel_name, pages in grouped_pages.items():
            if len(pages) == 1:
                page_roles = [("trace", pages[0])]
            elif len(pages) == 2:
                meta0 = pages[0].tags.get(32851).value if 32851 in pages[0].tags else ""
                meta1 = pages[1].tags.get(32851).value if 32851 in pages[1].tags else ""
                retrace0 = parse_retrace_value(meta0)
                retrace1 = parse_retrace_value(meta1)

                if retrace0 and not retrace1:
                    page_roles = [("retrace", pages[0]), ("trace", pages[1])]
                elif retrace1 and not retrace0:
                    page_roles = [("trace", pages[0]), ("retrace", pages[1])]
                else:
                    page_roles = [("trace", pages[0]), ("retrace", pages[1])]
            else:
                page_roles = [(f"trace{i+1}", p) for i, p in enumerate(pages)]

            # --- Save each page with correct label ---
            for role, page in page_roles:
                tags = page.tags
                suffix = role.replace(" ", "_")
                folder_name = f"TIFF_{channel_name}_{suffix}".replace(" ", "_").replace("(", "").replace(")", "")
                output_root = folder_path / folder_name
                image_folder = output_root / "images"
                metadata_folder = output_root / "metadata"
                image_folder.mkdir(parents=True, exist_ok=True)
                metadata_folder.mkdir(parents=True, exist_ok=True)

                out_base = f"{jpk_path.stem}_{channel_name}_{suffix}".replace(" ", "_").replace("(", "").replace(")", "")
                tif_path = image_folder / f"{out_base}.tif"
                txt_path = metadata_folder / f"{out_base}_metadata.txt"

                image_data = page.asarray()

                if page.photometric == 3 and hasattr(page, "colormap"):
                    palette = page.colormap
                    if palette.dtype == np.uint16:
                        palette = (palette / 256).astype(np.uint8)
                    rgb = np.zeros((image_data.shape[0], image_data.shape[1], 3), dtype=np.uint8)
                    for j in range(3):
                        rgb[..., j] = palette[j, image_data]
                    image_data = rgb

                # Full TIFF tag dump (all tags, even numeric/binary)
                tiff_tag_summary = ""
                for tag in tags.values():
                    try:
                        tiff_tag_summary += f"{tag.code}: {tag.name} = {tag.value}\n"
                    except Exception as e:
                        tiff_tag_summary += f"{tag.code}: <unreadable> ({e})\n"

                full_metadata = f"--- ASCII METADATA ---\n{readable_metadata}\n\n--- TIFF TAGS ---\n{tiff_tag_summary}"

                tifffile.imwrite(tif_path, image_data, description=full_metadata)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(full_metadata)

                print(f"✔ Saved: {tif_path.relative_to(folder_path)}")
                print(f"Metadata: {txt_path.relative_to(folder_path)}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")
