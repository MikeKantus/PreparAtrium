import os
import json
import tifffile
from playnano.io.loader import load_afm_stack

def preload_hsafm_folder(folder, output_folder, read_metadata_fn):
    jpk_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(".jpk")
    ])

    if not jpk_files:
        raise RuntimeError("No .jpk files found in HS-AFM folder.")

    os.makedirs(output_folder, exist_ok=True)
    expected_tiffs = [
        os.path.join(output_folder, jpk_name[:-4] + ".tif")
        for jpk_name in jpk_files
    ]
    source_tiffs = [
        os.path.join(folder, jpk_name[:-4] + ".tif")
        for jpk_name in jpk_files
    ]

    # A complete one-to-one cache lets us skip the expensive JPK decoder.
    if len(expected_tiffs) == len(jpk_files) and all(os.path.isfile(path) for path in expected_tiffs):
        return expected_tiffs
    if len(source_tiffs) == len(jpk_files) and all(os.path.isfile(path) for path in source_tiffs):
        return source_tiffs

    afm = load_afm_stack(folder)
    frames = afm.data
    if len(frames) != len(jpk_files):
        raise RuntimeError(
            f"JPK decoder returned {len(frames)} frames for {len(jpk_files)} source files."
        )

    generated_tiffs = []

    for i, frame in enumerate(frames):
        jpk_name = jpk_files[i]
        jpk_path = os.path.join(folder, jpk_name)

        tiff_name = jpk_name.replace(".jpk", ".tif")
        tiff_path = os.path.join(output_folder, tiff_name)

        json_name = jpk_name.replace(".jpk", ".json")
        json_path = os.path.join(output_folder, json_name)

        # Preserve already converted frames; write only sources missing a TIFF.
        if not os.path.exists(tiff_path):
            tifffile.imwrite(tiff_path, frame)

        if not os.path.exists(json_path):
            meta = read_metadata_fn(jpk_path)
            with open(json_path, "w") as f:
                json.dump(meta, f, indent=2)

        generated_tiffs.append(tiff_path)

    print("PRELOADER: Completed.")
    return generated_tiffs

