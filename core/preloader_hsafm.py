import os
import json
import tifffile
from playnano.io.loader import load_afm_stack

def preload_hsafm_folder(folder, output_folder, read_metadata_fn):
    # Guardamos en la misma carpeta que los .jpk
    output_folder = folder

    jpk_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(".jpk")
    ])

    if not jpk_files:
        raise RuntimeError("No .jpk files found in HS-AFM folder.")

    print("PRELOADER: HS-AFM folder =", folder)

    afm = load_afm_stack(folder)
    frames = afm.data

    generated_tiffs = []

    for i, frame in enumerate(frames):
        jpk_name = jpk_files[i]
        jpk_path = os.path.join(folder, jpk_name)

        tiff_name = jpk_name.replace(".jpk", ".tif")
        tiff_path = os.path.join(output_folder, tiff_name)

        json_name = jpk_name.replace(".jpk", ".json")
        json_path = os.path.join(output_folder, json_name)

        # Si ya existen TIFF/JSON, no los regeneramos
        if not os.path.exists(tiff_path):
            tifffile.imwrite(tiff_path, frame)
            print("PRELOADER: wrote", tiff_name)
        else:
            print("PRELOADER: TIFF exists, skipping", tiff_name)

        if not os.path.exists(json_path):
            meta = read_metadata_fn(jpk_path)
            with open(json_path, "w") as f:
                json.dump(meta, f, indent=2)
            print("PRELOADER: wrote", json_name)
        else:
            print("PRELOADER: JSON exists, skipping", json_name)

        generated_tiffs.append(tiff_path)

    print("PRELOADER: Completed.")
    return generated_tiffs

