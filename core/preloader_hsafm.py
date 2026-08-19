import os
import json
import tifffile
from playnano.io.loader import load_afm_stack

def preload_hsafm_folder(folder, output_folder, read_metadata_fn):
    """
    Preload HS-AFM folder:
    - Load full HS-AFM stack with PlayNano
    - Export each frame as TIFF
    - Export metadata of each .jpk as JSON
    - Link TIFF ↔ JPK by name

    Parameters
    ----------
    folder : str
        Folder containing HS-AFM .jpk files.
    output_folder : str
        Folder where TIFF + JSON will be saved.
    read_metadata_fn : callable
        Function that reads metadata from a .jpk file (your _read_metadata_jpk).

    Returns
    -------
    list of str
        Paths to generated TIFF files.
    """

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all .jpk files in the folder
    jpk_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(".jpk")
    ])

    if not jpk_files:
        raise RuntimeError("No .jpk files found in HS-AFM folder.")

    # Load full HS-AFM stack with PlayNano
    print("PRELOADER: Loading HS-AFM stack with PlayNano...")
    afm = load_afm_stack(folder)
    frames = afm.data

    if frames.shape[0] != len(jpk_files):
        print("WARNING: Number of frames does not match number of .jpk files.")

    generated_tiffs = []

    # Export each frame
    for i, frame in enumerate(frames):
        jpk_name = jpk_files[i]
        jpk_path = os.path.join(folder, jpk_name)

        # TIFF name
        tiff_name = jpk_name.replace(".jpk", ".tif")
        tiff_path = os.path.join(output_folder, tiff_name)

        # JSON name
        json_name = jpk_name.replace(".jpk", ".json")
        json_path = os.path.join(output_folder, json_name)

        # Save TIFF
        tifffile.imwrite(tiff_path, frame)

        # Save metadata JSON
        meta = read_metadata_fn(jpk_path)
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        generated_tiffs.append(tiff_path)

        print(f"PRELOADER: Saved {tiff_name} + {json_name}")

    print("PRELOADER: Completed.")
    return generated_tiffs
