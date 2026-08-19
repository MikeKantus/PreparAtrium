def load_jpk(path):
    from nanolocz import NanoLocz

    nlz = NanoLocz(path)
    stack = nlz.get_stack()          # (frames, heightmap)
    meta = nlz.get_metadata()        # pixel_size_nm, frame_rate, etc.

    return stack, meta

