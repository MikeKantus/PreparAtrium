def read_metadata_jpk(path):
    """
    Lee metadatos de archivos JPK usando TIFF tags.
    Basado en el script funcional proporcionado por Miguel.
    """

    import tifffile

    meta = {}

    # --- SCAN TAGS ---
    scan_fields = {
        "x_origin_nm": 32832,
        "y_origin_nm": 32833,
        "x_range_nm": 32834,
        "y_range_nm": 32835,
        "x_pixels": 32838,
        "y_pixels": 32839,
        "frame_rate": 32841,   # normalizamos nombre
    }

    # --- CANTILEVER / FEEDBACK ---
    cantilever_keys = {
        "amplitude",
        "calibration-environment",
        "cantilever-id",
        "cantilever-name",
        "defined",
        "frequency",
        "geometry",
        "qFactor",
        "sensitivity",
        "spring-constant",
    }

    feedback_keys = {
        "setpoint-feedback-settings.relative-setpoint"
    }

    def extract_scan(tags):
        scan = {}
        for key, code in scan_fields.items():
            value = tags.get(code)
            if value is None:
                continue

            try:
                if "nm" in key:
                    val = float(value)
                    # si está en metros, convertir a nm
                    scan[key] = val * 1e9 if val < 1 else val
                elif "pixels" in key:
                    scan[key] = int(value)
                else:
                    scan[key] = float(value)
            except Exception:
                pass

        return scan

    def extract_cantilever_and_feedback(text):
        cantilever = {}
        feedback = {}

        for line in text.splitlines():
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            # CANTILEVER
            if key.startswith("cantilever-calibration-info."):
                short = key.replace("cantilever-calibration-info.", "")
                if short in cantilever_keys:
                    cantilever[short] = value

            # FEEDBACK
            if key.startswith("feedback-mode.setpoint-feedback-settings."):
                short = key.replace("feedback-mode.setpoint-feedback-settings.", "")
                full_key = f"setpoint-feedback-settings.{short}"
                if full_key in feedback_keys:
                    feedback[full_key] = value

        return cantilever, feedback

    # --- LECTURA TIFF ---
    try:
        with tifffile.TiffFile(path) as tif:
            scan = {}
            cantilever = {}
            feedback = {}

            for page in tif.pages:
                tags = {tag.code: tag.value for tag in page.tags.values()}

                # SCAN
                if not scan:
                    scan = extract_scan(tags)

                # CANTILEVER + FEEDBACK
                for value in tags.values():
                    if isinstance(value, str) and "cantilever-calibration-info" in value:
                        c, f = extract_cantilever_and_feedback(value)
                        cantilever.update(c)
                        feedback.update(f)

            # fusionar todo
            meta.update(scan)
            meta.update(cantilever)
            meta.update(feedback)

            # canal (si existe)
            if "channel" not in meta:
                meta["channel"] = None

            return meta

    except Exception as e:
        print(f"[read_metadata_jpk] ERROR leyendo TIFF: {e}")

    # Si falla, devolver dict vacío
    return {}
