import tifffile

def extract_scan(tags):
    scan = {}
    fields = {
        "x_origin_nm": 32832,
        "y_origin_nm": 32833,
        "x_range_nm": 32834,
        "y_range_nm": 32835,
        "x_pixels": 32838,
        "y_pixels": 32839,
        "frame_rate_hz": 32841,
    }
    for key, code in fields.items():
        value = tags.get(code)
        if value is None:
            continue
        if "nm" in key:
            scan[key] = float(value) * 1e9
        elif "pixels" in key:
            scan[key] = int(value)
        else:
            scan[key] = float(value)
    return scan

def extract_cantilever_and_feedback(text):
    cantilever = {}
    feedback = {}
    cantilever_keys = {
        "amplitude", "calibration-environment", "cantilever-id",
        "cantilever-name", "defined", "frequency", "geometry",
        "qFactor", "sensitivity", "spring-constant",
    }
    feedback_keys = {
        "setpoint-feedback-settings.relative-setpoint"
    }
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key.startswith("cantilever-calibration-info."):
            short = key.replace("cantilever-calibration-info.", "")
            if short in cantilever_keys:
                cantilever[short] = value
        if key.startswith("feedback-mode.setpoint-feedback-settings."):
            short = key.replace("feedback-mode.setpoint-feedback-settings.", "")
            full_key = f"setpoint-feedback-settings.{short}"
            if full_key in feedback_keys:
                feedback[full_key] = value
    return cantilever, feedback

def extract_jpk_metadata(path):
    scan = {}
    cantilever = {}
    feedback = {}
    try:
        with tifffile.TiffFile(path) as tif:
            for page in tif.pages:
                tags = {tag.code: tag.value for tag in page.tags.values()}
                if not scan:
                    scan = extract_scan(tags)
                for value in tags.values():
                    if isinstance(value, str) and "cantilever-calibration-info" in value:
                        c, f = extract_cantilever_and_feedback(value)
                        cantilever.update(c)
                        feedback.update(f)
    except Exception:
        pass
    return {
        "scan": scan,
        "cantilever": cantilever,
        "feedback": feedback
    }
