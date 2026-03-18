import pandas as pd

REQUIRED_FEATURES = [
    "planet_radius",
    "planet_mass",
    "orbital_period",
    "semi_major_axis",
    "equilibrium_temperature",
    "star_temperature",
    "star_luminosity",
    "star_metallicity",
    "insolation_flux",
    "star_mass",
    "orbital_eccentricity",
    "planet_density",
    "habitability_index",
    "stellar_compatibility",
    "orbital_stability",
    "star_G",
    "star_K",
    "star_M",
    "star_Unknown",
]

FEATURE_RANGES = {
    "planet_radius":            (0.1,    30.0,     "Earth radii"),
    "planet_mass":              (0.01,   5000.0,   "Earth masses"),
    "orbital_period":           (0.1,    100000.0, "days"),
    "semi_major_axis":          (0.001,  100.0,    "AU"),
    "equilibrium_temperature":  (10.0,   5000.0,   "K"),
    "star_temperature":         (2000.0, 50000.0,  "K"),
    "star_luminosity":          (-5.0,   6.0,      "log(L/Lsun)"),
    "star_metallicity":         (-3.0,   1.5,      "dex"),
    "insolation_flux":          (0.0,    100000.0, "Earth flux"),
    "star_mass":                (0.05,   150.0,    "Solar masses"),
    "orbital_eccentricity":     (0.0,    1.0,      "dimensionless"),
    "planet_density":           (0.0,    100.0,    "g/cm3"),
    "habitability_index":       (0.0,    1.0,      "score"),
    "stellar_compatibility":    (0.0,    1.0,      "score"),
    "orbital_stability":        (0.0,    1.0,      "score"),
    "star_G":                   (0,      1,        "binary"),
    "star_K":                   (0,      1,        "binary"),
    "star_M":                   (0,      1,        "binary"),
    "star_Unknown":             (0,      1,        "binary"),
}

BINARY_FEATURES = {"star_G", "star_K", "star_M", "star_Unknown"}


def validate_input(data):
    cleaned = {}

    missing = [f for f in REQUIRED_FEATURES if f not in data]
    if missing:
        return None, f"Missing required fields: {missing}"

    for feat in REQUIRED_FEATURES:
        try:
            val = float(data[feat])
        except (TypeError, ValueError):
            return None, f"Field '{feat}' must be a number, got: {data[feat]!r}"

        low, high, unit = FEATURE_RANGES[feat]
        if not (low <= val <= high):
            return None, f"Field '{feat}' = {val} out of range [{low}, {high}] ({unit})"

        if feat in BINARY_FEATURES and val not in (0.0, 1.0):
            return None, f"Field '{feat}' must be 0 or 1"

        cleaned[feat] = val

    star_flags = sum(cleaned[f] for f in ("star_G", "star_K", "star_M", "star_Unknown"))
    if star_flags != 1:
        return None, "Exactly one of star_G, star_K, star_M, star_Unknown must be 1"

    return cleaned, None


def build_dataframe(cleaned):
    return pd.DataFrame(
        [[cleaned[f] for f in REQUIRED_FEATURES]],
        columns=REQUIRED_FEATURES,
    )


def interpret_prediction(label, proba):
    if label == 1:
        category    = "Potentially Habitable"
        description = "This exoplanet's parameters are consistent with habitability."
    else:
        category    = "Non-Habitable"
        description = "This exoplanet's parameters fall outside the habitable range."

    if proba >= 0.80:
        confidence_label = "High"
    elif proba >= 0.50:
        confidence_label = "Moderate"
    else:
        confidence_label = "Low"

    return {
        "habitable":        bool(label),
        "category":         category,
        "description":      description,
        "confidence_score": round(proba, 4),
        "confidence_label": confidence_label,
    }


def rank_planets(planets, model):
    results = []

    for idx, planet_data in enumerate(planets):
        planet_data = dict(planet_data)
        name = planet_data.pop("name", f"Planet_{idx + 1}")

        cleaned, err = validate_input(planet_data)
        if err:
            results.append({"name": name, "rank": None, "error": err})
            continue

        df    = build_dataframe(cleaned)
        label = int(model.predict(df)[0])
        proba = float(model.predict_proba(df)[0][1])

        results.append({"name": name, **interpret_prediction(label, proba)})

    valid   = [r for r in results if "error" not in r]
    invalid = [r for r in results if "error" in r]

    valid.sort(key=lambda x: x["confidence_score"], reverse=True)
    for i, r in enumerate(valid, 1):
        r["rank"] = i

    return valid + invalid