import json

PERMISSIVE_LICENSES = {
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "apache-2.0",
    "isc",
    "0bsd",
    "zlib",
    "unlicense",
    "python-2.0",
    "postgresql",
    "nvidia license",
    "python-license-2.0",
    "public domain",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def is_permissive(licenses):
    return all(
        license_name.strip().casefold().replace(" ", "-") in PERMISSIVE_LICENSES
        for license_name in licenses
    )
