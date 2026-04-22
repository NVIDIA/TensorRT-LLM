import json

NON_PERMISSIVE_LICENSE_PREFIXES = ("GPL", "LGPL", "GCC GPL", "AGPL", "SSPL", "CPAL", "EUPL")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def is_non_permissive(licenses):
    return len(licenses) == 0 or any(
        lic.startswith(NON_PERMISSIVE_LICENSE_PREFIXES) for lic in licenses
    )
