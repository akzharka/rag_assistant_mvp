import json

from pathlib import Path


DOCUMENT_REGISTRY_PATH = Path("./data/documents.json")

DOCUMENT_REGISTRY_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


def load_document_registry():

    if not DOCUMENT_REGISTRY_PATH.exists():
        return {}

    try:
        with open(
            DOCUMENT_REGISTRY_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        return {}


def save_document_registry(registry):

    with open(
        DOCUMENT_REGISTRY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            registry,
            file,
            indent=2,
            ensure_ascii=False,
        )


document_registry = load_document_registry()