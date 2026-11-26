"""Downloads external data dependencies."""

import json
import shutil
import urllib.request
import zipfile

from pathlib import Path


ROOT         = Path(__file__).parents[1]
DATA         = ROOT / "data"
DEPENDENCIES = DATA / "data_dependencies.json"


def download(name,
             dependency):

    directory = DATA / dependency["directory"]
    archive   = DATA / dependency["archive"]

    if directory.exists():
        print(f"{name}: already available")
        return

    DATA.mkdir(exist_ok = True)

    print(f"{name}: downloading {dependency['url']}")

    with urllib.request.urlopen(dependency["url"]) as response:
        with open(archive, "wb") as file:
            shutil.copyfileobj(response,
                               file)

    if dependency.get("extract", False):

        print(f"{name}: extracting")

        directory.mkdir(exist_ok = True)

        with zipfile.ZipFile(archive) as file:
            file.extractall(directory)

        archive.unlink()

    print(f"{name}: ready")


def main():

    with open(DEPENDENCIES,
              "r",
              encoding = "utf-8") as file:

        dependencies = json.load(file)

    for name, dependency in dependencies.items():
        download(name,
                 dependency)


if __name__ == "__main__":
    main()