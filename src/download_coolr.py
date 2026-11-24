from pathlib import Path
import json
import requests

BASE_URL = (
    "https://gis.earthdata.nasa.gov/portal/rest/services/"
    "Landslides/COOLR_Events_Polygons/FeatureServer/0/query"
)
OUTPUT = Path("data/raw/nasa_coolr_events.geojson")


def download():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    params = {
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
        "resultRecordCount": 2000,
        "resultOffset": 0,
    }

    all_features = []
    while True:
        response = requests.get(BASE_URL, params=params, timeout=60)
        response.raise_for_status()

        features = response.json().get("features", [])
        all_features.extend(features)
        print(f"Fetched offset {params['resultOffset']} - {len(features)} features")

        if len(features) < 2000:
            break

        params["resultOffset"] += 2000

    geojson = {
        "type": "FeatureCollection",
        "features": all_features,
    }

    OUTPUT.write_text(json.dumps(geojson), encoding="utf-8")
    print(f"Saved {len(all_features)} features to {OUTPUT}")


if __name__ == "__main__":
    download()
