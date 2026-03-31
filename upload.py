import requests
import json
import argparse

# usage: $ python upload.py --src my-geojson.json
# NOTE: the `--src` file must be in the same folder this script resides

def build_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--src", required=True, help="Local file to upload")
    return parser

def upload(src):
    print(f"Uploading file {src=} ...")
    with open(src, "rb") as f:
        files = {"file": (src, f)}
        http_resp = requests.post(
            f"{api_url}/upload",
            headers={
                "sig-auth": "AUTH_KEY",
            },
            files=files,
            timeout=10,
        )
    http_resp.raise_for_status()
    return http_resp.json()

if __name__ == "__main__":
    args_parser = build_parser()
    api_url = "https://sig3.sig-gis-00.corespeq.com/api"
    args = args_parser.parse_args()
    response = upload(args.src)
    print(f"Response: {response}")






