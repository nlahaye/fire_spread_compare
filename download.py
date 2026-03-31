import requests
import json
import argparse

# usage: $ python download.py --filename "remote-filename.tar.gz2"

base_url = "https://sig3.sig-gis-00.corespeq.com"

def build_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--filename", required=True, help="Remote file to download")
    return parser

def download(filename):
    print(f"Downloading file {filename=} ...")
    response = requests.get(
        f"{base_url}/api/download/{filename}",
        headers={
            "sig-auth": "AUTH_KEY",
        },
        stream=True,
    )
    if response.status_code == 200:
        # Open the file in binary mode for writing
        with open(filename, "wb") as file:
            # Write the content of the response to the file
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Successfully downloaded {filename}")
    else:
        print(
            f"Failed to download file '{filename}'. Status code: {response.status_code}"
        )
    response.raise_for_status()
    return response

if __name__ == "__main__":
    args_parser = build_parser()
    args = args_parser.parse_args()
    response = download(args.filename)
    print(f"Response: {response=}")
