import requests
import json
import argparse

# usage: $ python submit-job.py --request=request.json

base_url = "https://sig3.sig-gis-00.corespeq.com"

def build_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--request", required=True, help="A json file with all the inputs"
    )
    return parser

args_parser = build_parser()

def submit_job(
    request
):
    print(f"{request=}")
    with open(request, "r") as file:
        relay_args = json.load(file)
    body_dict = relay_args
    body = json.dumps(body_dict)
    pretty_body = json.dumps(body_dict, indent=2, sort_keys=True)
    print(f"Sending simulation request:\n{pretty_body}")
    http_resp = requests.post(
        f"{base_url}/api/submit-job",
        headers={
            "sig-auth": "AUTH_KEY",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        data=body,
        timeout=10,
    )
    http_resp.raise_for_status()
    return http_resp.json()

if __name__ == "__main__":
    args = args_parser.parse_args()
    response = submit_job(args.request)
    print(f"Response: {response}")


