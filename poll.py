import sys
import requests
import argparse
import json

# usage: $ python poll.py --job-id 87bcdf98-3a9c-475b-8376-7505562ca87a  (<-- replace with the real job id returned from the submit-job call)

def build_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--job-id",
        required=True,
        type=str,
    )
    return parser

def check_job(api_url, job_id):
    print(f"Fetching results from job {job_id}...")
    http_resp = requests.get(
        f"{api_url}/poll/{job_id}",
        headers={
            "sig-auth": "AUTH_KEY",
        },
        timeout=10,
    )
    http_resp.raise_for_status()
    body = http_resp.json()
    return body

if __name__ == "__main__":
    args_parser = build_parser()
    api_url = "https://sig3.sig-gis-00.corespeq.com/api"
    args = args_parser.parse_args()
    response = check_job(api_url, args.job_id)
    print(f"Response: {json.dumps(response, indent=2, sort_keys=True)}")


