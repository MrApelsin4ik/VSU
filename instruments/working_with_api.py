import os
import json
import argparse
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv

try:
    load_dotenv(".env")
except Exception as e:
    pass
API_KEY = os.getenv("API_KEY")
print(API_KEY)
BASE_URL = os.getenv("API_URL", "http://127.0.0.1:9000").rstrip("/")
TIMEOUT = 10

def _headers():
    return {"X-API-KEY": API_KEY, "Content-Type": "application/json"}

class SimpleAPIClient:
    def __init__(self, base_url: str = BASE_URL, api_key: Optional[str] = API_KEY):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get_missing(self) -> Dict[str, Any]:
        url = self._url("/api/ai/missing/")
        r = requests.get(url, headers=_headers(), timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()

    def get_full(self, model: str, pk: int) -> Dict[str, Any]:
        url = self._url(f"/api/ai/full/{model}/{pk}/")
        r = requests.get(url, headers=_headers(), timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()

    def update_items(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        url = self._url("/api/ai/update/")
        payload = {"items": items}
        r = requests.post(url, headers=_headers(), data=json.dumps(payload), timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()

    def update_item(self, model: str, pk: int, description_ai: Optional[str] = None, description_ai_l2: Optional[str] = None) -> Dict[str, Any]:
        data = {"model": model, "id": pk}
        if description_ai is not None:
            data["description_ai"] = description_ai
        if description_ai_l2 is not None:
            data["description_ai_l2"] = description_ai_l2
        url = self._url("/api/ai/update/")
        r = requests.post(url, headers=_headers(), data=json.dumps(data), timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()

def _print_json(obj: Any):
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("missing")
    p_full = sub.add_parser("full")
    p_full.add_argument("model", choices=["course", "news", "announcement"])
    p_full.add_argument("id", type=int)
    p_update = sub.add_parser("update-single")
    p_update.add_argument("model", choices=["course", "news", "announcement"])
    p_update.add_argument("id", type=int)
    p_update.add_argument("--description_ai", default=None)
    p_update.add_argument("--description_ai_l2", default=None)
    p_update_bulk = sub.add_parser("update-bulk")
    p_update_bulk.add_argument("jsonfile", help="path to json file with array of items or single object")
    args = parser.parse_args()
    client = SimpleAPIClient()
    if args.cmd == "missing":
        res = client.get_missing()
        _print_json(res)
    elif args.cmd == "full":
        res = client.get_full(args.model, args.id)
        _print_json(res)
    elif args.cmd == "update-single":
        res = client.update_item(args.model, args.id, args.description_ai, args.description_ai_l2)
        _print_json(res)
    elif args.cmd == "update-bulk":
        with open(args.jsonfile, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else (data.get("items") if isinstance(data, dict) and "items" in data else [data])
        res = client.update_items(items)
        _print_json(res)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
