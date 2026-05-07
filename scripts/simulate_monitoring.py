from __future__ import annotations

import argparse
import concurrent.futures
import random
import time
from typing import Any

import requests


DEFAULT_API_URL = "http://localhost:8000"


NORMAL_OBJECTIVE4_INPUT = {
    "activity_value": 1200,
    "emission_factor": 0.42,
    "dechets_kg": 85,
    "station_kwh": 340,
    "year": 2026,
    "month": 5,
    "mode": "bus",
    "activity_type": "transport",
}


DRIFT_OBJECTIVE4_INPUT = {
    "activity_value": 999999,
    "emission_factor": 80,
    "dechets_kg": 200000,
    "station_kwh": 500000,
    "year": 2045,
    "month": 15,
    "mode": "unknown_mode",
    "activity_type": "new_activity",
}


def post_json(url: str, payload: dict[str, Any], timeout: float = 10.0) -> tuple[int, float]:
    start = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        return response.status_code, time.perf_counter() - start
    except requests.RequestException:
        return 0, time.perf_counter() - start


def trigger(api_url: str, scenario: str, duration_seconds: int) -> None:
    url = f"{api_url}/simulate/{scenario}"
    response = requests.post(url, json={"duration_seconds": duration_seconds}, timeout=10)
    print(f"trigger {scenario}: {response.status_code} {response.text}")


def predict_once(api_url: str, payload: dict[str, Any] | None = None) -> tuple[int, float]:
    body = {"input": payload or NORMAL_OBJECTIVE4_INPUT}
    return post_json(f"{api_url}/predict/4", body)


def high_traffic(api_url: str, requests_count: int, concurrency: int) -> None:
    print(f"high_traffic: sending {requests_count} requests with concurrency={concurrency}")
    latencies: list[float] = []
    statuses: dict[int, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                predict_once,
                api_url,
                {**NORMAL_OBJECTIVE4_INPUT, "activity_value": random.randint(500, 5000)},
            )
            for _ in range(requests_count)
        ]
        for future in concurrent.futures.as_completed(futures):
            status, latency = future.result()
            statuses[status] = statuses.get(status, 0) + 1
            latencies.append(latency)
    avg_latency = sum(latencies) / max(1, len(latencies))
    print(f"statuses={statuses} avg_latency_seconds={avg_latency:.3f}")


def api_errors(api_url: str, duration_seconds: int, requests_count: int) -> None:
    trigger(api_url, "api_errors", duration_seconds)
    for _ in range(requests_count):
        status, latency = predict_once(api_url)
        print(f"status={status} latency_seconds={latency:.3f}")
        time.sleep(0.3)


def high_latency(api_url: str, duration_seconds: int, requests_count: int) -> None:
    trigger(api_url, "high_latency", duration_seconds)
    for _ in range(requests_count):
        status, latency = predict_once(api_url)
        print(f"status={status} latency_seconds={latency:.3f}")
        time.sleep(0.2)


def model_drift(api_url: str, duration_seconds: int, requests_count: int) -> None:
    trigger(api_url, "model_drift", duration_seconds)
    for _ in range(requests_count):
        status, latency = predict_once(api_url, DRIFT_OBJECTIVE4_INPUT)
        print(f"status={status} latency_seconds={latency:.3f}")
        time.sleep(0.5)


def baseline(api_url: str, requests_count: int) -> None:
    for _ in range(requests_count):
        status, latency = predict_once(api_url)
        print(f"status={status} latency_seconds={latency:.3f}")
        time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate production-like monitoring scenarios.")
    parser.add_argument(
        "scenario",
        choices=["baseline", "high_traffic", "api_errors", "high_latency", "model_drift"],
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--duration-seconds", type=int, default=120)
    args = parser.parse_args()

    if args.scenario == "baseline":
        baseline(args.api_url, args.requests)
    elif args.scenario == "high_traffic":
        high_traffic(args.api_url, args.requests, args.concurrency)
    elif args.scenario == "api_errors":
        api_errors(args.api_url, args.duration_seconds, args.requests)
    elif args.scenario == "high_latency":
        high_latency(args.api_url, args.duration_seconds, args.requests)
    elif args.scenario == "model_drift":
        model_drift(args.api_url, args.duration_seconds, args.requests)


if __name__ == "__main__":
    main()
