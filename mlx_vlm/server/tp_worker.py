"""Rank-1 entry point: python -m mlx_vlm.server.tp_worker --model <path>."""
import argparse, logging, os, sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    from .tp_mode import tp_hosts, tp_rank, worker_loop
    hosts = tp_hosts()
    if len(hosts) < 2:
        print("tp_worker: MLX_VLM_GLM5_TP_HOSTS must list >=2 hosts", file=sys.stderr)
        return 2
    worker_loop(a.model, hosts, tp_rank())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
