import argparse
from pathlib import Path
import yaml
from compass_ecl_mas.simulator.education import simulate_education_A

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default="data/education_A_v1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dcfg = cfg["data"]

    df = simulate_education_A(seed=args.seed, n_students=int(dcfg["n_students"]), n_schools=int(dcfg["n_schools"]), bias=dcfg.get("bias", {}))
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"education_A_seed{args.seed}.csv"
    df.to_csv(path, index=False)
    print(f"[OK] wrote {path}")

if __name__ == "__main__":
    main()
