import argparse
import json
from bs_pinn.trainer import BlackScholesSolver


def main(config_path: str) -> None:
    with open(config_path, "r") as f:
        config = json.load(f)

    solver = BlackScholesSolver(config)
    solver.train()
    solver.save(config.get("model_path", "model.pth"))
    print(f"\n✅ Model saved to {config.get('model_path', 'model.pth')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PINN on the Black-Scholes equation")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the configuration file (default: config.json)",
    )
    args = parser.parse_args()
    main(args.config)
