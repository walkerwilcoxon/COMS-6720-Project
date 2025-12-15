import argparse
import tomllib
from collections import defaultdict
from pathlib import Path

strategies = ["baseline", "pass2", "feedback", "commander"]


def load_problems(path: Path):
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data.get("problem", [])


def compute_stats(problems):
    total_problems = len(problems)
    total_time_seconds = sum(p.get("time", 0) for p in problems)
    total_time_hours = total_time_seconds / 3600.0

    total_verified = sum(1 for p in problems if p.get("verified", False))

    problems_by_category = defaultdict(int)
    verified_by_category = defaultdict(int)

    for p in problems:
        name = p.get("name", "")
        category = name.split("_", 1)[0] if "_" in name else name
        problems_by_category[category] += 1
        if p.get("verified", False):
            verified_by_category[category] += 1

    print(problems_by_category)

    return {
        "total_problems": total_problems,
        "total_time_hours": total_time_hours,
        "total_verified": total_verified,
        "problems_by_category": dict(problems_by_category),
        "verified_by_category": dict(verified_by_category),
    }


def format_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Deepseek")
    parser.add_argument(
        "--problem-set",
        default="test",
    )
    args = parser.parse_args()

    model = args.model
    problem_set = args.problem_set

    stats_by_strategy = {}

    for strategy in strategies:
        path = Path(f"output/{strategy}_{model}_{problem_set}_solutions.txt")
        if not path.exists():
            continue
        stats_by_strategy[strategy] = compute_stats(load_problems(path))

    if "baseline" not in stats_by_strategy:
        raise RuntimeError("Baseline file is required but was not found.")

    baseline = stats_by_strategy["baseline"]
    baseline_total = baseline["total_problems"]
    baseline_verified = baseline["total_verified"]

    baseline_acc = (baseline_verified / baseline_total) if baseline_total else 0.0

    output_path = Path(f"output/{model}_{problem_set}_summary.txt")
    with open(output_path, "w") as out:
        out.write(f"Model: {model}\n")
        out.write(f"Problem set: {problem_set}\n\n")

        out.write("=== Overall Statistics ===\n")

        # Baseline block
        out.write("\n[baseline]\n")
        out.write(f"Total problems: {baseline_total}\n")
        out.write(f"Total time (hours): {baseline['total_time_hours']:.2f}\n")
        out.write(f"Verified: {baseline_verified}\n")
        out.write(f"Accuracy: {format_pct(baseline_acc)}\n")

        for strategy in strategies:
            if strategy == "baseline" or strategy not in stats_by_strategy:
                continue

            s = stats_by_strategy[strategy]
            strategy_verified = s["total_verified"]
            combined_verified = baseline_verified + strategy_verified

            combined_acc = (combined_verified / baseline_total) if baseline_total else 0.0

            abs_improvement_problems = (strategy_verified / baseline_total) if baseline_total else 0.0

            out.write(f"\n[{strategy}]\n")
            out.write(f"Number of problems: {s['total_problems']}\n")
            out.write(f"Total time (hours): {s['total_time_hours']:.2f}\n")
            out.write(f"Number of problems verified: {strategy_verified}\n")
            out.write(f"Combined accuracy: {format_pct(combined_acc)}\n")
            out.write(f"Absolute improvement: {format_pct(abs_improvement_problems)}\n")

        out.write("\n=== Per-Category Statistics ===\n")

        # All categories from baseline define denominators
        baseline_probs_cat = baseline["problems_by_category"]
        baseline_ver_cat = baseline["verified_by_category"]

        for category in sorted(baseline_probs_cat.keys()):
            base_total_cat = baseline_probs_cat.get(category, 0)
            base_verified_cat = baseline_ver_cat.get(category, 0)

            out.write(f"\nCategory: {category}\n")
            out.write(f"  baseline:\n"
                      f"    Stage 1 problems: {base_total_cat}\n"
                      f"    Stage 1 verified: {base_verified_cat}\n"
                      f"    Percent verified: {format_pct(base_verified_cat / base_total_cat)}\n")

            for strategy in strategies:
                if strategy == "baseline" or strategy not in stats_by_strategy:
                    continue

                vstats = stats_by_strategy[strategy]
                strategy_verified_cat = vstats["verified_by_category"].get(category, 0)
                strategy_total_cat = vstats["problems_by_category"].get(category, 0)

                combined_verified_cat = base_verified_cat + strategy_verified_cat
                combined_acc_cat = (combined_verified_cat / base_total_cat) if base_total_cat else 0.0

                abs_improvement_cat = (strategy_verified_cat / base_total_cat) if base_total_cat else 0.0

                out.write(
                    f"  {strategy}:\n"
                    f"    Stage 2 problems: {strategy_total_cat}\n"
                    f"    Stage 2 verified: {strategy_verified_cat}\n"
                    f"    Total percent verified {format_pct(combined_acc_cat)}\n"
                    f"    Stage 2 percent improvement: +{format_pct(abs_improvement_cat)}\n"
                )

    print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()
