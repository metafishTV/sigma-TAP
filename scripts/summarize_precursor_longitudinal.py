import csv
import json
import sys
from collections import Counter


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('usage: python scripts/summarize_precursor_longitudinal.py <precursor_longitudinal.csv> [out.json]')

    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else 'outputs/precursor_longitudinal_summary.json'

    with open(src, newline='') as f:
        rows = list(csv.DictReader(f))

    short_counts = Counter(r.get('regime_short', '') for r in rows)
    long_counts = Counter(r.get('regime_long', '') for r in rows)
    under_resolved_long_true = sum(1 for r in rows if r.get('under_resolved_long', '').lower() == 'true')

    summary = {
        'rows': len(rows),
        'short_counts': dict(short_counts),
        'long_counts': dict(long_counts),
        'under_resolved_long_true': under_resolved_long_true,
        'precursor_candidate_count': len(rows),
        'resolved_explosive_count': int(long_counts.get('explosive', 0)),
    }

    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
