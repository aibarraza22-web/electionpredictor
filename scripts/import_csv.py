"""Placeholder replaceable CSV adapter: records source metadata; production parsers must preserve available_at."""
import argparse, csv
p=argparse.ArgumentParser();p.add_argument('path');args=p.parse_args()
with open(args.path) as f: print(f'Read {sum(1 for _ in csv.DictReader(f))} rows; validate schema before ingesting.')
