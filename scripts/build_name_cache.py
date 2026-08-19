"""
Look up batter_id -> name across all years, save to data/reference/batter_names.json

Run: python -m scripts.build_name_cache
"""
import json
from pathlib import Path

import psycopg2
import pybaseball

from src.config import DSN

OUT  = Path(__file__).parent.parent / "data" / "reference" / "batter_names.json"

def main():
    # Fetch valid fly-ball batter IDs across all years
    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT batter
                FROM statcast
                WHERE game_type = 'R'
                  AND type = 'X'
                  AND bb_type IN ('fly_ball', 'line_drive')
                  AND events != 'home_run'
                  AND hit_distance_sc IS NOT NULL
                  AND launch_speed    IS NOT NULL
                  AND launch_angle    IS NOT NULL
            """)
            ids = [r[0] for r in cur.fetchall()]

    print(f"Queried {len(ids)} batter IDs from DB")

    # pybaseball batch lookup
    df = pybaseball.playerid_reverse_lookup(ids, key_type='mlbam')
    name_map = {}
    for _, row in df.iterrows():
        bid = int(row['key_mlbam'])
        name_map[bid] = f"{row['name_last'].title()}, {row['name_first'].title()}"

    # Fill in gaps (playerid_reverse_lookup doesn't find every ID)
    missing = set(ids) - set(name_map.keys())
    if missing:
        print(f"Warning: {len(missing)} IDs not found: {sorted(missing)[:5]}...")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(name_map, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(name_map)} names to {OUT}")

if __name__ == '__main__':
    main()
