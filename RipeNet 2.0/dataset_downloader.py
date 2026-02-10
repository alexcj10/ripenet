import os
import requests
import time
import random
import math
from ddgs import DDGS
from tqdm import tqdm

BASE_DIR = "."

FRUITS = [
    # "apple",
    # "banana",
    # "dragonfruit",
    "mango",
    "orange",
    "papaya",
    "pineapple"
]

STAGES = ["Fresh", "Unripe", "Rotten"]
TOTAL_PER_CLASS = 1000


# ---------------------------
# QUERY SETS
# ---------------------------
def get_queries(fruit, stage):

    if stage == "Unripe":
        return [
            f"unripe {fruit} whole fruit",
            f"green unripe {fruit} on tree",
            f"raw {fruit} fruit",
            f"hard green {fruit}",
            f"immature {fruit} fruit",
            f"unripe {fruit} sliced",
            f"green {fruit} macro"
        ]

    if stage == "Fresh":
        return [
            f"fresh ripe {fruit}",
            f"ripe {fruit} close up",
            f"juicy {fruit}",
            f"{fruit} on table",
            f"fresh {fruit} slices",
            f"perfect {fruit} fruit",
            f"{fruit} fruit bowl"
        ]

    return [
        f"rotten {fruit}",
        f"overripe {fruit}",
        f"moldy {fruit}",
        f"spoiled {fruit}",
        f"bad {fruit} fruit",
        f"decaying {fruit}",
        f"rotten {fruit} close up"
    ]


# ---------------------------
# DOWNLOADER
# ---------------------------
def download_class(fruit, stage):

    folder = os.path.join(BASE_DIR, fruit, stage)
    os.makedirs(folder, exist_ok=True)

    queries = get_queries(fruit, stage)
    per_query = math.ceil(TOTAL_PER_CLASS / len(queries))

    count = 0

    with DDGS() as ddgs:
        for q in queries:
            print(f"   → {q}")

            try:
                results = ddgs.images(q, max_results=per_query)
            except:
                continue

            for r in results:
                if count >= TOTAL_PER_CLASS:
                    break

                try:
                    url = r["image"]
                    data = requests.get(url, timeout=8).content

                    with open(os.path.join(folder, f"{count}.jpg"), "wb") as f:
                        f.write(data)

                    count += 1
                    time.sleep(random.uniform(0.3, 0.6))
                except:
                    pass

    print(f"   ✅ Saved {count} images\n")


# ---------------------------
# MAIN
# ---------------------------
for fruit in FRUITS:
    for stage in STAGES:
        print(f"\nDownloading {fruit} → {stage}")
        download_class(fruit, stage)

print("\n🔥 DATASET COMPLETE")
