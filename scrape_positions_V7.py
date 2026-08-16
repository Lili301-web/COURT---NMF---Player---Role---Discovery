"""Retrieve positions for the external-validation sample."""
import re
import time
import urllib.request

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

PLAYERS = [
    ("John Galvin", "Fairleigh Dickinson", "john-galvin"),
    ("Brandon Davies", "BYU", "brandon-davies"),
    ("Akiean Frederick", "Kent St", "akiean-frederick"),
    ("Renaldo Woolridge", "Tennessee", "renaldo-woolridge"),
    ("Jair Bolden", "Butler", "jair-bolden"),
    ("Aaron Nettles", "Seattle", "aaron-nettles"),
    ("Chance McMillian", "Grand Canyon", "chance-mcmillian"),
    ("Maxwell Evans", "Vanderbilt", "maxwell-evans"),
    ("Cameron Carmical", "Eastern Kentucky", "cameron-carmical"),
    ("Kendall Cutler", "South Dakota", "kendall-cutler"),
    ("Hasan Abdullah", "Navy", "hasan-abdullah"),
    ("CJ Davis", "Columbia", "cj-davis"),
    ("Abu Kigab", "Oregon", "abu-kigab"),
    ("Mark Carter", "Alcorn", "mark-carter"),
    ("Damian Goodwin", "USC Upstate", "damian-goodwin"),
    ("Stephan Gabriel", "Bradley", "stephan-gabriel"),
    ("Tom Jervis", "Troy", "tom-jervis"),
    ("Robert Nwankwo", "Towson", "robert-nwankwo"),
    ("Mouphtaou Yarou", "Villanova", "mouphtaou-yarou"),
    ("Chris Vogt", "Northern Kentucky", "chris-vogt"),
]


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception:
        return None


results = []
for name, team, base in PLAYERS:
    html, used = None, None
    for suffix in ("-1", "-2", "-3"):
        h = fetch(f"https://www.sports-reference.com/cbb/players/{base}{suffix}.html")
        if h and name.split()[0].lower() in h.lower():
            # check the title matches the player name
            t = re.search(r"<title>([^<]*)</title>", h)
            if t and name.lower().replace("j.", "j")[:6] in t.group(1).lower().replace(".", ""):
                html, used = h, base + suffix
                break
    if html is None:
        results.append((name, team, "NOT_FOUND", "", ""))
        print(f"{name:20s} NOT FOUND", flush=True)
        time.sleep(1.0)
        continue
    # meta block -> text
    m = re.search(r'<div id="meta"[^>]*>(.*?)</div>\s*<div', html, re.S)
    block = m.group(1) if m else html[:20000]
    txt = re.sub(r"<[^>]+>", " ", block)
    txt = re.sub(r"&nbsp;|&#160;", " ", txt)
    txt = re.sub(r"\s+", " ", txt)
    pos = re.search(r"Position:\s*([A-Za-z\-]+)", txt)
    pos = pos.group(1) if pos else ""
    schools_meta = re.findall(r"Schools?:\s*([^|]+?)\s*(?:High School|More player|Born|Position|Hometown|$)", txt)
    # per-season positions from the stats table
    season_pos = re.findall(r'data-stat="pos"[^>]*>([^<]*)<', html)
    season_pos = [p.strip() for p in season_pos if p.strip()]
    print(f"{name:20s} [{used}] meta_pos={pos:18s} season_pos={season_pos} meta_school={schools_meta[:1]}",
          flush=True)
    results.append((name, team, used, pos, ",".join(season_pos)))
    time.sleep(1.2)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(BASE_DIR, "outputs", "v7_rerun")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "scraped_positions.tsv")

with open(output_path, "w",
          encoding="utf-8") as f:
    f.write("player\texpected_team\tslug\tmeta_position\tseason_positions\n")
    for row in results:
        f.write("\t".join(str(x) for x in row) + "\n")
print("saved")
