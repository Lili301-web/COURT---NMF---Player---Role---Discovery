"""Check the assembled submission and produce contact sheets for visual QA."""
from pathlib import Path
import hashlib
import json
import re
from pypdf import PdfReader
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1]
pdf = root / "output/pdf/COURT_NMF_ICMLC2027_submission.pdf"
qa = root / "output/qa"
qa.mkdir(parents=True, exist_ok=True)
reader = PdfReader(pdf)
assert len(reader.pages) <= 20
content = "\n".join(p.extract_text() for p in reader.pages)
assert "??" not in content
log = pdf.with_suffix(".log").read_text(encoding="utf-8", errors="replace")
assert "Overfull" not in log
assert "undefined" not in log.lower()
assert "duplicate ignored" not in log
assert "Label(s) may have changed" not in log
digest = hashlib.sha256((root / "svproc.cls").read_bytes()).hexdigest()
assert digest == "ddfc1f23edfbb51f13bb6a6e427b63c41cc9a965f08a3bc65e83c81ee9ae9560"
assert (root / "svproc.cls").read_bytes() == (root / "styles/svproc.cls").read_bytes()
sources = [root / "COURT_NMF_ICMLC2027_submission.tex", *sorted((root / "tex/submission").glob("*.tex"))]
alltex = "\n".join(p.read_text(encoding="utf-8") for p in sources)
for banned in (r"\setlength", r"\vspace", r"\fontsize", r"\scriptsize", r"\footnotesize", r"\captionsetup", r"\linespread", r"\geometry"):
    assert banned not in alltex, banned
labels = re.findall(r"\\label\{([^}]+)\}", alltex)
refs = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", alltex)
assert len(labels) == len(set(labels))
assert set(refs) <= set(labels)
citations = {x.strip() for group in re.findall(r"\\cite\{([^}]+)\}", alltex) for x in group.split(",")}
bibitems = re.findall(r"\\bibitem\{([^}]+)\}", alltex)
assert citations == set(bibitems)
assert len(bibitems) == len(set(bibitems))

# Numeric tokens in retained table rows must come from the original TeX snapshot.
number = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")
for new_name, old_name in [("table_benchmark", "table_benchmark"), ("table_roleposition", "table_roleposition"), ("table_softmembership", "table_softmembership"), ("table_sensitivity", "table_sensitivity"), ("table_montecarlo", "section_montecarlo"), ("table_solver", "supplement_montecarlo")]:
    new = (root / f"tex/submission/{new_name}.tex").read_text(encoding="utf-8")
    old = (root / f"tex/{old_name}.tex").read_text(encoding="utf-8")
    for line in new.splitlines():
        if " & " in line and re.search(r"\d\.\d", line):
            for value in number.findall(line):
                assert value in old, (new_name, value)

aux = pdf.with_suffix('.aux').read_text(encoding='utf-8')
appendix_page = int(re.search(r'\\newlabel\{app:variables\}\{\{A\}\{(\d+)\}', aux).group(1))
report = {
    "pdf": str(pdf), "pages": len(reader.pages), "page_limit": 20,
    "class_sha256": digest, "class_unchanged": True,
    "text_width_cm": 12.2, "text_height_cm": 19.3,
    "body_font_pt": 10, "class_float_font_pt": 9,
    "main_text_ends_on_page": appendix_page - 1, "appendix_starts_on_page": appendix_page,
    "references": len(bibitems), "tables": alltex.count(r"\begin{table}"),
    "figures": alltex.count(r"\begin{figure}"),
    "undefined_references": 0, "duplicate_labels": 0, "overfull_boxes": 0,
    "remaining_warning": "svproc loads aliascnt, which the installed TeX distribution marks obsolete",
    "numerical_validation_scope": "Preserved retained values from supplied TeX. Raw result artifacts conflict; see REVISION_NOTES_20P.md.",
    "page_text_lengths": [len(p.extract_text()) for p in reader.pages],
}
(qa / "submission_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
(qa / "submission_text.txt").write_text(content, encoding="utf-8")

imgs = sorted((root / "tmp/pdfs/submission").glob("page-*.png"))
assert len(imgs) == len(reader.pages)
for start in range(0, len(imgs), 6):
    group = imgs[start:start+6]
    canvas = Image.new("RGB", (1300, 3*865), "#dddddd")
    draw = ImageDraw.Draw(canvas)
    for pos, path in enumerate(group):
        im = Image.open(path).convert("RGB")
        im.thumbnail((630, 825))
        x, y = (pos % 2)*650+10, (pos//2)*865+30
        canvas.paste(im, (x,y))
        draw.text((x,y-22), f"PAGE {start+pos+1}", fill="black")
    canvas.save(qa / f"contact-{start+1:02d}.png")
print(json.dumps(report, indent=2))
