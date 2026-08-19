"""Build the MrVLA technical report PDF from lightweight markup files.

Markup supported (one construct per line unless noted):
  # / ## / ###   headings (H1 starts a new page)
  >              callout box (consecutive lines merge)
  $$             formula block (consecutive lines merge, monospace, boxed)
  - / * / 1.     bullet
  |a|b|          table row; a row of |---| marks the header separator
  ```            fenced code block
  (blank)        paragraph break
Inline: **bold**, *italic*, `code`, _{sub}, ^{super}
"""
from __future__ import annotations

import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageBreak, PageTemplate, Paragraph, Spacer, Table,
    TableStyle,
)

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5a5a5a")
RULE = colors.HexColor("#c8c8c8")
BOXBG = colors.HexColor("#f4f4f2")
CALLBG = colors.HexColor("#eef2f6")
CALLBAR = colors.HexColor("#4a6fa5")
HEADBG = colors.HexColor("#e7e7e4")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold",
                            fontSize=26, leading=31, textColor=INK, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", parent=ss["Normal"], fontName="Helvetica",
                               fontSize=12.5, leading=17, textColor=MUTED, alignment=TA_LEFT),
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=19,
                         leading=23, textColor=INK, spaceBefore=2, spaceAfter=10),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=13.5,
                         leading=17, textColor=INK, spaceBefore=14, spaceAfter=6),
    "h3": ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=11,
                         leading=14, textColor=colors.HexColor("#333333"),
                         spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=9.6,
                           leading=14.2, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7),
    "bullet": ParagraphStyle("bullet", parent=ss["Normal"], fontName="Helvetica", fontSize=9.6,
                             leading=14.0, textColor=INK, leftIndent=13, bulletIndent=3,
                             spaceAfter=3.5, alignment=TA_LEFT),
    "formula": ParagraphStyle("formula", parent=ss["Normal"], fontName="Courier", fontSize=9,
                              leading=13, textColor=INK, leftIndent=8, rightIndent=8,
                              backColor=BOXBG, borderPadding=7, spaceBefore=6, spaceAfter=8),
    "code": ParagraphStyle("code", parent=ss["Normal"], fontName="Courier", fontSize=8.2,
                           leading=11, textColor=INK, leftIndent=8, rightIndent=8,
                           backColor=BOXBG, borderPadding=6, spaceBefore=5, spaceAfter=7),
    "callout": ParagraphStyle("callout", parent=ss["Normal"], fontName="Helvetica", fontSize=9.3,
                              leading=13.6, textColor=INK, leftIndent=10, rightIndent=8,
                              backColor=CALLBG, borderColor=CALLBAR, borderWidth=0,
                              borderPadding=8, spaceBefore=7, spaceAfter=9),
    "cell": ParagraphStyle("cell", parent=ss["Normal"], fontName="Helvetica", fontSize=8.3,
                           leading=10.8, textColor=INK),
    "cellh": ParagraphStyle("cellh", parent=ss["Normal"], fontName="Helvetica-Bold",
                            fontSize=8.3, leading=10.8, textColor=INK),
}


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(t: str) -> str:
    """Escape XML then apply the inline markup, protecting code spans from bold/italic."""
    codes: list[str] = []

    def stash(m):
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    t = re.sub(r"`([^`]+)`", stash, t)
    t = esc(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", t)
    t = re.sub(r"\^\{([^}]*)\}", r"<super>\1</super>", t)

    def pop(m):
        return ('<font face="Courier" size="8.6" color="#20304a">'
                + esc(codes[int(m.group(1))]) + "</font>")

    return re.sub(r"\x00(\d+)\x00", pop, t)


def make_table(rows: list[list[str]], width: float):
    header = rows[0]
    body = rows[1:]
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in ([header] + body)]
    # first column gets more room when it reads like a label
    if ncol > 2:
        w0 = width * min(0.34, max(0.16, 0.42 - 0.03 * ncol))
        rest = (width - w0) / (ncol - 1)
        widths = [w0] + [rest] * (ncol - 1)
    else:
        widths = [width / ncol] * ncol
    data = [[Paragraph(inline(c), S["cellh"] if i == 0 else S["cell"]) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEADBG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#9a9a9a")),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#9a9a9a")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafaf9")]),
    ]))
    return t


def parse(text: str, width: float):
    story = []
    lines = text.split("\n")
    i, first_h1 = 0, True
    buf: list[str] = []

    def flush_para():
        nonlocal buf
        if buf:
            story.append(Paragraph(inline(" ".join(buf)), S["body"]))
            buf = []

    while i < len(lines):
        ln = lines[i]
        st = ln.strip()

        if not st:
            flush_para(); i += 1; continue

        if st.startswith("```"):
            flush_para(); i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i]); i += 1
            i += 1
            story.append(Paragraph("<br/>".join(esc(b) or "&nbsp;" for b in block), S["code"]))
            continue

        if st.startswith("$$"):
            flush_para(); i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("$$"):
                block.append(lines[i]); i += 1
            i += 1
            story.append(Paragraph("<br/>".join(esc(b) or "&nbsp;" for b in block), S["formula"]))
            continue

        if st.startswith(">"):
            flush_para()
            block = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                block.append(lines[i].strip()[1:].strip()); i += 1
            merged, cur = [], []
            for b in block:
                if b:
                    cur.append(b)
                else:
                    merged.append(" ".join(cur)); cur = []
            if cur:
                merged.append(" ".join(cur))
            story.append(Paragraph("<br/><br/>".join(inline(m) for m in merged), S["callout"]))
            continue

        if st.startswith("|"):
            flush_para()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") and c for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                story.append(Spacer(1, 3))
                story.append(make_table(rows, width))
                story.append(Spacer(1, 8))
            continue

        m = re.match(r"^(\d+)\.\s+(.*)", st)
        if st.startswith(("- ", "* ")) or m:
            flush_para()
            if m:
                story.append(Paragraph(inline(m.group(2)), S["bullet"], bulletText=m.group(1) + "."))
            else:
                story.append(Paragraph(inline(st[2:]), S["bullet"], bulletText="•"))
            i += 1; continue

        if st.startswith("### "):
            flush_para(); story.append(Paragraph(inline(st[4:]), S["h3"])); i += 1; continue
        if st.startswith("## "):
            flush_para(); story.append(Paragraph(inline(st[3:]), S["h2"])); i += 1; continue
        if st.startswith("# "):
            flush_para()
            if not first_h1:
                story.append(PageBreak())
            first_h1 = False
            story.append(Paragraph(inline(st[2:]), S["h1"]))
            i += 1; continue

        buf.append(st); i += 1

    flush_para()
    return story


class Doc(BaseDocTemplate):
    def __init__(self, path, **kw):
        super().__init__(path, pagesize=A4, leftMargin=20 * mm, rightMargin=18 * mm,
                         topMargin=17 * mm, bottomMargin=17 * mm, **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="n")
        self.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=self.decorate)])

    def decorate(self, canv, doc):
        canv.saveState()
        canv.setFont("Helvetica", 7.5)
        canv.setFillColor(MUTED)
        canv.drawString(self.leftMargin, 10 * mm,
                        "MrVLA — Causal Generality in VLA Sparse Autoencoders")
        canv.drawRightString(self.leftMargin + self.width, 10 * mm, str(doc.page))
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.4)
        canv.line(self.leftMargin, 12.5 * mm, self.leftMargin + self.width, 12.5 * mm)
        canv.restoreState()


def main(out, srcs):
    doc = Doc(out)
    story = []
    for s in srcs:
        with open(s) as f:
            story += parse(f.read(), doc.width)
    doc.build(story)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
