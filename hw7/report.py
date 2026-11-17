import re
from html import escape
from pathlib import Path


def main() -> None:
    from matplotlib.font_manager import FontProperties, findfont
    from PIL import Image as PILImage
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

    pdfmetrics.registerFont(TTFont('Body', findfont(FontProperties(family='DejaVu Sans'))))
    styles = {name: ParagraphStyle(name, fontName='Body', fontSize=size, leading=size * 1.4,
                                  textColor=colors.HexColor('#17313f'), spaceAfter=8)
              for name, size in [('body', 9), ('small', 7.6), ('title', 23), ('heading', 14)]}
    story = []

    def text(value, style='body'):
        value = escape(value).replace('`', '').replace('**', '')
        value = re.sub(r'\[([^]]+)\]\((https?://[^)]+)\)', r'<link href="\2" color="#1e5a78">\1</link>', value)
        value = re.sub(r'\[([^]]+)\]\([^)]+\)', r'\1', value)
        story.append(Paragraph(value, styles[style]))

    def table(rows):
        assert rows and all(len(row) == len(rows[0]) for row in rows)
        weights = [2] + [1] * (len(rows[0]) - 1)
        widths = [505 * weight / sum(weights) for weight in weights]
        cells = [[Paragraph(escape(str(cell)).replace('`', ''), styles['small']) for cell in row] for row in rows]
        grid = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
        grid.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dce8ee')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f6f7')]),
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.HexColor('#1e5a78')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.extend([grid, Spacer(1, 10)])

    lines = Path('REPORT.md').read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line.startswith('# '):
            text(line[2:], 'title')
        elif line.startswith('## '):
            story.append(PageBreak())
            text(line[3:], 'heading')
        elif line.startswith('### '):
            text(line[4:], 'heading')
        elif line.startswith('!['):
            path = re.fullmatch(r'!\[[^]]*\]\(([^)]+)\)', line)[1]
            with PILImage.open(path) as picture:
                width, height = picture.size
            story.extend([Image(path, width=490, height=490 * height / width), Spacer(1, 8)])
        elif line.startswith('|'):
            rows = [line.strip('|').split('|')]
            while i < len(lines) and lines[i].startswith('|'):
                if not re.fullmatch(r'[| :\-]+', lines[i]):
                    rows.append(lines[i].strip('|').split('|'))
                i += 1
            table([[cell.strip() for cell in row] for row in rows])
        elif line.startswith('```'):
            code = []
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(lines[i])
                i += 1
            story.append(Preformatted('\n'.join(code), styles['small'], maxLineLength=100))
            i += 1
        else:
            text(line)

    def decorate(canvas, document):
        canvas.setFont('Body', 7.5)
        canvas.setFillColor(colors.HexColor('#6c7b83'))
        canvas.drawString(45, 22, 'ДЗ7 · Qwen3 · MASSIVE ru-RU')
        canvas.drawRightString(550, 22, str(document.page))

    Path('outputs').mkdir(exist_ok=True)
    SimpleDocTemplate('outputs/report.pdf', pagesize=(595.28, 841.89), rightMargin=45,
                      leftMargin=45, topMargin=38, bottomMargin=38).build(story, onFirstPage=decorate, onLaterPages=decorate)


if __name__ == '__main__':
    main()
