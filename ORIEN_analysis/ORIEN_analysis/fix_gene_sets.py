#!/usr/bin/env python3
"""
Fix invalid gene_sets.json where list items are newline-separated, unquoted,
and lack commas. Converts:

    "CD8_1": [
        SPC25
        CDCA5
        ...
    ],

to:

    "CD8_1": [
        "SPC25",
        "CDCA5"
    ],

Usage:
    python fix_gene_sets.py gene_sets.json  # writes gene_sets.fixed.json
"""

from pathlib import Path
import json
import re
import sys


def fix_gene_sets_text(text: str) -> str:
    out_lines = []
    in_list = False
    items = []
    open_line = ""
    item_indent = None

    lines = text.splitlines()

    for line in lines:
        stripped = line.strip()

        if not in_list:
            out_lines.append(line)
            # Enter list mode when a line ends with '['
            if stripped.endswith('['):
                in_list = True
                items = []
                item_indent = None
                open_line = out_lines.pop()  # we'll re-insert before items
        else:
            # If we reached the closing bracket, flush items
            if stripped.startswith(']'):
                # Preserve a trailing comma after ']' if present
                trailing_comma = ',' if stripped.endswith('],') else ''
                closing_indent = re.match(r'^(\s*)', line).group(1)

                # If we never saw an item line to infer indent, use closing indent + 4 spaces
                if item_indent is None:
                    item_indent = closing_indent + '    '

                # Build quoted item lines; last one has no trailing comma
                quoted = [f'{item_indent}"{tok}",' for tok in items]
                if quoted:
                    quoted[-1] = quoted[-1].rstrip(',')

                # Reconstruct: opening line with '[', then items, then closing bracket
                out_lines.append(open_line)
                out_lines.extend(quoted)
                out_lines.append(f'{closing_indent}]{trailing_comma}')

                # Reset state
                in_list = False
                items = []
                open_line = ""
                item_indent = None
            else:
                # Accumulate a token line (ignore blanks and placeholder ellipses)
                if stripped:
                    if item_indent is None:
                        item_indent = re.match(r'^(\s*)', line).group(1)
                    tok = stripped.strip(',').strip('"')
                    if tok not in ('...', '....'):
                        items.append(tok)

    return '\n'.join(out_lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_gene_sets.py gene_sets.json")
        sys.exit(2)

    in_path = Path(sys.argv[1]).expanduser()
    out_path = in_path.with_suffix(in_path.suffix + ".fixed.json")

    text = in_path.read_text(encoding="utf-8")
    fixed = fix_gene_sets_text(text)

    # Validate JSON (raises if still invalid)
    json.loads(fixed)

    out_path.write_text(fixed, encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()