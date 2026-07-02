"""Parse a single MD trace file into structured JSON for run_eval.py."""
import re
import json
import sys


def parse_trace_md(md_text: str) -> dict:
    """Parse a conversation trace markdown file into structured JSON."""

    # ── Extract all turns ──
    turn_blocks = re.split(r'###\s+Turn\s+\d+', md_text)
    turn_blocks = [b for b in turn_blocks if b.strip()]  # drop preamble

    user_messages = []
    all_tables = []  # list of list-of-row-dicts, one per table found

    for block in turn_blocks:
        # Extract user message (blockquote lines after **User**)
        user_match = re.search(
            r'\*\*User\*\*\s*\n((?:\s*>.*\n?)+)', block
        )
        if user_match:
            raw_quote = user_match.group(1)
            # Strip leading > and whitespace from each line
            lines = []
            for line in raw_quote.strip().split('\n'):
                cleaned = re.sub(r'^\s*>\s?', '', line)
                lines.append(cleaned)
            user_messages.append('\n'.join(lines).strip())

        # Extract markdown tables from this turn
        # A table is a sequence of lines starting with |
        table_lines = []
        in_table = False
        for line in block.split('\n'):
            stripped = line.strip()
            if stripped.startswith('|') and stripped.endswith('|'):
                table_lines.append(stripped)
                in_table = True
            else:
                if in_table and table_lines:
                    # End of table block — parse it
                    parsed = _parse_markdown_table(table_lines)
                    if parsed:
                        all_tables.append(parsed)
                    table_lines = []
                    in_table = False
        # Handle table at end of block
        if table_lines:
            parsed = _parse_markdown_table(table_lines)
            if parsed:
                all_tables.append(parsed)

    # ── Build expected_shortlist from LAST table ──
    expected_shortlist = []
    if all_tables:
        last_table = all_tables[-1]
        for row in last_table:
            name = row.get('Name', '').strip()
            if name:
                expected_shortlist.append(name)

    # ── Build facts from all user messages ──
    facts = _extract_facts(user_messages)

    # ── Build opening_message ──
    opening_message = user_messages[0] if user_messages else ""

    # ── Build persona ──
    persona = opening_message

    return {
        "persona": persona,
        "facts": facts,
        "expected_shortlist": expected_shortlist,
        "opening_message": opening_message,
    }


def _parse_markdown_table(lines: list[str]) -> list[dict]:
    """Parse markdown table lines into list of row dicts.
    
    Handles empty cells (—, blank) correctly by splitting on | delimiters.
    """
    if len(lines) < 3:  # Need header + separator + at least one data row
        return []

    # Parse header
    header_cells = [c.strip() for c in lines[0].split('|')]
    # Remove empty strings from leading/trailing |
    header_cells = [c for c in header_cells if c or c == '']
    # Filter out truly empty boundary cells
    header_cells = lines[0].split('|')[1:-1]  # skip first and last empty splits
    header_cells = [c.strip() for c in header_cells]

    # Skip separator line (lines[1])
    # Parse data rows
    rows = []
    for line in lines[2:]:
        cells = line.split('|')[1:-1]  # skip boundary empties
        cells = [c.strip() for c in cells]
        row = {}
        for i, header in enumerate(header_cells):
            if i < len(cells):
                row[header] = cells[i]
            else:
                row[header] = ''
        rows.append(row)

    return rows


def _extract_facts(user_messages: list[str]) -> dict:
    """Extract structured facts from all user messages."""
    all_text = ' '.join(user_messages)

    # Basic extraction — role, seniority, skills, constraints
    facts = {
        "role": "",
        "seniority": "",
        "skills": [],
        "constraints": [],
    }

    # Use all user messages after the first as constraints/refinements
    if len(user_messages) > 1:
        for msg in user_messages[1:]:
            facts["constraints"].append(msg)

    # Extract role hints from first message
    first = user_messages[0] if user_messages else ""
    facts["role"] = first

    return facts


if __name__ == "__main__":
    import glob
    import os

    md_dir = "eval/traces/GenAI_SampleConversations"
    out_dir = "eval/traces"

    md_files = sorted(glob.glob(os.path.join(md_dir, "C*.md")))
    if not md_files:
        print(f"No C*.md files found in {md_dir}")
        sys.exit(1)

    print(f"{'Filename':<20} {'Expected Shortlist Items':>25}")
    print("-" * 47)

    for md_path in md_files:
        basename = os.path.splitext(os.path.basename(md_path))[0]
        out_path = os.path.join(out_dir, f"{basename}.json")

        with open(md_path, 'r', encoding='utf-8') as f:
            md_text = f.read()

        result = parse_trace_md(md_text)

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        count = len(result["expected_shortlist"])
        print(f"{basename + '.json':<20} {count:>25}")

    print(f"\nGenerated {len(md_files)} JSON trace files in {out_dir}/")
