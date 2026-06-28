from featfuse.report import latex_table, markdown_table


ROWS = [
    {"method": "A", "f1": 0.90, "ece": 0.10},
    {"method": "B", "f1": 0.95, "ece": 0.05},
]


def test_markdown_table_has_header_and_rows():
    md = markdown_table(ROWS, ["f1", "ece"])
    assert "| method | f1 | ece |" in md
    assert md.count("\n") == 3  # header, sep, 2 rows


def test_latex_bolds_best_per_column():
    tex = latex_table(ROWS, ["f1", "ece"])
    # best f1 is higher (B), best ece is lower (B)
    assert "\\textbf{0.9500}" in tex
    assert "\\textbf{0.0500}" in tex
    assert "booktabs" not in tex  # we emit \toprule etc., not the package import
    assert "\\toprule" in tex
