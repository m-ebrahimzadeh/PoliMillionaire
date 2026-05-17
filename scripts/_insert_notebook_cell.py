"""Insert post-game summary cells into PoliMillionaire.ipynb after cell 50."""
import json
from pathlib import Path

nb_path = Path(__file__).parent.parent / "PoliMillionaire.ipynb"

with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

# Guard: don't insert twice
existing_ids = [c.get("id", "") for c in nb["cells"]]
if "post_game_summary_md" in existing_ids:
    print("Cells already present — nothing to do.")
else:
    markdown_cell = {
        "cell_type": "markdown",
        "id": "post_game_summary_md",
        "metadata": {},
        "source": [
            "### 2.6.1 Post-game retrieval summary (live games only)\n",
            "\n",
            "After each live game session, this cell prints a full diagnostic breakdown:\n",
            "context source used (offline RAG / live search / no context), per-category accuracy,\n",
            "difficulty curve, live-search efficiency, confidence calibration, and worst misses.\n",
            "\n",
            "Requires `RUN_LIVE_GAMES = True` in the cell above.",
        ],
    }

    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "post_game_summary_code",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Post-game retrieval summary — one report per competition played.\n",
            "# Needs RUN_LIVE_GAMES=True above; skips gracefully otherwise.\n",
            "from polimibot.observability import print_game_summary\n",
            "\n",
            "if RUN_LIVE_GAMES and results:\n",
            "    for r in results:\n",
            "        print_game_summary(r)\n",
            "else:\n",
            "    print('Skipping post-game summary (RUN_LIVE_GAMES=False or no results).')",
        ],
    }

    nb["cells"].insert(51, code_cell)
    nb["cells"].insert(51, markdown_cell)

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"Done. Total cells: {len(nb['cells'])}")
    print("Cell 51 (md)  :", repr("".join(nb["cells"][51]["source"])[:70]))
    print("Cell 52 (code):", repr("".join(nb["cells"][52]["source"])[:70]))
