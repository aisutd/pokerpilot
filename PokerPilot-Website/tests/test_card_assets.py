import re
import unittest
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[2]


class CardAssetTests(unittest.TestCase):
    def test_svg_deck_has_all_52_faces_and_a_back(self):
        root = ElementTree.parse(ROOT / "Assets" / "Cards" / "deck.svg").getroot()
        symbols = {node.attrib.get("id") for node in root.iter() if node.tag.endswith("symbol")}
        expected = {f"card-{rank}{suit}" for rank in "A23456789TJQK" for suit in "shdc"}
        self.assertEqual(symbols, expected | {"card-back"})
        self.assertEqual(root.attrib["viewBox"], "0 0 250 350")

    def test_both_renderers_keep_face_down_empty_and_accessible_cards(self):
        for app in ("PokerPilot-Website", "Multiplayer-Mode"):
            source = (ROOT / app / "static" / "app.js").read_text(encoding="utf-8")
            self.assertIn("appendCardArt(el, 'card-back')", source)
            self.assertIn("classList.add('slot')", source)
            self.assertIn("setAttribute('aria-label', 'Face-down card')", source)
            self.assertRegex(source, re.compile(r"/cards/deck\.svg#\$\{symbol\}"))


if __name__ == "__main__":
    unittest.main()
