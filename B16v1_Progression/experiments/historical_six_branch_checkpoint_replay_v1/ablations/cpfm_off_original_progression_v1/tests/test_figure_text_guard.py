import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
"""Anonymous regression tests: inactive ticks pass; true clipping still fails."""
import unittest

import draw_ablation_figures as drawings


class FigureTextGuardTests(unittest.TestCase):
    def tearDown(self):
        drawings.plt.close("all")

    def test_unrendered_ticks_do_not_fail(self):
        from matplotlib.ticker import FixedLocator
        fig, ax = drawings.plt.subplots(figsize=(4, 3))
        ax.set_xlim(0, .2)
        ax.xaxis.set_major_locator(FixedLocator([-5, 0, .1, .2, 5]))
        ax.plot([0, .1, .2], [0, 1, 0])
        qa = drawings.check_text_boundaries(fig)
        self.assertGreater(qa["rendered_text_count"], 0)
        self.assertEqual(ax.get_xlim(), (0, .2))

    def test_real_outside_text_still_fails(self):
        fig, _ = drawings.plt.subplots(figsize=(4, 3))
        fig.text(1.05, .5, "ACTUALLY_OUTSIDE", fontsize=7)
        with self.assertRaisesRegex(RuntimeError, "ACTUALLY_OUTSIDE"):
            drawings.check_text_boundaries(fig)

    def test_visible_figure_text_is_not_ignored(self):
        fig, _ = drawings.plt.subplots(figsize=(4, 3))
        label = fig.text(.5, .97, "VISIBLE_CONTROL", ha="center", va="top", fontsize=7)
        drawings.check_text_boundaries(fig)
        self.assertIn(label, list(drawings.displayed_texts(fig)))

    def test_disabled_axis_does_not_create_visible_ticks(self):
        fig, ax = drawings.plt.subplots(figsize=(4, 3))
        fig.subplots_adjust(left=.01)
        ax.set_ylim(0, 72)
        ax.axis("off")
        content = ax.text(.5, 30, "VISIBLE_SCHEMATIC", ha="center", fontsize=7)
        drawings.check_text_boundaries(fig)
        self.assertIn(content, list(drawings.displayed_texts(fig)))


if __name__ == "__main__":
    unittest.main()
