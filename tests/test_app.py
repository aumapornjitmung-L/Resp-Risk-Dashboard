"""ทดสอบว่าทุกหน้าเรนเดอร์ได้โดยไม่เกิด exception (โหมดค่าอ้างอิง ไม่ต้องมีข้อมูลจริง)"""
import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["dashboard", "models", "features", "explore", "quality", "predict"]


@pytest.mark.parametrize("mod", PAGES)
def test_page_renders_reference_mode(mod):
    code = f"import sys; sys.path.insert(0, '.')\nfrom respdash import ui\nfrom views import {mod}\nui.inject_css()\n{mod}.render()\n"
    at = AppTest.from_string(code, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]


def test_reference_json_consistent():
    from respdash.reference import load_reference
    ref = load_reference()
    assert ref is not None
    assert sum(ref["dataset"]["risk_counts"].values()) == ref["dataset"]["records"]
    assert sum(ref["dataset"]["test_counts"].values()) == ref["dataset"]["test"]
    for m, cm in ref["confusion"].items():
        assert sum(map(sum, cm)) == ref["dataset"]["test"], m
