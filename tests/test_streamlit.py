from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'frontend' / 'streamlit_app.py'

def test_demo_views_and_reports():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == 'Fairness Lab'
    assert len(at.tabs) == 4
    assert any(m.label == 'Review risk' for m in at.metric)
    next(s for s in at.selectbox if s.label == 'Group comparison').set_value(2).run()
    assert not at.exception
    assert any('gender × age_group' in m.value for m in at.markdown)
    at.checkbox[0].check().run()
    next(b for b in at.button if b.label == 'Compare threshold candidates').click().run()
    assert not at.exception
    assert len(at.dataframe) >= 4
    at.radio[0].set_value('Public diabetes demo').run()
    assert not at.exception
    assert any('teaching example' in i.value for i in at.info)
    at.radio[0].set_value('Upload CSV').run()
    assert not at.exception
    assert any('Upload a CSV' in i.value for i in at.info)
