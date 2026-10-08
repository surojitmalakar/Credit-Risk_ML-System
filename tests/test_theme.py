"""Regression tests for the shared design system in ``msme_ews.theme``.

The dashboard renders every navigation tab through one stylesheet and one
per-tab identity map. These tests lock that contract: all tabs are covered,
heroes render with their identity, risk-band colours stay on the brand ramp,
and no Python source ever leaks into the injected CSS.
"""

from __future__ import annotations

from msme_ews.theme import (
    BAND_COLOURS,
    CRITICAL,
    G4,
    HIGH,
    NEUTRAL,
    STYLESHEET,
    WATCH,
    about_block_html,
    about_tiles_html,
    band_colour,
    page_feature,
    page_hero_html,
    profile_card_html,
    site_footer_html,
)

EXPECTED_TABS = {
    "Home",
    "MSME Financial Health",
    "Risk Prediction",
    "Early-Warning Indicators",
    "Data Intelligence",
    "AI Assistant",
    "Reports",
    "Methodology",
    "About",
}


def test_every_tab_has_identity_with_required_fields() -> None:
    for tab in EXPECTED_TABS:
        feature = page_feature(tab)
        assert feature["icon"]
        assert feature["accent"]
        assert feature["tag"]
        assert feature["title"]
        assert feature["blurb"]
        assert feature["chips"]


def test_unknown_tab_falls_back_to_neutral_identity() -> None:
    feature = page_feature("No Such Tab")
    assert feature["title"] == "No Such Tab"
    assert feature["chips"] == []


def test_page_hero_renders_identity() -> None:
    html = page_hero_html("About")
    assert "page-hero" in html
    assert "Built by Surojit Malakar" in html
    assert "GitHub profile" in html


def test_band_colours_follow_brand_ramp() -> None:
    assert band_colour("Low Risk") == G4 == BAND_COLOURS["Low Risk"]
    assert band_colour("Moderate Risk") == WATCH
    assert band_colour("High Risk") == HIGH
    assert band_colour("Critical Risk") == CRITICAL
    assert band_colour("Unknown") == NEUTRAL


def test_stylesheet_has_no_python_leak_and_covers_app_classes() -> None:
    assert "return BAND_COLOURS" not in STYLESHEET
    assert STYLESHEET.strip().startswith("<style>")
    assert STYLESHEET.strip().endswith("</style>")
    for cls in (
        ".page-hero",
        ".kpi-card",
        ".risk-category-value",
        ".risk-summary-card",
        ".signal-indicator",
        ".assessment-card",
        ".profile-card",
        ".about-tile",
        ".site-footer",
    ):
        assert cls in STYLESHEET


def test_about_helpers_render() -> None:
    assert "profile-photo" in profile_card_html()
    assert "about-grid" in about_tiles_html()
    block = about_block_html("Heading", "Body copy")
    assert "Heading" in block and "Body copy" in block


def test_site_footer_carries_the_brand_and_the_disclaimer() -> None:
    footer = site_footer_html()
    assert 'class="site-footer"' in footer
    assert "SkillseED India" in footer
    assert "Beyond Knowledge Into Action" in footer
    assert "skillseedindia@gmail.com" in footer
    assert "skillseedindia.com" in footer
    assert "Surojit Malakar" in footer
    assert "not lending decisions" in footer
    assert "© 2026" in footer
