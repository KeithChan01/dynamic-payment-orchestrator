from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st

from dpo.config_loader import load_all
from dpo.orchestrator import Orchestrator
from dpo.paths import models_dir
from dpo.schemas import DISCLAIMER, RouteRequest
from dpo.simulate import random_requests, summarize

st.set_page_config(page_title="DPO · Payment routing", layout="wide")

STYLES = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")

SCENARIOS = {
    "Domestic AUD deposit": dict(
        transaction_type="deposit",
        amount=150.0,
        currency="AUD",
        country="Australia",
        policy="balanced",
    ),
    "Large USD cross-border": dict(
        transaction_type="deposit",
        amount=8200.0,
        currency="USD",
        local_currency="EUR",
        country="United States",
        ip_country="Germany",
        is_cross_border=True,
        policy="cost",
    ),
    "Weekend JPY withdrawal": dict(
        transaction_type="withdrawal",
        amount=2400.0,
        currency="JPY",
        country="Japan",
        policy="speed",
    ),
    "SGD PayPal-sized deposit": dict(
        transaction_type="deposit",
        amount=12500.0,
        currency="SGD",
        country="Singapore",
        policy="approval",
    ),
}

POLICY_BLURB = {
    "balanced": "Mix of success, fee, and speed",
    "cost": "Prefer the cheaper eligible route",
    "speed": "Prefer faster settlement",
    "approval": "Prefer the higher success chance",
}


@st.cache_resource
def engine() -> Orchestrator:
    return Orchestrator()


def _cfg():
    return load_all()


def _metrics() -> dict | None:
    path = models_dir() / "metrics.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def inject_chrome(mode: str) -> None:
    st.markdown(f"<style>{STYLES}</style>", unsafe_allow_html=True)
    st.sidebar.markdown(
        """
<div class="dpo-brand">
  <div class="dpo-mark"></div>
  <div>
    <div class="dpo-brand-name">DPO</div>
    <div class="dpo-brand-sub">Payment routing</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f'<div class="dpo-chip"><span class="dpo-chip-dot"></span>'
        f'{html.escape(mode)} models</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<div class="dpo-navhint">Navigate</div>', unsafe_allow_html=True)


def minutes(value: float) -> str:
    if value < 90:
        return f"{value:.1f} min"
    if value < 1440:
        return f"{value / 60:.1f} hr"
    return f"{value / 1440:.1f} d"


def money(value: float, currency: str) -> str:
    return f"{value:,.2f} {html.escape(currency)}"


def header() -> None:
    st.markdown(
        """
<div class="dpo-hero">
  <div>
    <div class="dpo-kicker">Route intelligence</div>
    <h1 class="dpo-title">Which provider should we try first?</h1>
    <p class="dpo-lede">Eligible rails are scored on success chance, fee, and settlement time. The first pick is the ranking winner; the rest are backups.</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="dpo-banner">{html.escape(DISCLAIMER)}</div>', unsafe_allow_html=True)


def render_result(result, currency: str) -> None:
    if not result.ranked:
        st.info("No eligible routes for this payment.")
        return
    top = result.ranked[0]
    labels = {row.route: row.label for row in result.ranked}
    fallbacks = " · ".join(html.escape(labels.get(name, name)) for name in result.fallbacks) or "None"
    st.markdown(
        f"""
<div class="dpo-card dpo-winner">
  <h3>Recommended</h3>
  <div class="dpo-winner-name">{html.escape(top.label)}</div>
  <div class="dpo-winner-why">{html.escape(" · ".join(top.reasons[:3]))}</div>
  <div class="dpo-stats">
    <div class="dpo-stat"><span>Success</span><strong>{top.success_probability:.1%}</strong></div>
    <div class="dpo-stat"><span>Fee</span><strong>{money(top.fee_amount, currency)}</strong></div>
    <div class="dpo-stat"><span>P90 arrival</span><strong>{minutes(top.settlement_p90_minutes)}</strong></div>
  </div>
  <div class="dpo-pills">
    <span class="dpo-pill">Policy {html.escape(result.policy)}</span>
    <span class="dpo-pill">Backups {fallbacks}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    rows = [
        '<div class="dpo-card" style="margin-top:0.85rem">',
        "<h3>Ranked routes</h3>",
        '<div class="dpo-row head"><div class="dpo-row-main"><span>#</span><span>Provider</span>'
        "<span>Success</span><span>Fee</span><span>P90</span></div></div>",
    ]
    ceiling = max(row.score for row in result.ranked) or 1
    for index, row in enumerate(result.ranked, start=1):
        width = max(8, round(100 * row.score / ceiling))
        klass = "dpo-row top" if index == 1 else "dpo-row"
        rows.append(
            f'<div class="{klass}">'
            f'<div class="dpo-row-main">'
            f'<span class="dpo-rank">{index}</span>'
            f'<span class="dpo-name">{html.escape(row.label)}</span>'
            f'<span class="dpo-mono">{row.success_probability:.1%}</span>'
            f'<span class="dpo-mono">{row.fee_amount:,.2f}</span>'
            f'<span class="dpo-mono">{minutes(row.settlement_p90_minutes)}</span>'
            f"</div>"
            f'<div class="dpo-bar"><div class="dpo-fill" style="--w:{width}%"></div></div>'
            f"</div>"
        )
    rows.append("</div>")
    st.markdown("\n".join(rows), unsafe_allow_html=True)

    if result.ineligible:
        blocked = [
            '<div class="dpo-card" style="margin-top:0.85rem"><h3>Not eligible</h3>'
        ]
        for row in result.ineligible:
            blocked.append(
                f'<div class="dpo-ineligible"><b>{html.escape(row.label)}</b>'
                f'<span class="dpo-muted">{html.escape(row.ineligible_reason or "")}</span></div>'
            )
        blocked.append("</div>")
        st.markdown("\n".join(blocked), unsafe_allow_html=True)


def _apply_scenario(scenario: str) -> None:
    preset = SCENARIOS.get(scenario, {})
    st.session_state.tx_type = preset.get("transaction_type", "deposit")
    st.session_state.amount = float(preset.get("amount", 480.0))
    st.session_state.currency = preset.get("currency", "USD")
    st.session_state.country = preset.get("country", "United States")
    st.session_state.ip_country = preset.get("ip_country", st.session_state.country)
    st.session_state.policy = preset.get("policy", "balanced")
    st.session_state.cross_border = bool(preset.get("is_cross_border", False))
    st.session_state.local_currency = preset.get(
        "local_currency", preset.get("currency", "USD")
    )


def page_route():
    cfg = _cfg()
    orch = engine()
    form, result_col = st.columns((0.92, 1.35), gap="large")
    with form:
        st.markdown('<div class="dpo-kicker">Payment</div>', unsafe_allow_html=True)
        scenario = st.selectbox("Scenario", ["Custom"] + list(SCENARIOS), key="scenario")
        if st.session_state.get("_applied_scenario") != scenario:
            _apply_scenario(scenario)
            st.session_state._applied_scenario = scenario
        c1, c2 = st.columns(2)
        tx_type = c1.selectbox("Type", ["deposit", "withdrawal"], key="tx_type")
        amount = c2.number_input("Amount", min_value=10.0, max_value=1_000_000.0, key="amount")
        c3, c4 = st.columns(2)
        currency = c3.selectbox("Currency", cfg["routes"]["currencies"], key="currency")
        local_currency = c4.selectbox(
            "Payout currency", cfg["routes"]["currencies"], key="local_currency"
        )
        c5, c6 = st.columns(2)
        country = c5.selectbox("Country", cfg["routes"]["countries"], key="country")
        ip_country = c6.selectbox("IP country", cfg["routes"]["countries"], key="ip_country")
        policy = st.selectbox("Policy", list(cfg["ranking"]["policies"]), key="policy")
        st.caption(POLICY_BLURB.get(policy, ""))
        cross = st.checkbox("Cross-border payment", key="cross_border")
        request = RouteRequest(
            transaction_type=tx_type,
            amount=amount,
            currency=currency,
            local_currency=local_currency,
            country=country,
            ip_country=ip_country,
            is_cross_border=cross,
            policy=policy,
        )
    with result_col:
        render_result(orch.evaluate(request), currency)


def page_compare():
    orch = engine()
    st.markdown(
        '<p class="dpo-muted">Same payment, four policies: $8,200 USD deposit, payer in Germany, payout EUR.</p>',
        unsafe_allow_html=True,
    )
    request = RouteRequest(
        transaction_type="deposit",
        amount=8200,
        currency="USD",
        local_currency="EUR",
        country="United States",
        ip_country="Germany",
        is_cross_border=True,
    )
    compared = orch.compare(request)
    cards = ['<div class="dpo-grid4">']
    for name, result in compared.items():
        top = result.ranked[0] if result.ranked else None
        pick = html.escape(result.recommended_label or "None")
        if top:
            body = (
                f'<div class="dpo-stat"><span>Success</span><strong>{top.success_probability:.1%}</strong></div>'
                f'<div class="dpo-stat"><span>Fee</span><strong>{top.fee_amount:,.2f} USD</strong></div>'
                f'<div class="dpo-stat"><span>P90</span><strong>{minutes(top.settlement_p90_minutes)}</strong></div>'
            )
        else:
            body = ""
        cards.append(
            f'<div class="dpo-card dpo-policy"><div class="label">{html.escape(name)}</div>'
            f'<div class="pick">{pick}</div>{body}</div>'
        )
    cards.append("</div>")
    st.markdown("\n".join(cards), unsafe_allow_html=True)

    st.markdown("#### Simulated day book")
    st.caption("200 synthetic payments. Orchestrator vs always-Stripe, cheapest, and fastest.")
    if st.button("Run simulation", type="primary"):
        with st.spinner("Scoring routes…"):
            summary = summarize(orch, random_requests(200, seed=7), policy="balanced")
        labels = {
            "orchestrator": "Orchestrator",
            "always_stripe": "Always Stripe",
            "cheapest": "Cheapest eligible",
            "fastest": "Fastest eligible",
        }
        sim_cards = ['<div class="dpo-grid4">']
        for key, row in summary["strategies"].items():
            sim_cards.append(
                f'<div class="dpo-card dpo-policy"><div class="label">{labels.get(key, key)}</div>'
                f'<div class="pick">{row.get("mean_success", 0):.1%}</div>'
                f'<div class="dpo-stat"><span>Avg fee</span><strong>{row.get("mean_fee", 0):,.2f}</strong></div>'
                f'<div class="dpo-stat"><span>Avg P90</span><strong>{minutes(row.get("mean_p90_minutes", 0))}</strong></div>'
                f"</div>"
            )
        sim_cards.append("</div>")
        st.markdown("\n".join(sim_cards), unsafe_allow_html=True)
        st.caption("Synthetic features, not live traffic.")


def page_data():
    st.markdown(
        '<p class="dpo-muted">Three Hugging Face tables, loaded separately because they do not share a schema.</p>',
        unsafe_allow_html=True,
    )
    tables = [
        ("payment_method", "Which provider fits this payment?"),
        ("success", "Will that provider succeed?"),
        ("arrival", "How long until the money arrives?"),
    ]
    cols = st.columns(3)
    for col, (name, task) in zip(cols, tables):
        col.markdown(
            f'<div class="dpo-card"><h3>{html.escape(name)}</h3>'
            f'<div class="dpo-winner-why">{html.escape(task)}</div></div>',
            unsafe_allow_html=True,
        )
    metrics = _metrics()
    if not metrics:
        st.info("No trained bundle yet. The app is using heuristic priors. Run `dpo-train`.")
        return
    route = metrics.get("route", {})
    success = metrics.get("success", {})
    settle = metrics.get("settlement", {})
    st.markdown(
        f"""
<div class="dpo-grid4" style="margin-top:0.9rem">
  <div class="dpo-card dpo-policy"><div class="label">Route accuracy</div>
    <div class="pick">{route.get("accuracy", 0):.0%}</div>
    <span class="dpo-muted">majority baseline {route.get("majority_baseline_accuracy", 0):.0%}</span></div>
  <div class="dpo-card dpo-policy"><div class="label">Success AUC</div>
    <div class="pick">{success.get("roc_auc", 0):.2f}</div>
    <span class="dpo-muted">coin-flip is 0.50</span></div>
  <div class="dpo-card dpo-policy"><div class="label">Settlement MAE</div>
    <div class="pick">{settle.get("mae_p50", 0):.0f} min</div>
    <span class="dpo-muted">median dummy {settle.get("mae_median_baseline", 0):.0f}</span></div>
  <div class="dpo-card dpo-policy"><div class="label">Training rows</div>
    <div class="pick">{metrics.get("rows", {}).get("success", 0):,}</div>
    <span class="dpo-muted">older dates train, newer dates test</span></div>
</div>
""",
        unsafe_allow_html=True,
    )


def main():
    orch = engine()
    inject_chrome(orch.mode)
    page = st.sidebar.radio("Page", ["Route", "Compare", "Data"], label_visibility="collapsed")
    header()
    if page == "Route":
        page_route()
    elif page == "Compare":
        page_compare()
    else:
        page_data()


main()
