import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="MASTER Risk Assessment",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Shared CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font import ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 760px; }

/* ── Brand palette ── */
:root {
    --navy:   #1B2A4A;
    --gold:   #F4A225;
    --coral:  #E85D4A;
    --teal:   #2BBCB5;
    --bg:     #F5F7FA;
    --card:   #FFFFFF;
    --text:   #2D3748;
    --muted:  #718096;
    --border: #E2E8F0;
}

/* ── App background ── */
.stApp { background-color: var(--bg); }

/* ── Header banner ── */
.master-header {
    background: linear-gradient(135deg, #1B2A4A 0%, #243660 60%, #2d4a7a 100%);
    border-radius: 16px;
    padding: 2.5rem 2.5rem 2rem;
    margin-bottom: 2rem;
    color: white;
    box-shadow: 0 8px 32px rgba(27,42,74,0.18);
}
.master-header .badge {
    display: inline-block;
    background: rgba(244,162,37,0.18);
    border: 1px solid rgba(244,162,37,0.5);
    color: #F4A225;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 0.9rem;
}
.master-header h1 {
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0 0 0.4rem;
    letter-spacing: -0.02em;
    line-height: 1.15;
}
.master-header .subtitle {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.72);
    margin: 0;
    font-weight: 400;
    line-height: 1.5;
}

/* ── Progress stepper ── */
.stepper-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    margin-bottom: 2rem;
}
.step-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
}
.step-circle {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 700;
    border: 2px solid var(--border);
    background: white;
    color: var(--muted);
    transition: all 0.3s;
}
.step-circle.active {
    background: var(--navy);
    border-color: var(--navy);
    color: white;
    box-shadow: 0 0 0 4px rgba(27,42,74,0.12);
}
.step-circle.done {
    background: var(--teal);
    border-color: var(--teal);
    color: white;
}
.step-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    white-space: nowrap;
}
.step-label.active { color: var(--navy); }
.step-label.done   { color: var(--teal); }
.step-connector {
    height: 2px;
    width: 80px;
    background: var(--border);
    margin-bottom: 22px;
    flex-shrink: 0;
}
.step-connector.done { background: var(--teal); }

/* ── Card ── */
.card {
    background: var(--card);
    border-radius: 14px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid var(--border);
}
.card-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--navy);
    margin-bottom: 0.25rem;
}
.card-subtitle {
    font-size: 0.85rem;
    color: var(--muted);
    margin-bottom: 1.25rem;
    line-height: 1.5;
}

/* ── Role cards (grid) ── */
.role-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 1rem;
}
.role-card {
    background: white;
    border: 2px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1rem;
    cursor: pointer;
    transition: all 0.2s;
    text-align: center;
}
.role-card:hover {
    border-color: var(--navy);
    box-shadow: 0 4px 16px rgba(27,42,74,0.1);
}
.role-card.selected {
    border-color: var(--navy);
    background: rgba(27,42,74,0.04);
}
.role-icon { font-size: 1.8rem; margin-bottom: 0.4rem; }
.role-name { font-weight: 700; font-size: 0.92rem; color: var(--navy); }
.role-desc { font-size: 0.75rem; color: var(--muted); margin-top: 3px; line-height: 1.4; }

/* ── Section divider ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--gold);
    margin: 1.5rem 0 0.75rem;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stRadio"] label {
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: var(--text) !important;
}
div.stButton > button {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.01em;
    padding: 0.65rem 2.2rem;
    border-radius: 10px;
    border: none;
    background: linear-gradient(135deg, #1B2A4A, #2d4a7a);
    color: white;
    box-shadow: 0 4px 14px rgba(27,42,74,0.25);
    transition: all 0.2s;
    width: 100%;
}
div.stButton > button:hover {
    box-shadow: 0 6px 20px rgba(27,42,74,0.35);
    transform: translateY(-1px);
}
div.stButton > button[kind="secondary"] {
    background: white;
    color: var(--navy);
    border: 2px solid var(--border);
    box-shadow: none;
}

/* ── Info callout ── */
.info-callout {
    background: rgba(43,188,181,0.08);
    border-left: 4px solid var(--teal);
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.1rem;
    font-size: 0.84rem;
    color: var(--text);
    line-height: 1.55;
    margin-bottom: 1.25rem;
}

/* ── Error / warning ── */
.warn-callout {
    background: rgba(232,93,74,0.08);
    border-left: 4px solid var(--coral);
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: var(--coral);
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session-state initialisation ──────────────────────────────────────────────
DEFAULTS = {
    "step": 1,
    "role": None,
    "background": {},
    "answers": {},
    "scores": {},
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Helpers ───────────────────────────────────────────────────────────────────
def go_to(step: int):
    st.session_state.step = step


def render_header():
    st.markdown("""
    <div class="master-header">
        <div class="badge">Risk Assessment</div>
        <h1>MASTER Risk Blueprint</h1>
        <p class="subtitle">
            Evaluate your readiness across Mindset, Abilities &amp; Situation —
            and receive a personalised action plan to grow in your risk zone.
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_stepper(current: int):
    steps = ["Background", "Assessment", "Blueprint"]
    icons  = ["✓", "✓", "✓"]
    html = '<div class="stepper-wrap">'
    for i, (label, icon) in enumerate(zip(steps, icons), start=1):
        if i < current:
            circle_cls = "done"
            label_cls  = "done"
            circle_txt = icon
        elif i == current:
            circle_cls = "active"
            label_cls  = "active"
            circle_txt = str(i)
        else:
            circle_cls = ""
            label_cls  = ""
            circle_txt = str(i)

        html += f"""
        <div class="step-item">
            <div class="step-circle {circle_cls}">{circle_txt}</div>
            <div class="step-label {label_cls}">{label}</div>
        </div>"""
        if i < len(steps):
            conn_cls = "done" if i < current else ""
            html += f'<div class="step-connector {conn_cls}"></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ── Role definitions ───────────────────────────────────────────────────────────
ROLES = {
    "Entrepreneur": {
        "icon": "🚀",
        "desc": "Founder, co-founder, or business owner",
    },
    "Manager": {
        "icon": "🏢",
        "desc": "Team lead, director, or senior executive",
    },
    "Employee": {
        "icon": "💼",
        "desc": "Individual contributor or specialist",
    },
    "Freelancer": {
        "icon": "🎨",
        "desc": "Independent consultant or contractor",
    },
}

# ── Follow-up questions per role ───────────────────────────────────────────────
FOLLOWUP = {
    "Entrepreneur": [
        {
            "key":   "biz_stage",
            "label": "What stage is your venture at?",
            "options": [
                "Idea / Pre-launch",
                "Early stage (0–2 years)",
                "Growth stage (2–5 years)",
                "Established (5+ years)",
            ],
        },
        {
            "key":   "team_size",
            "label": "How large is your team?",
            "options": ["Solo (just me)", "2–5 people", "6–20 people", "20+ people"],
        },
        {
            "key":   "primary_challenge",
            "label": "What is your biggest current challenge?",
            "options": [
                "Finding or keeping customers",
                "Funding and cash flow",
                "Building and leading the team",
                "Scaling operations",
            ],
        },
    ],
    "Manager": [
        {
            "key":   "org_type",
            "label": "What type of organisation do you work in?",
            "options": [
                "Large corporation (500+ employees)",
                "Mid-size company (50–500 employees)",
                "Small business (<50 employees)",
                "Non-profit / Public sector",
            ],
        },
        {
            "key":   "team_size",
            "label": "How many direct reports do you manage?",
            "options": ["None (individual contributor + title)", "1–5", "6–15", "16+"],
        },
        {
            "key":   "mgmt_level",
            "label": "What best describes your management level?",
            "options": [
                "Team Lead / Supervisor",
                "Mid-level Manager",
                "Senior Manager / Director",
                "VP / C-suite / Executive",
            ],
        },
    ],
    "Employee": [
        {
            "key":   "career_stage",
            "label": "Where are you in your career?",
            "options": [
                "Early career (0–3 years)",
                "Mid career (3–10 years)",
                "Senior / Expert (10+ years)",
                "Career changer",
            ],
        },
        {
            "key":   "employment_status",
            "label": "What is your current employment situation?",
            "options": [
                "Happily employed, seeking growth",
                "Employed but exploring options",
                "Recently changed roles",
                "Between jobs / job seeking",
            ],
        },
        {
            "key":   "aspiration",
            "label": "What is your primary professional aspiration right now?",
            "options": [
                "Advance within my current organisation",
                "Move to a new company or industry",
                "Develop a new skill or specialisation",
                "Launch my own venture eventually",
            ],
        },
    ],
    "Freelancer": [
        {
            "key":   "freelance_stage",
            "label": "How established is your freelance practice?",
            "options": [
                "Just starting out (<1 year)",
                "Building momentum (1–3 years)",
                "Stable and growing (3+ years)",
                "Scaling or productising",
            ],
        },
        {
            "key":   "client_base",
            "label": "How would you describe your client situation?",
            "options": [
                "Still finding my first clients",
                "A few clients, working to diversify",
                "Steady roster, occasional gaps",
                "Fully booked, managing demand",
            ],
        },
        {
            "key":   "primary_challenge",
            "label": "What is your biggest current challenge?",
            "options": [
                "Attracting and converting clients",
                "Pricing and positioning",
                "Managing time and workload",
                "Building long-term stability",
            ],
        },
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Background Intake
# ══════════════════════════════════════════════════════════════════════════════
def render_step1():
    render_header()
    render_stepper(current=1)

    # ── Role selector ──────────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">What best describes your professional role?</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">Your role shapes which version of the assessment you receive, so we can make the insights as relevant as possible.</div>', unsafe_allow_html=True)

    cols = st.columns(2)
    role_names = list(ROLES.keys())

    for idx, role in enumerate(role_names):
        col = cols[idx % 2]
        info = ROLES[role]
        selected = st.session_state.role == role
        border_style = "border: 2px solid #1B2A4A; background: rgba(27,42,74,0.04);" if selected else "border: 2px solid #E2E8F0;"
        with col:
            st.markdown(f"""
            <div style="
                {border_style}
                border-radius: 12px;
                padding: 1.1rem 1rem;
                text-align: center;
                margin-bottom: 4px;
            ">
                <div style="font-size:1.9rem; margin-bottom:0.3rem;">{info['icon']}</div>
                <div style="font-weight:700; font-size:0.93rem; color:#1B2A4A;">{role}</div>
                <div style="font-size:0.74rem; color:#718096; margin-top:3px; line-height:1.4;">{info['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"{'✓ Selected' if selected else 'Select'}", key=f"role_btn_{role}"):
                st.session_state.role = role
                # Reset follow-up answers when role changes
                st.session_state.background = {"role": role}
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Follow-up questions (shown once role is selected) ──────────────────────
    if st.session_state.role:
        role = st.session_state.role
        questions = FOLLOWUP[role]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="card-title">A little more about your context</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="card-subtitle">These three questions help us calibrate the assessment and '
            f'personalise your action plan.</div>',
            unsafe_allow_html=True,
        )

        bg = st.session_state.background

        for q in questions:
            current_val = bg.get(q["key"])
            idx = q["options"].index(current_val) if current_val in q["options"] else 0
            answer = st.selectbox(
                q["label"],
                options=q["options"],
                index=idx,
                key=f"bg_{q['key']}",
            )
            bg[q["key"]] = answer

        # ── Industry / Sector (common to all roles) ────────────────────────────
        industries = [
            "Technology & Software",
            "Finance & Professional Services",
            "Healthcare & Life Sciences",
            "Education & Non-profit",
            "Creative & Media",
            "Retail & E-commerce",
            "Manufacturing & Engineering",
            "Consulting & Strategy",
            "Other",
        ]
        current_ind = bg.get("industry")
        ind_idx = industries.index(current_ind) if current_ind in industries else 0
        bg["industry"] = st.selectbox(
            "Which industry or sector are you in?",
            options=industries,
            index=ind_idx,
            key="bg_industry",
        )

        st.session_state.background = bg
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Info callout ───────────────────────────────────────────────────────
        st.markdown("""
        <div class="info-callout">
            <strong>What happens next:</strong> You'll answer around 30 questions across three pillars —
            <em>Mindset</em>, <em>Abilities</em>, and <em>Situation</em> — each rated on a 1–5 scale.
            The whole assessment takes about 8–12 minutes.
        </div>
        """, unsafe_allow_html=True)

        # ── CTA ───────────────────────────────────────────────────────────────
        all_answered = all(
            bg.get(q["key"]) for q in questions
        ) and bg.get("industry")

        if all_answered:
            if st.button("Start the Assessment →", key="go_step2"):
                st.session_state.background = bg
                go_to(2)
                st.rerun()
        else:
            st.markdown('<div class="warn-callout">Please answer all questions above before continuing.</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; color:#718096; font-size:0.88rem; padding:1rem 0;">
            ← Select your role above to continue
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Router
# ══════════════════════════════════════════════════════════════════════════════
def main():
    step = st.session_state.step
    if step == 1:
        render_step1()
    elif step == 2:
        # Placeholder — assessment coming next
        render_header()
        render_stepper(current=2)
        st.info("Step 2 — Assessment questions coming soon.")
        if st.button("← Back to Background"):
            go_to(1)
            st.rerun()
    elif step == 3:
        render_header()
        render_stepper(current=3)
        st.info("Step 3 — Results Blueprint coming soon.")


if __name__ == "__main__":
    main()
