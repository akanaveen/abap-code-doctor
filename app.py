import os
import re
import json
import hashlib
from difflib import SequenceMatcher

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="ABAP Code Doctor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# openai/gpt-oss-120b is Groq's current recommended model (June 2026+).
# llama-3.3-70b-versatile was deprecated by Groq on August 16 2026.
MODEL         = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# API key: Streamlit Cloud secrets -> .env fallback
try:
    API_KEY = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    API_KEY = os.getenv("GROQ_API_KEY", "")

# App password: Streamlit Cloud secrets -> .env fallback
try:
    APP_PASSWORD = st.secrets["APP_PASSWORD"]
except (KeyError, FileNotFoundError):
    APP_PASSWORD = os.getenv("APP_PASSWORD", "")

# Password gate — only active when APP_PASSWORD is set
if APP_PASSWORD:
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.markdown(
            """
            <div style='max-width:360px; margin:10vh auto; text-align:center'>
                <div style='font-size:2.8rem'>&#x1F9FA;</div>
                <h2 style='margin-bottom:4px'>ABAP Code Doctor</h2>
                <p style='color:#6b7280; margin-bottom:24px'>
                    Enter the access password to continue.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            pwd_input = st.text_input(
                "Password",
                type="password",
                label_visibility="collapsed",
                placeholder="Password",
            )
            if st.button("Enter", use_container_width=True, type="primary"):
                if pwd_input == APP_PASSWORD:
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
        st.stop()

if not API_KEY:
    st.error("GROQ_API_KEY is missing. Add it to Streamlit secrets or your .env file.")
    st.stop()

client = OpenAI(
    api_key=API_KEY,
    base_url=GROQ_BASE_URL,
)


# ============================================================
# TOKEN BUDGET
# 8 000 tokens / minute shared across ALL requests in a window.
# We keep a conservative ceiling so analysis + fix in the same
# minute cannot blow the limit.
# ============================================================

# Groq free tier: openai/gpt-oss-120b — 8 000 tokens/min, 30 RPM, 1 000 req/day
# Keep output caps generous so JSON responses are never truncated.
TPM_LIMIT          = 8_000   # Groq free tier TPM for openai/gpt-oss-120b
ANALYSIS_MAX_OUT   = 2_400   # output cap for analysis call
FIX_MAX_OUT        = 2_800   # output cap for fix call
PROMPT_SAFETY_PAD  = 300     # buffer for system messages + overhead


def estimate_tokens(text: str) -> int:
    """Conservative token estimate: ~3.5 chars per token."""
    return max(1, int(len(text) / 3.5) + 1)


def budget_check(input_tokens: int, output_cap: int, label: str) -> None:
    """Raise early if a single request would exceed the per-minute limit."""
    total = input_tokens + output_cap + PROMPT_SAFETY_PAD
    if total > TPM_LIMIT:
        raise ValueError(
            f"{label} request needs ~{total} tokens but the Groq limit is "
            f"{TPM_LIMIT} tokens/min. Reduce the ABAP source size or analyze "
            "the program in logical sections."
        )


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "analysis": None,
    "analyzed_code": "",
    "fixed_code": "",
    "show_chat": False,
    "chat_pairs": [],
    "chat_mode": "Direct",
    "uploaded_filename": "",
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# UI CSS
# ============================================================

st.markdown(
    """
    <style>

    .doctor-title {
        font-size: 2.15rem;
        font-weight: 700;
        line-height: 1.15;
        margin-bottom: 3px;
    }

    .doctor-subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 18px;
    }

    .chat-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .chat-subtitle {
        color: #6b7280;
        font-size: 0.82rem;
        margin-bottom: 12px;
    }

    div[data-testid="InputInstructions"] {
        display: none !important;
    }

    div[data-testid="stTextInput"] input {
        padding-top: 0.75rem !important;
        padding-bottom: 0.75rem !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

SEVERITY_RANK = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 4,
}


def normalize_severity(value):
    value = str(value or "").strip().title()
    if value not in SEVERITY_RANK:
        return "Medium"
    return value


def higher_severity(a, b):
    a = normalize_severity(a)
    b = normalize_severity(b)
    return a if SEVERITY_RANK[a] >= SEVERITY_RANK[b] else b


def severity_icon(severity):
    return {
        "Critical": "🔴",
        "High": "🟠",
        "Medium": "🟡",
        "Low": "🟢",
    }[normalize_severity(severity)]


def health_label(score):
    if score >= 90:
        return "Healthy"
    if score >= 75:
        return "Good"
    if score >= 55:
        return "Needs Improvement"
    return "High Risk"


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def extract_lines(code, start, end=None):
    lines = code.splitlines()
    if not start:
        return ""
    start = max(1, start)
    end = end or start
    end = min(len(lines), end)
    return "\n".join(lines[start - 1:end])


def location_text(start, end=None):
    if not start:
        return "Code context"
    if end and end != start:
        return f"Lines {start}-{end}"
    return f"Line {start}"


def make_finding(
    rule_id,
    category,
    severity,
    title,
    problem,
    recommendation="",
    start_line=None,
    end_line=None,
    existing_code="",
    recommended_code="",
    confidence="High",
    developer_verification=False,
    why_it_matters="",
):
    return {
        "id": rule_id,
        "category": category,
        "severity": normalize_severity(severity),
        "title": title,
        "location": location_text(start_line, end_line),
        "start_line": start_line,
        "end_line": end_line or start_line,
        "existing_code": existing_code,
        "problem": problem,
        "why_it_matters": why_it_matters,
        "recommendation": recommendation,
        "recommended_code": recommended_code,
        "confidence": confidence,
        "developer_verification": developer_verification,
    }


# ============================================================
# ABAP COMMENT MASKING
# ============================================================

def mask_abap_comments(code):
    result = []
    for line in code.splitlines():
        if line.lstrip().startswith("*"):
            result.append("")
            continue
        output = []
        inside_string = False
        i = 0
        while i < len(line):
            char = line[i]
            if char == "'":
                output.append(char)
                if inside_string:
                    if i + 1 < len(line) and line[i + 1] == "'":
                        output.append("'")
                        i += 2
                        continue
                    inside_string = False
                else:
                    inside_string = True
                i += 1
                continue
            if char == '"' and not inside_string:
                break
            output.append(char)
            i += 1
        result.append("".join(output))
    return "\n".join(result)


def mask_abap_comments_and_strings(code):
    result = []
    for line in code.splitlines():
        if line.lstrip().startswith("*"):
            result.append("")
            continue
        output = []
        inside_string = False
        i = 0
        while i < len(line):
            char = line[i]
            if char == "'":
                if inside_string:
                    if i + 1 < len(line) and line[i + 1] == "'":
                        output.extend((" ", " "))
                        i += 2
                        continue
                    output.append(" ")
                    inside_string = False
                    i += 1
                    continue
                output.append(" ")
                inside_string = True
                i += 1
                continue
            if char == '"' and not inside_string:
                output.extend(" " for _ in line[i:])
                break
            output.append(" " if inside_string else char)
            i += 1
        result.append("".join(output))
    return "\n".join(result)


def line_from_offset(text, offset):
    return text.count("\n", 0, offset) + 1


# ============================================================
# SEVERITY FLOOR
# ============================================================

SEVERITY_FLOORS = {
    "DB_DESTRUCTIVE_DELETE":     "Critical",
    "DB_DIRECT_DELETE":          "Critical",
    "DB_SELECT_IN_LOOP":         "High",
    "DB_MOD_IN_LOOP":            "High",
    "ITAB_STANDARD_LINEAR_LOOKUP": "High",
    "SEC_AUTH_RESULT_IGNORED":   "High",
    "SEC_DYNAMIC_TRANSACTION":   "High",
    "SEC_HARDCODED_USER":        "High",
    "ERR_EMPTY_CATCH":           "High",
    "BAPI_RETURN_IGNORED":       "High",
    "BAPI_COMMIT_IN_LOOP":       "High",
    "LUW_COMMIT_IN_LOOP":        "High",
    "LOGIC_STALE_SY_SUBRC":      "High",
    "FM_NO_EXCEPTION_HANDLING":  "High",
    "DB_SELECT_SINGLE_PARTIAL":  "Medium",
    "DB_DYNAMIC_SELECT":         "High",
    "LOGIC_VARIABLE_DIVISOR":    "High",
    "LUW_COMMIT_IN_LOOP_RAW":    "High",
    "ITAB_SORT_MISSING":         "Medium",
    "LOGIC_SY_TABIX_STALE":      "Medium",
    "DB_SELECT_NO_SUBRC_CHECK":  "High",
}


def apply_severity_floor(finding):
    rule = finding.get("id", "")
    floor = SEVERITY_FLOORS.get(rule)
    if floor:
        finding["severity"] = higher_severity(finding.get("severity"), floor)
    return finding


# ============================================================
# STATIC ABAP ANALYZER
# ============================================================

def local_abap_scan(code):

    scan_code  = mask_abap_comments(code)
    db_scan_code = mask_abap_comments_and_strings(code)

    lines          = scan_code.splitlines()
    original_lines = code.splitlines()

    findings = []

    def add(*args, **kwargs):
        findings.append(make_finding(*args, **kwargs))

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    executable_lines = sum(
        1 for line in original_lines
        if line.strip()
        and not line.strip().startswith("*")
        and not line.strip().startswith('"')
    )

    comment_lines = sum(
        1 for line in original_lines
        if line.strip().startswith("*") or line.strip().startswith('"')
    )

    # --------------------------------------------------------
    # LOOP RANGES
    # --------------------------------------------------------

    loop_ranges = []
    loop_stack  = []

    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if re.match(r"^LOOP\s+AT\b", stripped, re.IGNORECASE):
            loop_stack.append(number)
        elif re.match(r"^ENDLOOP\b", stripped, re.IGNORECASE):
            if loop_stack:
                start = loop_stack.pop()
                loop_ranges.append((start, number))

    # --------------------------------------------------------
    # SELECT
    # --------------------------------------------------------

    select_pattern = re.compile(r"\bSELECT\b.*?\.", re.IGNORECASE | re.DOTALL)
    select_matches = list(select_pattern.finditer(scan_code))

    for match in select_matches:
        statement  = match.group(0)
        start_line = line_from_offset(scan_code, match.start())
        end_line   = line_from_offset(scan_code, match.end())

        # SELECT *
        if re.search(r"\bSELECT\s+\*", statement, re.IGNORECASE):
            add(
                "DB_SELECT_STAR", "Database", "Medium",
                "SELECT * retrieves unnecessary columns",
                "The statement retrieves every column from the database table.",
                "Select only the fields required by the program.",
                start_line, end_line,
                extract_lines(code, start_line, end_line),
                confidence="High",
                why_it_matters=(
                    "Unnecessary columns increase data transfer, "
                    "memory usage and coupling to the table structure."
                ),
            )

        # SELECT inside loop
        inside_loop = any(s <= start_line <= e for s, e in loop_ranges)
        if inside_loop:
            add(
                "DB_SELECT_IN_LOOP", "Database Performance", "High",
                "Database query inside loop",
                "A database operation executes during iterative processing.",
                "Move the database retrieval outside the loop and use set-based retrieval.",
                start_line, end_line,
                extract_lines(code, start_line, end_line),
                confidence="High",
                why_it_matters=(
                    "Database access grows with the number of records processed "
                    "and becomes a major scalability bottleneck."
                ),
            )

        # SELECT SINGLE without full key indicator
        if re.search(r"\bSELECT\s+SINGLE\b", statement, re.IGNORECASE):
            where_part = re.search(r"\bWHERE\b(.*?)(?:\.|$)", statement, re.IGNORECASE | re.DOTALL)
            if where_part:
                conditions = where_part.group(1)
                # Flag if WHERE has fewer than 2 AND-connected conditions (likely partial key)
                and_count = len(re.findall(r"\bAND\b", conditions, re.IGNORECASE))
                if and_count < 1:
                    add(
                        "DB_SELECT_SINGLE_PARTIAL", "Database", "Medium",
                        "SELECT SINGLE with potentially incomplete key",
                        "SELECT SINGLE uses a WHERE clause with only one condition — the full primary key may not be specified.",
                        "Ensure the full primary key is provided in the WHERE clause to guarantee deterministic results.",
                        start_line, end_line,
                        extract_lines(code, start_line, end_line),
                        confidence="Medium",
                        developer_verification=True,
                        why_it_matters=(
                            "SELECT SINGLE on a partial key returns an arbitrary matching row "
                            "and can produce non-deterministic behavior."
                        ),
                    )

        # Dynamic SELECT with field list variable
        if re.search(r"\bSELECT\b\s*\([A-Za-z_][A-Za-z0-9_]*\)", statement, re.IGNORECASE):
            add(
                "DB_DYNAMIC_SELECT", "Database / Security", "High",
                "Dynamic SELECT field list",
                "The SELECT statement uses a variable to specify the field list.",
                "Validate and restrict the dynamic field list. Prefer explicit column names.",
                start_line, end_line,
                extract_lines(code, start_line, end_line),
                confidence="High",
                why_it_matters=(
                    "Dynamic field lists can expose unintended columns and are harder to review for security."
                ),
            )

        # ── SY-SUBRC not checked after SELECT ──────────────────
        #
        # SELECT SINGLE / SELECT ... INTO scalar:
        #   Missing SY-SUBRC check means the program uses the
        #   target variable even when no row was found.
        #
        # SELECT ... INTO TABLE:
        #   An empty result is often acceptable (LOOP just skips).
        #   Flag only when the table variable is referenced before
        #   any guard in the next 15 lines.
        # ────────────────────────────────────────────────────────

        is_single  = bool(re.search(r"\bSELECT\s+SINGLE\b", statement, re.IGNORECASE))
        into_table = bool(re.search(r"\bINTO\s+(?:@?DATA\(.*?\)\s+)?TABLE\b",
                                    statement, re.IGNORECASE | re.DOTALL))

        into_scalar_match = re.search(
            r"\bINTO\b\s+(?:TABLE\s+)?(?:@?DATA\()?@?(?P<var>[A-Za-z_][A-Za-z0-9_]*)",
            statement, re.IGNORECASE,
        )
        target_var = into_scalar_match.group("var") if into_scalar_match else None

        look_start  = end_line
        look_end    = min(len(lines), end_line + 15)
        after_lines = lines[look_start: look_end]

        subrc_checked     = False
        target_used_first = False

        for after_line in after_lines:
            stripped_after = after_line.strip()
            if not stripped_after or stripped_after.startswith("*"):
                continue

            # Guards — evaluated before generic var-use check
            # 1. Explicit SY-SUBRC check
            if re.search(r"\bsy-subrc\b", stripped_after, re.IGNORECASE):
                subrc_checked = True
                break

            # 2. LOOP AT <target> right after INTO TABLE — safe
            if into_table and re.match(r"\bLOOP\s+AT\b", stripped_after, re.IGNORECASE):
                subrc_checked = True
                break

            # 3. IS [NOT] INITIAL guard on the target variable
            if target_var and re.search(
                rf"\b{re.escape(target_var)}\b.*?\bIS\s+(?:NOT\s+)?INITIAL\b",
                stripped_after, re.IGNORECASE,
            ):
                subrc_checked = True
                break

            # Unguarded use — only reached when no guard matched above
            if target_var and re.search(
                rf"\b{re.escape(target_var)}\b", stripped_after, re.IGNORECASE,
            ):
                target_used_first = True
                break

        if not subrc_checked:
            if is_single:
                add(
                    "DB_SELECT_NO_SUBRC_CHECK",
                    "Logic / Error Handling", "High",
                    "SY-SUBRC not checked after SELECT SINGLE",
                    (
                        "SELECT SINGLE does not visibly check SY-SUBRC after execution. "
                        "If no row matches, the target variable retains its initial value "
                        "and subsequent code silently processes stale data."
                    ),
                    "Add IF sy-subrc <> 0. immediately after SELECT SINGLE and handle the not-found case.",
                    start_line, end_line,
                    extract_lines(code, start_line, end_line),
                    recommended_code=(
                        "SELECT SINGLE ... INTO @ls_data WHERE ....\n"
                        "IF sy-subrc <> 0.\n"
                        "  \" record not found — handle appropriately\n"
                        "ENDIF."
                    ),
                    confidence="High",
                    why_it_matters=(
                        "Using the result of a SELECT SINGLE without checking SY-SUBRC "
                        "silently processes empty/initial data when no row is found, "
                        "leading to incorrect results or downstream errors."
                    ),
                )
            elif target_used_first:
                severity = "High" if not into_table else "Medium"
                add(
                    "DB_SELECT_NO_SUBRC_CHECK",
                    "Logic / Error Handling", severity,
                    "Result used after SELECT without SY-SUBRC check",
                    (
                        f"The variable \'{target_var}\' is referenced after SELECT "
                        "without a visible SY-SUBRC or IS NOT INITIAL check. "
                        "If no row matched, the variable holds its initial value."
                    ),
                    (
                        "Check SY-SUBRC or test IS NOT INITIAL immediately after the SELECT "
                        "and guard all uses of the result."
                    ),
                    start_line, end_line,
                    extract_lines(code, start_line, end_line),
                    recommended_code=(
                        "SELECT ... INTO @ls_data WHERE ....\n"
                        "IF sy-subrc <> 0.\n"
                        "  \" not found — handle appropriately\n"
                        "ENDIF."
                    ),
                    confidence="Medium",
                    developer_verification=True,
                    why_it_matters=(
                        "Processing an uninitialised or empty result causes incorrect "
                        "business outcomes that may not surface until production."
                    ),
                )

    # --------------------------------------------------------
    # FOR ALL ENTRIES
    # --------------------------------------------------------

    fae_pattern = re.compile(
        r"\bFOR\s+ALL\s+ENTRIES\s+IN\s+@?(?P<table>[A-Za-z_][A-Za-z0-9_]*)",
        re.IGNORECASE,
    )

    for match in fae_pattern.finditer(scan_code):
        table_name = match.group("table")
        line       = line_from_offset(scan_code, match.start())
        previous   = "\n".join(lines[max(0, line - 10): line - 1])

        guarded = bool(re.search(
            rf"\bIF\s+{re.escape(table_name)}\s+IS\s+NOT\s+INITIAL\b",
            previous, re.IGNORECASE,
        ))

        if not guarded:
            add(
                "DB_FAE_CHECK", "Database", "High",
                "FOR ALL ENTRIES without driver-table validation",
                f"{table_name} is used as the FOR ALL ENTRIES driver without a visible IS NOT INITIAL check.",
                f"Validate {table_name} before FOR ALL ENTRIES and review duplicate driver keys.",
                line,
                existing_code=extract_lines(code, line, min(len(lines), line + 8)),
                recommended_code=(
                    f"IF {table_name} IS NOT INITIAL.\n"
                    f"  \" SELECT ... FOR ALL ENTRIES IN {table_name}\n"
                    f"ENDIF."
                ),
                confidence="High",
                why_it_matters="An empty driver table causes an unintended full table read.",
            )

    # --------------------------------------------------------
    # READ TABLE
    # --------------------------------------------------------

    read_pattern = re.compile(
        r"""
        \bREAD\s+TABLE\s+
        (?P<table>[A-Za-z_][A-Za-z0-9_]*)
        (?P<body>.*?)
        (?=\.)
        """,
        re.IGNORECASE | re.DOTALL | re.VERBOSE,
    )

    binary_search_targets = set()   # tables confirmed to have BINARY SEARCH

    for match in read_pattern.finditer(scan_code):
        table_name  = match.group("table")
        body        = match.group("body")
        line        = line_from_offset(scan_code, match.start())

        with_key      = bool(re.search(r"\bWITH\s+KEY\b",     body, re.IGNORECASE))
        binary_search = bool(re.search(r"\bBINARY\s+SEARCH\b", body, re.IGNORECASE))
        inside_loop   = any(s <= line <= e for s, e in loop_ranges)

        if binary_search:
            binary_search_targets.add(table_name.upper())

        if inside_loop and with_key and not binary_search:
            add(
                "ITAB_REPEATED_LOOKUP", "Internal Tables", "Medium",
                "Repeated internal-table lookup without BINARY SEARCH",
                f"READ TABLE {table_name} WITH KEY is executed during loop processing without BINARY SEARCH.",
                "Use HASHED or SORTED table type, or add BINARY SEARCH after ensuring the table is sorted.",
                line,
                existing_code=extract_lines(code, line, min(len(lines), line + 5)),
                confidence="Medium",
                developer_verification=True,
                why_it_matters="Linear lookup complexity becomes expensive for large datasets.",
            )

    # --------------------------------------------------------
    # SORT without matching BINARY SEARCH  (new rule)
    # --------------------------------------------------------

    sort_pattern = re.compile(
        r"\bSORT\s+(?P<table>[A-Za-z_][A-Za-z0-9_]*)\b",
        re.IGNORECASE,
    )

    for match in sort_pattern.finditer(scan_code):
        table_name = match.group("table").upper()
        line       = line_from_offset(scan_code, match.start())

        # Check whether BINARY SEARCH appears anywhere after this SORT
        rest = scan_code[match.end():]
        has_binary = bool(re.search(
            rf"\bREAD\s+TABLE\s+{re.escape(table_name)}\b.*?\bBINARY\s+SEARCH\b",
            rest, re.IGNORECASE | re.DOTALL,
        ))

        if not has_binary and table_name not in binary_search_targets:
            add(
                "ITAB_SORT_MISSING", "Internal Tables", "Medium",
                "SORT without subsequent BINARY SEARCH",
                f"SORT {table_name} is performed but no BINARY SEARCH READ TABLE follows.",
                "Either add BINARY SEARCH to the corresponding READ TABLE, or remove the SORT if it is not needed for presentation.",
                line,
                existing_code=extract_lines(code, line),
                confidence="Medium",
                developer_verification=True,
                why_it_matters="An unused SORT wastes CPU and can mislead maintainers into thinking keyed lookup is optimized.",
            )

    # --------------------------------------------------------
    # NESTED LOOPS
    # --------------------------------------------------------

    loop_stack_nested = []
    reported_nested   = set()

    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if re.match(r"^LOOP\s+AT\b", stripped, re.IGNORECASE):
            if loop_stack_nested:
                outer_start = loop_stack_nested[-1]
                if outer_start not in reported_nested:
                    reported_nested.add(outer_start)
                    end_line = min(len(lines), number + 8)
                    add(
                        "ITAB_NESTED_LOOP", "Internal Tables", "Medium",
                        "Nested loop processing",
                        "A LOOP AT begins while another LOOP AT is active.",
                        "Replace repeated scans with keyed lookups, set-based retrieval or a more suitable data structure.",
                        outer_start, end_line,
                        existing_code=extract_lines(code, outer_start, end_line),
                        confidence="High",
                        why_it_matters="Nested scans produce multiplicative processing cost as the dataset grows.",
                    )
            loop_stack_nested.append(number)
        elif re.match(r"^ENDLOOP\b", stripped, re.IGNORECASE):
            if loop_stack_nested:
                loop_stack_nested.pop()

    # --------------------------------------------------------
    # DB MODIFICATIONS
    # --------------------------------------------------------

    update_pattern = re.compile(r"\bUPDATE\s+(?P<table>[A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
    delete_pattern = re.compile(r"\bDELETE\s+FROM\s+(?P<table>[A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
    insert_pattern = re.compile(r"\bINSERT\s+(?:INTO\s+)?(?P<table>[A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
    modify_pattern = re.compile(r"\bMODIFY\s+(?P<table>[A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)

    modification_patterns = [
        ("UPDATE", update_pattern),
        ("DELETE", delete_pattern),
        ("INSERT", insert_pattern),
        ("MODIFY", modify_pattern),
    ]

    db_modification_count = 0

    for operation, pattern in modification_patterns:
        for match in pattern.finditer(db_scan_code):
            table_name = match.group("table")
            line = line_from_offset(scan_code, match.start())
            db_modification_count += 1

            if operation == "DELETE":
                severity       = "Critical"
                title          = "Destructive database deletion"
                problem        = f"DELETE directly modifies persistent table {table_name}."
                recommendation = (
                    "Verify whether a supported application-level business API should be used. "
                    "Confirm selection conditions, authorization, locking and transaction handling."
                )
                why = "A direct DELETE can permanently remove persistent data and may bypass application-level validation."
            else:
                severity       = "High"
                title          = f"Direct database {operation}"
                problem        = f"{operation} directly modifies persistent table {table_name}."
                recommendation = (
                    "Review whether an application-level API, BAPI, RAP behavior or other SAP-supported "
                    "mechanism should be used. Verify authorization, validation, locking and transaction handling."
                )
                why = "Direct persistent database modification requires careful validation, authorization and transaction control."

            add(
                f"DB_DIRECT_{operation}", "Database Safety", severity,
                title, problem, recommendation, line,
                existing_code=extract_lines(code, line, min(len(lines), line + 8)),
                confidence="High",
                developer_verification=True,
                why_it_matters=why,
            )

            inside_loop = any(s <= line <= e for s, e in loop_ranges)
            if inside_loop:
                add(
                    "DB_MOD_IN_LOOP", "Database Performance / LUW", "High",
                    f"{operation} executed inside loop",
                    f"{operation} on {table_name} occurs during iterative processing.",
                    "Review whether the changes can be accumulated and processed using an appropriate transaction strategy.",
                    line,
                    existing_code=extract_lines(code, line, min(len(lines), line + 8)),
                    confidence="High",
                    why_it_matters="Repeated database modifications increase database work and can create partial-success scenarios.",
                )

    # --------------------------------------------------------
    # CALL FUNCTION — missing exception handling  (new rule)
    # --------------------------------------------------------

    cf_pattern = re.compile(
        r"\bCALL\s+FUNCTION\s+['\"](?P<fm>[A-Za-z0-9_/]+)['\"]"
        r"(?P<body>.*?)\.",
        re.IGNORECASE | re.DOTALL,
    )

    for match in cf_pattern.finditer(scan_code):
        fm_name = match.group("fm")
        body    = match.group("body")
        line    = line_from_offset(scan_code, match.start())

        # Skip BAPI calls — handled separately
        if fm_name.upper().startswith("BAPI_"):
            continue

        has_exceptions = bool(re.search(r"\bEXCEPTIONS\b", body, re.IGNORECASE))
        others_zero    = bool(re.search(r"\bOTHERS\s*=\s*0\b", body, re.IGNORECASE))

        if not has_exceptions:
            add(
                "FM_NO_EXCEPTION_HANDLING", "Error Handling", "High",
                "Function module called without exception handling",
                f"CALL FUNCTION '{fm_name}' has no EXCEPTIONS block.",
                "Add an EXCEPTIONS block and check SY-SUBRC after the call.",
                line,
                existing_code=extract_lines(code, line, min(len(lines), line + 8)),
                recommended_code=(
                    f"CALL FUNCTION '{fm_name}'\n"
                    "  ...\n"
                    "  EXCEPTIONS\n"
                    "    error_condition = 1\n"
                    "    OTHERS          = 2.\n\n"
                    "IF sy-subrc <> 0.\n"
                    "  \" handle error\n"
                    "ENDIF."
                ),
                confidence="High",
                why_it_matters=(
                    "Unhandled function module exceptions cause runtime errors "
                    "that terminate the program without meaningful diagnostics."
                ),
            )
        elif others_zero:
            add(
                "FM_OTHERS_ZERO", "Error Handling", "Medium",
                "EXCEPTIONS OTHERS = 0 suppresses all errors",
                f"CALL FUNCTION '{fm_name}' sets OTHERS = 0, which means all unspecified exceptions are silently ignored.",
                "Set OTHERS to a non-zero value and check SY-SUBRC after the call.",
                line,
                existing_code=extract_lines(code, line, min(len(lines), line + 8)),
                confidence="High",
                why_it_matters="OTHERS = 0 silently swallows every exception not explicitly listed.",
            )

    # --------------------------------------------------------
    # COMMIT WORK inside loop — without BAPI context  (new rule)
    # --------------------------------------------------------

    for number, line_text in enumerate(lines, 1):
        if re.search(r"\bCOMMIT\s+WORK\b", line_text, re.IGNORECASE):
            inside_loop = any(s <= number <= e for s, e in loop_ranges)
            if inside_loop:
                add(
                    "LUW_COMMIT_IN_LOOP_RAW", "LUW / Transaction Safety", "High",
                    "COMMIT WORK inside loop",
                    "COMMIT WORK is executed during loop processing.",
                    "Define the correct LUW boundary. Move COMMIT WORK outside the loop or implement explicit error recovery.",
                    number,
                    existing_code=extract_lines(code, number),
                    confidence="High",
                    why_it_matters=(
                        "Each iteration becomes a separate database transaction, "
                        "creating partial-success risk with no rollback path."
                    ),
                )

    # --------------------------------------------------------
    # DIVISION BY ZERO — literal zero
    # --------------------------------------------------------

    for match in re.finditer(r"/\s*0(?:\b|\.)", scan_code, re.IGNORECASE):
        line = line_from_offset(scan_code, match.start())

        # Skip WRITE: /0 or WRITE /05 output formatting
        _lt = lines[line - 1].strip()
        if re.match(r"\bWRITE\b", _lt, re.IGNORECASE):
            continue
        # Must have a value/variable before the slash (arithmetic context)
        _before = scan_code[max(0, match.start() - 80): match.start()]
        _before_line = _before.split("\n")[-1]
        if not re.search(r"[\w\).]\s*$", _before_line):
            continue

        add(
            "LOGIC_DIVISION_BY_ZERO", "Logic / Runtime Safety", "Critical",
            "Division by literal zero",
            "A division operation uses a literal zero as the divisor.",
            "Validate the divisor before division and handle the zero case explicitly.",
            line,
            existing_code=extract_lines(code, line),
            recommended_code=(
                "IF lv_divisor <> 0.\n"
                "  lv_result = lv_value / lv_divisor.\n"
                "ENDIF."
            ),
            confidence="High",
            why_it_matters="A zero divisor causes a runtime error or invalid calculation.",
        )

    # --------------------------------------------------------
    # DIVISION BY ZERO — variable divisor without guard  (new rule)
    # --------------------------------------------------------

    div_var_pattern = re.compile(
        r"/\s*(?P<var>[A-Za-z_][A-Za-z0-9_]*(?:-[A-Za-z0-9_]+)?)",
        re.IGNORECASE,
    )

    for match in div_var_pattern.finditer(scan_code):
        var  = match.group("var")
        line = line_from_offset(scan_code, match.start())

        # Skip false positives
        line_text = lines[line - 1].strip()

        # 1. WRITE: / or WRITE / — new-line output operator, not division
        if re.match(r"\bWRITE\b", line_text, re.IGNORECASE):
            continue

        # 2. ULINE / SKIP / FORMAT / output-layout keywords use / differently
        if re.match(r"\b(ULINE|SKIP|FORMAT|AT|NEW-LINE|NEW-PAGE)\b",
                    line_text, re.IGNORECASE):
            continue

        # 3. SAP namespace slash e.g. /BIC/MYFIELD — not division
        if re.match(r"^/[A-Za-z]", match.group(0)):
            continue

        # 4. Must have a value/variable/closing-paren immediately before the /
        #    on the same line — otherwise it is not an arithmetic division
        before_slash = scan_code[max(0, match.start() - 80): match.start()]
        before_on_line = before_slash.split("\n")[-1]
        if not re.search(r"[\w\).]\s*$", before_on_line):
            continue

        preceding_block = "\n".join(lines[max(0, line - 8): line - 1])
        guarded = bool(re.search(
            rf"\b(?:IF|CHECK)\b.*?\b{re.escape(var)}\b\s*<>\s*0",
            preceding_block, re.IGNORECASE,
        ))

        if not guarded:
            add(
                "LOGIC_VARIABLE_DIVISOR", "Logic / Runtime Safety", "High",
                "Division by variable without zero guard",
                f"Division by {var} occurs without a visible preceding zero check.",
                f"Add: IF {var} <> 0. before the division.",
                line,
                existing_code=extract_lines(code, line),
                recommended_code=(
                    f"IF {var} <> 0.\n"
                    f"  lv_result = lv_value / {var}.\n"
                    f"ENDIF."
                ),
                confidence="Medium",
                developer_verification=True,
                why_it_matters="An unguarded variable divisor causes a runtime dump when the divisor is zero.",
            )

    # --------------------------------------------------------
    # STALE SY-SUBRC
    # --------------------------------------------------------

    # --------------------------------------------------------
    # STALE SY-SUBRC
    #
    # Problem with line-by-line tracking: SELECT is usually
    # multi-line.  By the time IF sy-subrc appears, the last
    # line seen was the WHERE clause — not SELECT — so the
    # producer regex missed it and raised a false finding.
    #
    # Fix: reconstruct logical statements by joining lines up
    # to the dot terminator, then match the producer regex
    # against the first keyword of each full statement.
    # --------------------------------------------------------

    sy_subrc_condition = re.compile(r"\b(?:IF|CHECK|CASE)\s+sy-subrc\b", re.IGNORECASE)

    # Keywords whose statement sets SY-SUBRC
    sy_subrc_producers = re.compile(
        r"^(?:SELECT\b|READ\s+TABLE\b|AUTHORITY-CHECK\b|UPDATE\b|"
        r"DELETE\s+FROM\b|INSERT\b|MODIFY\b|CALL\s+FUNCTION\b|"
        r"CALL\s+TRANSACTION\b|SUBMIT\b|ENQUEUE\b|DEQUEUE\b|"
        r"OPEN\s+CURSOR\b|FETCH\s+NEXT\s+CURSOR\b|CLOSE\s+CURSOR\b)",
        re.IGNORECASE,
    )

    # Build a list of (line_number, first_keyword_of_statement)
    # by accumulating lines until we hit a dot terminator.
    stmt_first_keyword: list[tuple[int, str]] = []
    stmt_start_line = 1
    stmt_buf: list[str] = []

    for number, raw_line in enumerate(lines, 1):
        stripped = raw_line.strip()

        # Skip full-line comments
        if stripped.startswith("*"):
            continue

        # Strip inline comment for statement detection
        no_inline = re.split(r'"', stripped)[0].strip()
        if not no_inline:
            continue

        if not stmt_buf:
            stmt_start_line = number

        stmt_buf.append(no_inline)

        # Statement ends at a dot that is NOT inside a string
        if no_inline.endswith("."):
            full_stmt = " ".join(stmt_buf)
            first_kw  = full_stmt.strip()
            stmt_first_keyword.append((stmt_start_line, first_kw))
            stmt_buf = []

    # Also flush any unterminated trailing statement
    if stmt_buf:
        stmt_first_keyword.append((stmt_start_line, " ".join(stmt_buf)))

    # Walk the statement list and flag sy-subrc checks where
    # the immediately preceding statement is not a producer.
    for idx, (stmt_line, stmt_text) in enumerate(stmt_first_keyword):
        if not sy_subrc_condition.search(stmt_text):
            continue

        if idx == 0:
            continue  # no predecessor — can't judge

        prev_line, prev_text = stmt_first_keyword[idx - 1]

        if sy_subrc_producers.match(prev_text.strip()):
            continue  # correct: SY-SUBRC follows its producer

        add(
            "LOGIC_STALE_SY_SUBRC", "Logic / Error Handling", "High",
            "Potential stale SY-SUBRC check",
            (
                "SY-SUBRC is checked but the immediately preceding statement "
                f"('{prev_text[:60].strip()}...') does not visibly set it."
            ),
            "Check SY-SUBRC immediately after the operation that sets it.",
            stmt_line,
            existing_code=extract_lines(code, max(1, prev_line), stmt_line),
            confidence="High",
            why_it_matters=(
                "A stale SY-SUBRC reflects an earlier operation and causes "
                "incorrect program flow."
            ),
        )

    # --------------------------------------------------------
    # SY-TABIX after failed READ TABLE  (new rule)
    # --------------------------------------------------------

    tabix_pattern  = re.compile(r"\bsy-tabix\b", re.IGNORECASE)
    read_tbl_pat   = re.compile(r"\bREAD\s+TABLE\b", re.IGNORECASE)

    for match in tabix_pattern.finditer(scan_code):
        line = line_from_offset(scan_code, match.start())

        # Look back up to 10 lines for a READ TABLE
        preceding = "\n".join(lines[max(0, line - 10): line - 1])

        if not read_tbl_pat.search(preceding):
            continue

        # Check whether SY-SUBRC is verified between READ TABLE and SY-TABIX
        subrc_checked = bool(re.search(
            r"\bIF\s+sy-subrc\b|\bCHECK\s+sy-subrc\b",
            preceding, re.IGNORECASE,
        ))

        if not subrc_checked:
            add(
                "LOGIC_SY_TABIX_STALE", "Logic / Error Handling", "Medium",
                "SY-TABIX used after READ TABLE without SY-SUBRC check",
                "SY-TABIX is referenced after a READ TABLE but SY-SUBRC is not checked first.",
                "Check SY-SUBRC = 0 before using SY-TABIX to ensure the READ TABLE succeeded.",
                line,
                existing_code=extract_lines(code, max(1, line - 3), line),
                confidence="Medium",
                developer_verification=True,
                why_it_matters="SY-TABIX retains its previous value when READ TABLE fails, leading to incorrect index usage.",
            )

    # --------------------------------------------------------
    # AUTHORITY-CHECK
    # --------------------------------------------------------

    authority_pattern = re.compile(r"\bAUTHORITY-CHECK\b.*?\.", re.IGNORECASE | re.DOTALL)
    authority_matches = list(authority_pattern.finditer(scan_code))

    for match in authority_matches:
        start = line_from_offset(scan_code, match.start())
        end   = line_from_offset(scan_code, match.end())

        following = "\n".join(lines[end: min(len(lines), end + 12)])

        checked = bool(re.search(
            r"\bIF\s+sy-subrc\b|\bCHECK\s+sy-subrc\b|\bCASE\s+sy-subrc\b",
            following, re.IGNORECASE | re.VERBOSE,
        ))

        if not checked:
            add(
                "SEC_AUTH_RESULT_IGNORED", "Security", "High",
                "Authorization result is not enforced",
                "AUTHORITY-CHECK is present, but its SY-SUBRC does not visibly control subsequent execution.",
                "Check SY-SUBRC immediately after AUTHORITY-CHECK and stop or restrict the sensitive operation on failure.",
                start, end,
                existing_code=extract_lines(code, start, min(len(lines), end + 8)),
                recommended_code=(
                    "AUTHORITY-CHECK OBJECT '...'\n"
                    "  ID '...' FIELD '...'.\n\n"
                    "IF sy-subrc <> 0.\n"
                    "  MESSAGE 'Not authorized' TYPE 'E'.\n"
                    "ENDIF."
                ),
                confidence="High",
                why_it_matters="An authorization check that does not affect execution provides no effective authorization control.",
            )

    # --------------------------------------------------------
    # CALL TRANSACTION
    # --------------------------------------------------------

    transaction_pattern = re.compile(
        r"\bCALL\s+TRANSACTION\s+"
        r"(?P<target>'[^']+'|\([^)]+\)|[A-Za-z_][A-Za-z0-9_]*)",
        re.IGNORECASE,
    )

    for match in transaction_pattern.finditer(scan_code):
        target = match.group("target")
        line   = line_from_offset(scan_code, match.start())

        if target.startswith("("):
            add(
                "SEC_DYNAMIC_TRANSACTION", "Security", "High",
                "Dynamic transaction execution",
                "The transaction target is determined dynamically.",
                "Validate and strictly control the transaction target before execution.",
                line,
                existing_code=extract_lines(code, line, min(len(lines), line + 4)),
                confidence="High",
                why_it_matters="Dynamic execution expands the set of transactions that can be invoked.",
            )
        else:
            add(
                "SEC_CALL_TRANSACTION", "Security", "Medium",
                "CALL TRANSACTION requires authorization review",
                f"The program explicitly executes transaction {target}.",
                "Verify authorization, transaction input validation and execution context.",
                line,
                existing_code=extract_lines(code, line),
                confidence="High",
                why_it_matters="Transaction execution can invoke sensitive business operations.",
            )

    # --------------------------------------------------------
    # SUBMIT
    # --------------------------------------------------------

    submit_pattern = re.compile(
        r"\bSUBMIT\s+(?P<report>[A-Za-z_][A-Za-z0-9_]*).*?\.",
        re.IGNORECASE | re.DOTALL,
    )

    for match in submit_pattern.finditer(scan_code):
        statement = match.group(0)
        line      = line_from_offset(scan_code, match.start())

        if re.search(r"\bWITH\s+\w+\s*=\s*['\"]", statement, re.IGNORECASE):
            add(
                "SEC_SUBMIT_HARDCODED_INPUT", "Security / Maintainability", "Medium",
                "Hardcoded SUBMIT input",
                "A called report receives hardcoded input values.",
                "Verify whether the values should come from validated input or controlled configuration.",
                line,
                existing_code=extract_lines(code, line, min(len(lines), line + 8)),
                confidence="Medium",
                developer_verification=True,
                why_it_matters="Hardcoded inputs silently constrain the called report to one business context.",
            )

    # --------------------------------------------------------
    # HARDCODED USER
    # --------------------------------------------------------

    for number, line in enumerate(lines, 1):
        if re.search(r"\bsy-uname\b\s*=\s*['\"]", line, re.IGNORECASE):
            add(
                "SEC_HARDCODED_USER", "Security", "High",
                "Hardcoded user-specific logic",
                "Program behavior depends on a hardcoded user identity.",
                "Do not use a hardcoded user ID as an authorization mechanism. Use authorization or controlled configuration.",
                number,
                existing_code=extract_lines(code, number),
                confidence="High",
                why_it_matters="User-specific hardcoding is fragile and creates unexpected behavior differences.",
            )

    # --------------------------------------------------------
    # EMPTY CATCH
    # --------------------------------------------------------

    empty_catch_pattern = re.compile(
        r"\bCATCH\s+[A-Za-z_][A-Za-z0-9_~]*\s*\.\s*ENDTRY\s*\.",
        re.IGNORECASE | re.DOTALL,
    )

    for match in empty_catch_pattern.finditer(scan_code):
        line = line_from_offset(scan_code, match.start())
        add(
            "ERR_EMPTY_CATCH", "Error Handling", "High",
            "Exception is swallowed by an empty CATCH",
            "An exception is caught but no handling, logging, recovery or propagation is performed.",
            "Handle the exception appropriately, propagate it or record actionable diagnostics.",
            line,
            existing_code=extract_lines(code, line, min(len(lines), line + 5)),
            confidence="High",
            why_it_matters="Runtime failures disappear silently while processing continues.",
        )

    # --------------------------------------------------------
    # BAPI
    # --------------------------------------------------------

    bapi_pattern = re.compile(
        r"\bCALL\s+FUNCTION\s+['\"](?P<bapi>BAPI_[A-Z0-9_]+)['\"]",
        re.IGNORECASE,
    )

    bapi_calls = []

    for match in bapi_pattern.finditer(scan_code):
        bapi_name = match.group("bapi")
        line      = line_from_offset(scan_code, match.start())
        bapi_calls.append((bapi_name, line))

        nearby = scan_code[match.end(): min(len(scan_code), match.end() + 1800)]

        # Find the end of this CALL FUNCTION statement (next dot
        # not inside a string) so we know what parameters were mapped.
        stmt_end_match = re.search(r"\.\s*$|\.", nearby, re.MULTILINE)
        stmt_body = nearby[: stmt_end_match.end()] if stmt_end_match else nearby[:400]

        # ── Case 1: RETURN = DATA(lt_xxx)  — inline declaration ──
        return_inline = re.search(
            r"\bRETURN\s*=\s*DATA\(([^)]+)\)", stmt_body, re.IGNORECASE,
        )

        # ── Case 2: RETURN = lt_xxx  — existing table ─────────────
        return_existing = re.search(
            r"\bRETURN\s*=\s*@?(?!DATA\()(?P<var>[A-Za-z_][A-Za-z0-9_]*)",
            stmt_body, re.IGNORECASE,
        )

        # ── Case 3: RETURN not mapped at all ──────────────────────
        return_mapped = bool(return_inline or return_existing)

        if not return_mapped:
            # RETURN table completely absent — errors are invisible
            add(
                "BAPI_RETURN_IGNORED", "BAPI / Error Handling", "High",
                "BAPI called without mapping the RETURN parameter",
                (
                    f"{bapi_name} is called but the RETURN parameter is not mapped. "
                    "Any errors, warnings or messages reported by the BAPI are silently lost."
                ),
                (
                    "Add a RETURN table parameter, evaluate it after the call, "
                    "and handle error/warning messages before committing."
                ),
                line,
                existing_code=extract_lines(code, line, min(len(lines), line + 10)),
                recommended_code=(
                    f"DATA lt_return TYPE TABLE OF bapiret2.\n\n"
                    f"CALL FUNCTION '{bapi_name}'\n"
                    f"  EXPORTING ...\n"
                    f"  TABLES\n"
                    f"    return = lt_return.\n\n"
                    f"READ TABLE lt_return INTO DATA(ls_err)\n"
                    f"  WITH KEY type = 'E'.\n"
                    f"IF sy-subrc = 0.\n"
                    f"  MESSAGE ls_err-message TYPE 'E'.\n"
                    f"ENDIF."
                ),
                confidence="High",
                why_it_matters=(
                    "Without a RETURN table the program cannot detect BAPI errors "
                    "and will silently continue or commit even when the BAPI failed."
                ),
            )
        else:
            # RETURN is mapped — check whether it is actually evaluated afterward
            return_name = (
                return_inline.group(1) if return_inline
                else return_existing.group("var")
            )
            after_return = nearby[
                (return_inline or return_existing).end():
            ]

            return_checked = bool(
                re.search(rf"\bLOOP\s+AT\s+{re.escape(return_name)}\b",
                          after_return, re.IGNORECASE)
                or re.search(rf"\bREAD\s+TABLE\s+{re.escape(return_name)}\b",
                             after_return, re.IGNORECASE)
                or re.search(rf"\bIF\b.*?\b{re.escape(return_name)}\b",
                             after_return, re.IGNORECASE | re.DOTALL)
                or re.search(rf"\bLINES\(\s*{re.escape(return_name)}\s*\)\b",
                             after_return, re.IGNORECASE)
            )

            if not return_checked:
                add(
                    "BAPI_RETURN_IGNORED", "BAPI / Error Handling", "High",
                    "BAPI return messages are not evaluated",
                    (
                        f"{bapi_name} maps its RETURN parameter to '{return_name}', "
                        "but the table is not visibly evaluated after the call."
                    ),
                    "Loop through the RETURN table, check for E/A type messages and handle errors before committing.",
                    line,
                    existing_code=extract_lines(code, line, min(len(lines), line + 12)),
                    recommended_code=(
                        f"READ TABLE {return_name} INTO DATA(ls_err)\n"
                        f"  WITH KEY type = 'E'.\n"
                        f"IF sy-subrc = 0.\n"
                        f"  MESSAGE ls_err-message TYPE 'E'.\n"
                        f"ENDIF."
                    ),
                    confidence="High",
                    why_it_matters=(
                        "The program can continue or commit even when the BAPI "
                        "reports an error in the RETURN table."
                    ),
                )

    # --------------------------------------------------------
    # BAPI + COMMIT inside loop
    # --------------------------------------------------------

    for bapi_name, bapi_line in bapi_calls:
        for start, end in loop_ranges:
            if not (start <= bapi_line <= end):
                continue
            for number in range(bapi_line, end + 1):
                if re.search(r"\bCOMMIT\s+WORK\b", lines[number - 1], re.IGNORECASE):
                    add(
                        "BAPI_COMMIT_IN_LOOP", "BAPI / LUW", "High",
                        "BAPI change is committed inside loop",
                        f"{bapi_name} is executed and followed by COMMIT WORK during loop processing.",
                        "Evaluate BAPI return messages and define the intended transaction boundary.",
                        bapi_line, number,
                        existing_code=extract_lines(code, bapi_line, number),
                        confidence="High",
                        why_it_matters="Each iteration becomes a separate transaction, creating partial-success risk.",
                    )
                    break

    # --------------------------------------------------------
    # INTERNAL TABLE MODIFICATION DURING LOOP
    # --------------------------------------------------------

    itab_mod_pattern = re.compile(
        r"\b(DELETE|MODIFY|INSERT|APPEND)\s+(?P<table>[A-Za-z_][A-Za-z0-9_]*)",
        re.IGNORECASE,
    )

    for match in itab_mod_pattern.finditer(scan_code):
        table_name = match.group("table")
        line       = line_from_offset(scan_code, match.start())

        preceding = scan_code[max(0, match.start() - 20): match.start()]
        if re.search(r"(DELETE\s+FROM|INSERT\s+INTO)", preceding, re.IGNORECASE):
            continue

        if any(s <= line <= e for s, e in loop_ranges):
            add(
                "ITAB_MODIFY_DURING_LOOP", "Logic / Internal Tables", "Medium",
                "Internal table is modified during iteration",
                f"{table_name} is modified while loop processing is active.",
                "Use a separate result/filtering strategy or explicitly control iteration semantics.",
                line,
                existing_code=extract_lines(code, line, min(len(lines), line + 7)),
                confidence="High",
                why_it_matters="Changing the table being processed can alter which records are visited.",
            )

    # --------------------------------------------------------
    # HARDCODED BUSINESS VALUE LISTS
    # --------------------------------------------------------

    comparison_pattern = re.compile(
        r"""
        (?P<field>[A-Za-z_][A-Za-z0-9_]*(?:-[A-Za-z0-9_]+)+)
        \s*=\s*
        ['"](?P<value>[^'"]+)['"]
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    field_values       = {}
    business_list_ranges = []

    for match in comparison_pattern.finditer(scan_code):
        field = match.group("field").lower()
        value = match.group("value")
        line  = line_from_offset(scan_code, match.start())
        field_values.setdefault(field, []).append({"value": value, "line": line})

    for field, values in field_values.items():
        unique_values = list(dict.fromkeys(item["value"] for item in values))
        if len(unique_values) < 2:
            continue
        lines_used = [item["line"] for item in values]
        first_line = min(lines_used)
        last_line  = max(lines_used)
        if len(unique_values) >= 3 or last_line - first_line <= 30:
            business_list_ranges.append((field, first_line, last_line))
            add(
                "MAINT_HARDCODED_BUSINESS_LIST", "Maintainability / Configuration", "Medium",
                "Hardcoded list of business values",
                f"Multiple literal values are compared against {field}: {', '.join(unique_values)}.",
                "Determine whether these are fixed technical values or business configuration. Consider a range, parameter, TVARVC, customizing or configuration table.",
                first_line, last_line,
                existing_code=extract_lines(code, first_line, last_line),
                confidence="High",
                developer_verification=True,
                why_it_matters="Business lists embedded in source code require a code change and transport when the list changes.",
            )

    # --------------------------------------------------------
    # ORGANIZATIONAL LITERALS
    # --------------------------------------------------------

    org_fields = ("werks", "bukrs", "ekorg", "ekgrp", "vkorg", "vtweg", "spart", "lgort")

    for number, line in enumerate(lines, 1):
        for field in org_fields:
            if re.search(rf"\b{field}\b\s*=\s*['\"]", line, re.IGNORECASE):
                if any(lf == field and f <= number <= l for lf, f, l in business_list_ranges):
                    continue
                add(
                    "MAINT_HARDCODED_ORG", "Maintainability / Configuration", "Medium",
                    "Hardcoded organizational value",
                    f"{field.upper()} is compared or assigned using a literal value.",
                    "Verify whether the value should come from validated input or controlled configuration.",
                    number,
                    existing_code=extract_lines(code, number),
                    confidence="Medium",
                    developer_verification=True,
                    why_it_matters="Organizational values commonly vary by system, client or business configuration.",
                )
                break

    # --------------------------------------------------------
    # PARAMETER DEFAULTS
    # --------------------------------------------------------

    parameter_pattern = re.compile(
        r"""
        (?P<name>p_[A-Za-z_][A-Za-z0-9_]*)
        \s+TYPE.*?DEFAULT\s*['"](?P<value>[^'"]+)['"]
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    org_tokens = ("werks", "bukrs", "ekorg", "ekgrp", "vkorg", "vtweg", "spart", "lgort", "matnr")

    for match in parameter_pattern.finditer(scan_code):
        name  = match.group("name")
        value = match.group("value")
        if not any(token in name.lower() for token in org_tokens):
            continue
        line = line_from_offset(scan_code, match.start())
        add(
            "MAINT_PARAMETER_DEFAULT", "Maintainability / Configuration", "Medium",
            "Business parameter has hardcoded default",
            f"{name} has hardcoded default value '{value}'.",
            "Verify whether this default is genuinely required or should be configuration-driven.",
            line,
            existing_code=extract_lines(code, line),
            confidence="Medium",
            developer_verification=True,
            why_it_matters="Hardcoded defaults silently constrain the program to one business context.",
        )

    # --------------------------------------------------------
    # CONSTANTS WITH POSSIBLE BUSINESS VALUES
    # --------------------------------------------------------

    constant_pattern = re.compile(
        r"""\bCONSTANTS?\b.*?\bVALUE\s*['"](?P<value>[^'"]+)['"]""",
        re.IGNORECASE | re.DOTALL,
    )

    for match in constant_pattern.finditer(scan_code):
        value = match.group("value")
        looks_business_like = bool(re.match(r"^(?:[A-Z]{1,8}\d{2,}|\d{2,}|[A-Z]{2,}\d+)$", value))
        if not looks_business_like:
            continue
        line = line_from_offset(scan_code, match.start())
        add(
            "MAINT_CONSTANT_BUSINESS_VALUE", "Maintainability / Configuration", "Low",
            "Constant may contain a business value",
            f"The constant contains '{value}'. Using a constant is better than repeating a literal, but the value may still represent business configuration.",
            "Verify whether the value is truly immutable technical information or should be maintained as configuration.",
            line,
            existing_code=extract_lines(code, line, min(len(lines), line + 5)),
            confidence="Medium",
            developer_verification=True,
            why_it_matters="Constants still require source-code change when the business value changes.",
        )

    # --------------------------------------------------------
    # BUSINESS THRESHOLDS / RATES
    # --------------------------------------------------------

    for number, line in enumerate(lines, 1):
        if re.search(
            r"""
            \*\s*(?:1\.\d+|\d+\.\d+)
            |/\s*(?:1\.\d+|\d+\.\d+)
            |>\s*\d{2,}
            |<\s*\d{2,}
            """,
            line, re.IGNORECASE | re.VERBOSE,
        ):
            add(
                "MAINT_BUSINESS_LITERAL", "Business Logic", "Medium",
                "Potential hardcoded business rule or threshold",
                "A numeric literal appears in a calculation or business decision.",
                "Verify whether the value is a fixed technical value or a configurable business rule.",
                number,
                existing_code=extract_lines(code, number),
                confidence="Medium",
                developer_verification=True,
                why_it_matters="Unexplained thresholds and rates hide business rules that require controlled maintenance.",
            )

    # --------------------------------------------------------
    # LEGACY ABAP
    # --------------------------------------------------------

    legacy_match = re.search(r"\b(FORM|PERFORM|MOVE|COMPUTE)\b", scan_code, re.IGNORECASE)
    if legacy_match:
        line = line_from_offset(scan_code, legacy_match.start())
        add(
            "MOD_LEGACY_SYNTAX", "Modern ABAP", "Low",
            "Legacy procedural syntax detected",
            "Legacy procedural constructs are present.",
            "Modernize where doing so improves readability, testability or maintainability.",
            line,
            existing_code=extract_lines(code, line),
            confidence="High",
            why_it_matters="Modern ABAP can improve encapsulation and readability.",
        )

    # --------------------------------------------------------
    # COMMENTED-OUT CODE
    # --------------------------------------------------------

    commented_code_lines = []
    for number, line in enumerate(original_lines, 1):
        stripped = line.strip()
        if stripped.startswith("*") or stripped.startswith('"'):
            if re.search(r"\b(SELECT|LOOP|IF|CALL|DATA|READ|UPDATE|DELETE|MODIFY)\b", stripped, re.IGNORECASE):
                commented_code_lines.append(number)

    if len(commented_code_lines) >= 2:
        add(
            "MAINT_COMMENTED_CODE", "Maintainability", "Low",
            "Commented-out code detected",
            "Multiple sections appear to contain disabled code.",
            "Remove obsolete code and rely on version control for historical implementations.",
            commented_code_lines[0],
            existing_code=extract_lines(code, commented_code_lines[0]),
            confidence="High",
            why_it_matters="Commented-out implementations increase noise and make active logic harder to understand.",
        )

    # --------------------------------------------------------
    # DOCUMENTATION
    # --------------------------------------------------------

    if executable_lines > 30 and comment_lines == 0:
        add(
            "DOC_NO_WHY_COMMENTS", "Documentation", "Low",
            "No explanatory comments detected",
            "The source contains no visible comments explaining important business or technical decisions.",
            "Document WHY-level business rules, assumptions, workarounds and non-obvious decisions.",
            confidence="Medium",
            developer_verification=True,
            why_it_matters="Important intent can be lost during future maintenance.",
        )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = {
        "lines":                  len(original_lines),
        "code_lines":             executable_lines,
        "comment_lines":          comment_lines,
        "selects":                len(select_matches),
        "loops":                  len(re.findall(r"\bLOOP\s+AT\b", scan_code, re.IGNORECASE)),
        "reads":                  len(re.findall(r"\bREAD\s+TABLE\b", scan_code, re.IGNORECASE)),
        "authority_checks":       len(authority_matches),
        "commits":                len(re.findall(r"\bCOMMIT\s+WORK\b", scan_code, re.IGNORECASE)),
        "database_modifications": db_modification_count,
        "bapi_calls":             len(bapi_calls),
        "function_calls":         len(re.findall(r"\bCALL\s+FUNCTION\b", scan_code, re.IGNORECASE)),
        "constants":              len(re.findall(r"\bCONSTANTS?\b", scan_code, re.IGNORECASE)),
    }

    # Apply severity floors
    findings = [apply_severity_floor(item) for item in findings]

    return findings, metrics


# ============================================================
# COMPACT STATIC SIGNALS  (sent to AI — kept small)
# ============================================================

def build_static_signals(findings, metrics):
    counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    rules  = []

    for finding in findings:
        severity = normalize_severity(finding.get("severity"))
        counts[severity] += 1
        rule_id = finding.get("id")
        if rule_id:
            rules.append(rule_id)

    # Also send finding titles so AI dedup is semantically grounded
    titles = [
        finding.get("title", "")[:60]
        for finding in findings[:30]
    ]

    return {
        "metrics":        metrics,
        "finding_counts": counts,
        "signals":        rules[:60],
        "titles":         titles,
    }


# ============================================================
# AI REVIEW PROMPT  — trimmed to fit 8K/min budget
# ============================================================

def build_analysis_prompt(code, static_signals):
    return f"""You are a strict SAP ABAP production-code reviewer.

The deterministic scanner already detected the issues listed below.
DO NOT repeat them. Find ADDITIONAL semantic issues the static rules missed.

ALREADY DETECTED:
{json.dumps(static_signals, separators=(",", ":"))}

REVIEW THESE AREAS for issues NOT already listed:
1. DATABASE — missing WHERE filters, large-volume risks, duplicate driver keys, unnecessary joins
2. INTERNAL TABLES — wrong table type for access pattern, large-volume sort/read combinations
3. LOGIC — contradictory conditions, unreachable branches, overwritten values, missing boundary checks
4. SECURITY — dynamic SQL, input validation gaps, missing authorization on sensitive paths
5. LUW — partial-success paths, locks without matching dequeue, rollback coverage
6. BAPI/FM — return/exception paths not visible to the scanner
7. HARD-CODING — thresholds, rates, doc types, movement types, status codes not already flagged
8. BUSINESS LOGIC — calculations, percentages, boundary conditions, duplicated rules
9. MAINTAINABILITY — duplicate logic, excessive nesting, unclear interfaces, dead code

After first pass: re-read the source, deduplicate, verify severity.
Only report issues directly supported by the supplied source.
Do NOT invent tables, fields, SAP APIs or business requirements.

SEVERITY: Critical=data-integrity/security | High=production/correctness risk | Medium=meaningful issue | Low=minor opportunity
CONFIDENCE: High=proven from source | Medium=context-dependent | Low=needs verification

Return ONLY valid JSON:
{{
  "summary": "",
  "primary_risk": "",
  "issues": [
    {{
      "id": "",
      "severity": "Critical|High|Medium|Low",
      "category": "",
      "title": "",
      "location": "",
      "existing_code": "",
      "problem": "",
      "why_it_matters": "",
      "recommendation": "",
      "recommended_code": "",
      "confidence": "High|Medium|Low",
      "developer_verification": false
    }}
  ]
}}

ABAP SOURCE:
<ABAP_CODE>
{code}
</ABAP_CODE>"""


# ============================================================
# FINDING DEDUPLICATION / VALIDATION
# ============================================================

def _normalize_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _finding_key(finding):
    rule       = _normalize_text(finding.get("id"))
    title      = _normalize_text(finding.get("title"))
    start_line = safe_int(finding.get("start_line"), 0)
    code_block = _normalize_text(finding.get("existing_code"))

    # Use finer buckets (5-line) to avoid merging distinct nested loops
    if rule == "itab_nested_loop" and start_line:
        return rule, title, start_line // 5

    if code_block:
        return rule, title, code_block

    return rule, title, start_line


def _parse_location(location):
    text  = str(location or "")
    match = re.search(r"(?:lines?|line)\s*(\d+)\s*(?:[-–]\s*(\d+))?", text, re.IGNORECASE)
    if not match:
        return None, None
    start = safe_int(match.group(1), 0)
    end   = safe_int(match.group(2), start) if match.group(2) else start
    return start, end


def _find_code_location(code, snippet):
    snippet = str(snippet or "").strip()
    if not snippet:
        return None, None

    offset = code.find(snippet)
    if offset >= 0:
        start = line_from_offset(code, offset)
        end   = start + snippet.count("\n")
        return start, end

    normalized_snippet = _normalize_text(snippet)
    if not normalized_snippet:
        return None, None

    source_lines   = code.splitlines()
    snippet_lines  = [x.strip() for x in snippet.splitlines() if x.strip()]
    if not snippet_lines:
        return None, None

    normalized_target = " ".join(snippet_lines).lower()
    window_size       = len(snippet_lines)

    for index in range(0, len(source_lines) - window_size + 1):
        window = " ".join(
            line.strip() for line in source_lines[index:index + window_size] if line.strip()
        ).lower()
        if window == normalized_target:
            return index + 1, index + window_size

    return None, None


def _prepare_ai_finding(code, item):
    if not isinstance(item, dict):
        return None

    finding      = dict(item)
    source_lines = len(code.splitlines())

    finding["severity"] = normalize_severity(finding.get("severity"))
    finding.setdefault("category",               "AI Review")
    finding.setdefault("title",                  "Semantic ABAP issue")
    finding.setdefault("problem",                "")
    finding.setdefault("why_it_matters",         "")
    finding.setdefault("recommendation",         "")
    finding.setdefault("recommended_code",       "")
    finding.setdefault("confidence",             "Medium")
    finding.setdefault("developer_verification", True)
    finding.setdefault("existing_code",          "")

    snippet_start, snippet_end = _find_code_location(code, finding.get("existing_code", ""))

    if snippet_start:
        finding["start_line"]    = snippet_start
        finding["end_line"]      = snippet_end
        finding["location"]      = location_text(snippet_start, snippet_end)
        finding["existing_code"] = extract_lines(code, snippet_start, snippet_end)
        return finding

    reported_start, reported_end = _parse_location(finding.get("location", ""))

    if reported_start and 1 <= reported_start <= source_lines:
        reported_end             = min(reported_end or reported_start, source_lines)
        finding["start_line"]    = reported_start
        finding["end_line"]      = reported_end
        finding["location"]      = location_text(reported_start, reported_end)
        finding["existing_code"] = extract_lines(code, reported_start, reported_end)
        return finding

    finding["start_line"]    = None
    finding["end_line"]      = None
    finding["location"]      = "Code context"
    finding["existing_code"] = ""
    return finding


# All deterministic rule IDs — AI must never re-report these
DETERMINISTIC_IDS = {
    "DB_SELECT_STAR", "DB_SELECT_IN_LOOP", "DB_SELECT_SINGLE_PARTIAL",
    "DB_DYNAMIC_SELECT", "DB_FAE_CHECK",
    "ITAB_REPEATED_LOOKUP", "ITAB_NESTED_LOOP", "ITAB_SORT_MISSING",
    "DB_DIRECT_UPDATE", "DB_DIRECT_DELETE", "DB_DIRECT_INSERT", "DB_DIRECT_MODIFY",
    "DB_MOD_IN_LOOP", "LUW_COMMIT_IN_LOOP", "LUW_COMMIT_IN_LOOP_RAW",
    "SEC_AUTH_RESULT_IGNORED", "SEC_DYNAMIC_TRANSACTION", "SEC_CALL_TRANSACTION",
    "SEC_SUBMIT_HARDCODED_INPUT", "SEC_HARDCODED_USER",
    "ERR_EMPTY_CATCH", "FM_NO_EXCEPTION_HANDLING", "FM_OTHERS_ZERO",
    "BAPI_RETURN_IGNORED", "BAPI_COMMIT_IN_LOOP",
    "ITAB_MODIFY_DURING_LOOP",
    "MAINT_HARDCODED_BUSINESS_LIST", "MAINT_HARDCODED_ORG",
    "MAINT_PARAMETER_DEFAULT", "MAINT_CONSTANT_BUSINESS_VALUE",
    "MAINT_BUSINESS_LITERAL", "MOD_LEGACY_SYNTAX",
    "MAINT_COMMENTED_CODE", "DOC_NO_WHY_COMMENTS",
    "LOGIC_DIVISION_BY_ZERO", "LOGIC_VARIABLE_DIVISOR",
    "LOGIC_STALE_SY_SUBRC", "LOGIC_SY_TABIX_STALE",
    "DB_SELECT_NO_SUBRC_CHECK",
}


def _same_deterministic_issue(static_finding, ai_finding):
    static_id    = str(static_finding.get("id") or "")
    static_title = _normalize_text(static_finding.get("title"))
    ai_title     = _normalize_text(ai_finding.get("title"))

    if static_id not in DETERMINISTIC_IDS:
        return False

    if static_title == ai_title:
        return True

    return SequenceMatcher(None, static_title, ai_title).ratio() >= 0.88


def finding_similarity(a, b):
    title_a    = _normalize_text(a.get("title"))
    title_b    = _normalize_text(b.get("title"))
    category_a = _normalize_text(a.get("category"))
    category_b = _normalize_text(b.get("category"))
    line_a     = safe_int(a.get("start_line"), 0)
    line_b     = safe_int(b.get("start_line"), 0)

    title_score    = SequenceMatcher(None, title_a, title_b).ratio()
    category_score = 1.0 if category_a == category_b else 0.0
    line_score     = 1.0 if line_a and line_b and abs(line_a - line_b) <= 5 else 0.0

    return title_score * 0.55 + category_score * 0.20 + line_score * 0.25


def _deduplicate_static_findings(findings):
    final = []
    seen  = set()
    for finding in findings:
        item = dict(finding)
        key  = _finding_key(item)
        if key in seen:
            continue
        seen.add(key)
        final.append(apply_severity_floor(item))
    return final


def merge_findings(static_findings, ai_findings, code):
    final = _deduplicate_static_findings(static_findings)

    for raw_item in ai_findings:
        ai_item = _prepare_ai_finding(code, raw_item)
        if not ai_item:
            continue

        if any(_same_deterministic_issue(existing, ai_item) for existing in final):
            continue

        duplicate = None
        for existing in final:
            # Tighter threshold (0.88) to avoid collapsing distinct findings
            if finding_similarity(existing, ai_item) >= 0.88:
                duplicate = existing
                break

        if duplicate:
            duplicate["severity"] = higher_severity(duplicate.get("severity"), ai_item.get("severity"))
            for field in ("problem", "why_it_matters", "recommendation", "recommended_code"):
                if not duplicate.get(field):
                    duplicate[field] = ai_item.get(field, "")
            if ai_item.get("confidence") == "High":
                duplicate["confidence"] = "High"
        else:
            final.append(ai_item)

    final = _deduplicate_static_findings(final)
    for finding in final:
        apply_severity_floor(finding)

    final.sort(key=lambda item: (
        -SEVERITY_RANK[normalize_severity(item.get("severity"))],
        safe_int(item.get("start_line"), 999999),
    ))
    return final


# ============================================================
# HEALTH SCORE  — rebalanced penalties
# ============================================================

def calculate_health_score(findings):
    score = 100

    # Proportional penalties — Low is now 1pt (was 2pt) to avoid
    # doc/style findings dominating the score
    penalties = {
        "Critical": 25,
        "High":     12,
        "Medium":    5,
        "Low":       1,
    }

    for finding in findings:
        score -= penalties[normalize_severity(finding.get("severity"))]

    score = max(0, min(100, score))

    # Hard caps — exponential degradation for multiple Criticals
    critical_count = sum(1 for x in findings if normalize_severity(x.get("severity")) == "Critical")
    high_count     = sum(1 for x in findings if normalize_severity(x.get("severity")) == "High")

    if   critical_count >= 4: score = min(score, 19)
    elif critical_count == 3: score = min(score, 29)
    elif critical_count == 2: score = min(score, 39)
    elif critical_count == 1: score = min(score, 59)
    elif high_count     >= 5: score = min(score, 59)
    elif high_count     >= 3: score = min(score, 69)
    elif high_count     >= 1: score = min(score, 79)

    return score


# ============================================================
# AI ANALYSIS
# ============================================================

def analyze_abap(code):
    static_findings, metrics = local_abap_scan(code)
    static_signals           = build_static_signals(static_findings, metrics)

    prompt           = build_analysis_prompt(code, static_signals)
    estimated_input  = estimate_tokens(
        "You are a strict SAP ABAP code reviewer. Return only valid JSON.\n" + prompt
    )

    budget_check(estimated_input, ANALYSIS_MAX_OUT, "Analysis")

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a strict SAP ABAP code reviewer. Return only valid JSON."},
                {"role": "user",   "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=ANALYSIS_MAX_OUT,
        )
    except Exception as exc:
        err = str(exc)
        if "json_validate_failed" in err or "Failed to validate JSON" in err:
            raise ValueError(
                "The model could not fit its response within the token limit and returned "
                "incomplete JSON. Try analyzing a smaller section of the code, or the source "
                "may be too large for the current token budget."
            ) from exc
        if "model_not_found" in err or "does not exist" in err.lower():
            raise ValueError(
                f"Model '{MODEL}' is not available on your Groq API key. "
                "Update the MODEL constant in app.py to a model available on your account "
                "(e.g. llama-3.3-70b-versatile)."
            ) from exc
        raise

    raw = response.choices[0].message.content
    if not raw:
        raise ValueError("The AI returned an empty response.")

    try:
        ai_result = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        try:
            ai_result = json.loads(cleaned.strip())
        except json.JSONDecodeError:
            raise ValueError(
                "The AI returned malformed JSON. This usually means the response was "
                "truncated mid-output. Try a smaller ABAP source or increase ANALYSIS_MAX_OUT."
            )

    if not isinstance(ai_result, dict):
        raise ValueError("AI response was not a JSON object.")

    ai_findings = ai_result.get("issues", [])
    if not isinstance(ai_findings, list):
        ai_findings = []

    findings = merge_findings(static_findings, ai_findings, code)
    score    = calculate_health_score(findings)
    summary  = ai_result.get("summary", "")
    primary_risk = ai_result.get("primary_risk", "")

    if not summary:
        summary = (
            f"The Doctor identified {len(findings)} actionable finding(s), including "
            f"{sum(1 for x in findings if x['severity'] == 'Critical')} Critical and "
            f"{sum(1 for x in findings if x['severity'] == 'High')} High-severity issue(s)."
        )

    if not primary_risk:
        if any(x["severity"] == "Critical" for x in findings):
            primary_risk = "Critical data-integrity or security risk detected."
        elif any(x["severity"] == "High" for x in findings):
            primary_risk = "High-severity production risk detected."
        else:
            primary_risk = "No Critical or High-severity issue was detected by the current review."

    return {
        "summary":      summary,
        "primary_risk": primary_risk,
        "health_score": score,
        "health_label": health_label(score),
        "issues":       findings,
        "metrics":      metrics,
    }


# ============================================================
# FIX GENERATION  — tight prompt to respect 8K/min budget
# ============================================================

def build_fix_prompt(code, diagnosis):
    diagnosis = diagnosis if isinstance(diagnosis, dict) else {}
    issues    = diagnosis.get("issues", [])
    if not isinstance(issues, list):
        issues = []

    priority_order = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}

    compact_findings = []
    seen = set()

    for finding in sorted(
        (i for i in issues if isinstance(i, dict)),
        key=lambda i: -priority_order.get(normalize_severity(i.get("severity")), 2),
    ):
        severity       = normalize_severity(finding.get("severity"))
        title          = str(finding.get("title", "")).strip()
        location       = str(finding.get("location", "")).strip()
        recommendation = str(finding.get("recommendation", "")).strip()

        key = (severity.lower(), _normalize_text(title), _normalize_text(location))
        if key in seen:
            continue
        seen.add(key)

        compact_findings.append(f"{severity}|{title}|{location}|{recommendation[:120]}")

        if len(compact_findings) >= 18:
            break

    findings_text = "\n".join(compact_findings) or "No findings."

    return f"""You are a senior SAP ABAP developer.
Rewrite the COMPLETE ABAP source. Return ABAP source only — no markdown, no commentary.

RULES:
- Return the full program from first to last statement. Never abbreviate or omit sections.
- Fix each confirmed finding once. Do not invent SAP APIs or business requirements.
- For confirmed business hardcoding: declare a named CONSTANT and replace the executable literal.
- For context-dependent findings that cannot be safely rewritten: preserve behavior and add a short ABAP comment marking the point for developer review.
- Guard variable divisors. Move DB access out of loops where the source supports it.
- Check SY-SUBRC immediately after every operation that sets it.
- Add EXCEPTIONS blocks to CALL FUNCTION statements that lack them.

CONFIRMED FINDINGS:
{findings_text}

ABAP SOURCE:
<ABAP_CODE>
{code}
</ABAP_CODE>"""


def _strip_code_fences(text):
    fixed = (text or "").strip()
    for prefix in ("```abap", "```ABAP", "```"):
        if fixed.startswith(prefix):
            fixed = fixed[len(prefix):]
            break
    if fixed.endswith("```"):
        fixed = fixed[:-3]
    return fixed.strip()


def generate_fix(code, diagnosis):
    prompt = build_fix_prompt(code, diagnosis)

    system_msg      = "You are a senior SAP ABAP developer. Return only complete corrected ABAP source."
    estimated_input = estimate_tokens(system_msg + "\n" + prompt)

    budget_check(estimated_input, FIX_MAX_OUT, "Generate Fix")

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=FIX_MAX_OUT,
        )
    except Exception as exc:
        error_text = str(exc)
        if "413" in error_text or "tokens per minute" in error_text.lower() or "request too large" in error_text.lower():
            raise ValueError(
                "Generate Fix exceeded the Groq token/minute budget. "
                "Analyze the program in smaller logical sections."
            ) from exc
        raise

    choice        = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    fixed         = _strip_code_fences(choice.message.content or "")

    if finish_reason == "length":
        raise ValueError(
            "Generate Fix reached the model output limit before the complete ABAP source was returned. "
            "The partial result was not shown. Reduce the source size or analyze in sections."
        )

    if not fixed:
        raise ValueError("The AI returned an empty corrected ABAP program.")

    if not re.search(
        r"\b(REPORT|PROGRAM|CLASS|INTERFACE|FUNCTION-POOL|MODULE-POOL|TYPE-POOL)\b",
        fixed, re.IGNORECASE,
    ):
        raise ValueError("The generated response does not appear to contain a complete ABAP source.")

    return fixed


# ============================================================
# ASK ABAP DOCTOR  — compact chat prompt
# ============================================================

def build_chat_prompt(question, mode, code, diagnosis, history):
    diagnosis = diagnosis if isinstance(diagnosis, dict) else {}
    history   = history   if isinstance(history,   list)  else []
    code      = code      if isinstance(code,       str)   else ""

    style = (
        "Answer directly and concisely."
        if mode == "Direct"
        else "Give a detailed developer-oriented answer with ABAP examples where useful."
    )

    compact_findings = [
        {
            "severity": f.get("severity"),
            "title":    f.get("title"),
            "location": f.get("location"),
            "problem":  f.get("problem"),
        }
        for f in diagnosis.get("issues", [])[:20]
        if isinstance(f, dict)
    ]

    history_text = "".join(
        f"\nUSER: {p.get('question', '')}\nDOCTOR: {p.get('answer', '')}\n"
        for p in history[:4]
        if isinstance(p, dict)
    )

    return f"""You are Ask ABAP Doctor, a senior SAP ABAP developer.
{style}
Answer using the current ABAP source and diagnosis. Do not invent SAP fields, tables, APIs or business requirements.

CURRENT FINDINGS:
{json.dumps(compact_findings, separators=(",", ":"))}

RECENT CONVERSATION:
{history_text}

QUESTION:
{question}

ABAP SOURCE:
<ABAP_CODE>
{code}
</ABAP_CODE>"""


def ask_doctor(question, mode, code, diagnosis, history):
    diagnosis = diagnosis if isinstance(diagnosis, dict) else {}
    history   = history   if isinstance(history,   list)  else []
    code      = code      if isinstance(code,       str)   else ""

    system_msg      = "You are an expert SAP ABAP assistant."
    prompt          = build_chat_prompt(question, mode, code, diagnosis, history)
    estimated_input = estimate_tokens(system_msg + "\n" + prompt)

    budget_check(estimated_input, 1_200, "Chat")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=1_200,
    )

    return (response.choices[0].message.content or "I could not generate an answer.").strip()


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns([7, 2])

with header_left:
    st.markdown('<div class="doctor-title">🩺 ABAP Code Doctor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="doctor-subtitle">Strict ABAP code health, risk and optimization analysis</div>',
        unsafe_allow_html=True,
    )

with header_right:
    chat_button = "× Close Doctor" if st.session_state.show_chat else "💬 Ask Doctor"
    if st.button(chat_button, use_container_width=True):
        st.session_state.show_chat = not st.session_state.show_chat
        st.rerun()


# ============================================================
# LAYOUT
# ============================================================

if st.session_state.show_chat:
    main_col, chat_col = st.columns([68, 32], gap="large")
else:
    main_col = st.container()
    chat_col = None


# ============================================================
# CHAT PANEL
# ============================================================

if st.session_state.show_chat:
    with chat_col:
        st.markdown('<div class="chat-title">💬 Ask ABAP Doctor</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="chat-subtitle">Ask follow-up questions about ABAP or your analysis.</div>',
            unsafe_allow_html=True,
        )

        st.session_state.chat_mode = st.radio(
            "Response style",
            ["Direct", "Detailed"],
            horizontal=True,
            label_visibility="collapsed",
            key="chat_mode_radio",
        )

        with st.form("doctor_chat_form", clear_on_submit=True):
            question = st.text_input(
                "Ask another ABAP question",
                placeholder="Ask another ABAP question...",
                label_visibility="collapsed",
            )
            send = st.form_submit_button("➤ Ask Doctor", use_container_width=True)

        if send and question.strip():
            user_question    = question.strip()
            previous_history = st.session_state.chat_pairs.copy()

            with st.spinner("Thinking..."):
                try:
                    answer = ask_doctor(
                        user_question,
                        st.session_state.chat_mode,
                        st.session_state.analyzed_code or "",
                        st.session_state.analysis or {},
                        previous_history,
                    )
                except Exception as exc:
                    answer = f"Unable to answer right now: {exc}"

            st.session_state.chat_pairs.insert(0, {"question": user_question, "answer": answer})
            st.rerun()

        with st.container(height=560, border=True):
            if not st.session_state.chat_pairs:
                st.info("Ask about a finding, performance issue, ABAP syntax or your diagnosis.")
            else:
                for pair in st.session_state.chat_pairs:
                    st.markdown("**You**")
                    st.write(pair["question"])
                    st.markdown("**🩺 Doctor**")
                    st.markdown(pair["answer"])
                    st.divider()


# ============================================================
# MAIN WORKSPACE
# ============================================================

with main_col:
    st.subheader("📥 ABAP Code")

    input_mode = st.radio(
        "Input method",
        ["📋 Paste Code", "📄 Upload File"],
        horizontal=True,
        label_visibility="collapsed",
    )

    code = ""

    if input_mode == "📋 Paste Code":
        code = st.text_area(
            "ABAP source",
            height=390,
            placeholder="Paste ABAP source here...",
            label_visibility="collapsed",
            key="paste_code",
        )

    else:
        uploaded_file = st.file_uploader(
            "Upload ABAP source",
            type=["abap", "txt", "prog"],
            key="abap_upload",
            help="Supported: .abap, .txt, .prog",
        )

        if uploaded_file is not None:
            try:
                code = uploaded_file.getvalue().decode("utf-8", errors="replace")
                st.session_state.uploaded_filename = uploaded_file.name
                st.success(f"Loaded {uploaded_file.name} • {len(code.splitlines())} lines")

                with st.expander("Preview uploaded ABAP", expanded=False):
                    st.code(code[:12000], language="abap")
                    if len(code) > 12000:
                        st.caption("Preview truncated. The complete file will be analyzed.")

            except Exception as exc:
                st.error(f"Unable to read uploaded file: {exc}")
                code = ""

    if code.strip():
        token_estimate = estimate_tokens(code)
        st.caption(
            f"{len(code.splitlines())} lines • "
            f"{len(code):,} characters • "
            f"~{token_estimate} tokens"
        )

        # Warn early if code alone is very large
        if token_estimate > 3_500:
            st.warning(
                f"⚠️ Source is ~{token_estimate} tokens. "
                "Within the 8 000 token/min Groq limit, analysis may be tight. "
                "Consider analyzing in logical sections for very large programs."
            )

    with st.form("analyze_form", clear_on_submit=False):
        analyze_clicked = st.form_submit_button(
            "🩺 Analyze ABAP",
            type="primary",
            use_container_width=True,
        )

    if analyze_clicked:
        if not code.strip():
            st.warning("Paste or upload ABAP source first.")
        else:
            st.session_state.analysis     = None
            st.session_state.fixed_code   = ""
            st.session_state.analyzed_code = code
            st.session_state.chat_pairs   = []

            with st.spinner("🩺 Performing strict ABAP review..."):
                try:
                    diagnosis = analyze_abap(code)
                    st.session_state.analysis = diagnosis
                    st.success("Strict review completed.")
                except Exception as exc:
                    error_text = str(exc)
                    if "413" in error_text or "rate_limit_exceeded" in error_text or "tokens per minute" in error_text:
                        st.error(
                            "The AI request exceeded the Groq 8 000 token/min limit. "
                            "The static analysis findings are still shown below if analysis completed partially. "
                            "For large programs, analyze in logical sections."
                        )
                    else:
                        st.error(f"Analysis failed: {exc}")


# ============================================================
# RESULTS
# ============================================================

diagnosis = st.session_state.analysis

if diagnosis:
    findings = diagnosis.get("issues", [])

    critical = [x for x in findings if normalize_severity(x.get("severity")) == "Critical"]
    high     = [x for x in findings if normalize_severity(x.get("severity")) == "High"]
    medium   = [x for x in findings if normalize_severity(x.get("severity")) == "Medium"]
    low      = [x for x in findings if normalize_severity(x.get("severity")) == "Low"]

    # ========================================================
    # FINDINGS
    # ========================================================

    st.markdown("---")
    st.subheader("🔍 Findings")

    severity_tabs = st.tabs([
        f"🔴 Critical ({len(critical)})",
        f"🟠 High ({len(high)})",
        f"🟡 Medium ({len(medium)})",
        f"🟢 Low ({len(low)})",
    ])

    for tab, group in zip(severity_tabs, [critical, high, medium, low]):
        with tab:
            if not group:
                st.success("No findings in this severity level.")

            for finding in group:
                severity = normalize_severity(finding.get("severity"))
                title    = finding.get("title", "Finding")

                with st.expander(
                    f"{severity_icon(severity)} {title}",
                    expanded=(severity == "Critical"),
                ):
                    if finding.get("location"):
                        st.caption(f"📍 {finding['location']}")

                    if finding.get("existing_code"):
                        st.markdown("**Existing code**")
                        st.code(finding["existing_code"], language="abap")

                    if finding.get("problem"):
                        st.markdown("**Problem**")
                        st.write(finding["problem"])

                    if finding.get("why_it_matters"):
                        st.markdown("**Why it matters**")
                        st.write(finding["why_it_matters"])

                    if finding.get("recommendation"):
                        st.markdown("**Recommendation**")
                        st.write(finding["recommendation"])

                    if finding.get("recommended_code"):
                        st.markdown("**Recommended code**")
                        st.code(finding["recommended_code"], language="abap")

                    confidence = finding.get("confidence", "Medium")
                    st.caption(f"Confidence: {confidence}")

                    if finding.get("developer_verification", False):
                        st.warning(
                            "Developer verification recommended: "
                            "the Doctor does not have your SAP DDIC, configuration or business context."
                        )

    # ========================================================
    # DOCTOR'S ANALYSIS
    # ========================================================

    st.markdown("---")
    st.subheader("🩺 Doctor's Analysis")
    st.write(diagnosis.get("summary", ""))

    if diagnosis.get("primary_risk"):
        st.info("**Primary risk:** " + diagnosis["primary_risk"])

    score = safe_int(diagnosis.get("health_score", 0))

    health_col1, health_col2 = st.columns([1, 3])
    with health_col1:
        st.metric("Health Score", f"{score}/100")
    with health_col2:
        st.markdown(f"### {health_label(score)}")
        st.caption(
            "Heuristic Doctor score based on the findings detected in this review. "
            "Not an official SAP ATC score."
        )

    # ========================================================
    # CODE PROFILE
    # ========================================================

    metrics = diagnosis.get("metrics", {})

    with st.expander("🔬 Code Profile", expanded=False):
        profile = [
            ("Lines",          metrics.get("lines", 0)),
            ("SELECTs",        metrics.get("selects", 0)),
            ("Loops",          metrics.get("loops", 0)),
            ("READ TABLE",     metrics.get("reads", 0)),
            ("AUTHORITY-CHECK",metrics.get("authority_checks", 0)),
            ("COMMIT",         metrics.get("commits", 0)),
            ("DB Changes",     metrics.get("database_modifications", 0)),
            ("BAPIs",          metrics.get("bapi_calls", 0)),
            ("Function Calls", metrics.get("function_calls", 0)),
        ]

        cols = st.columns(4)
        for index, (name, value) in enumerate(profile):
            with cols[index % 4]:
                st.metric(name, value)

    # ========================================================
    # CORRECTED ABAP
    # ========================================================

    st.markdown("---")
    st.subheader("💊 Corrected ABAP")

    if st.button("💊 Generate Fix", use_container_width=True):
        with st.spinner("Generating corrected ABAP..."):
            try:
                st.session_state.fixed_code = generate_fix(
                    st.session_state.analyzed_code, diagnosis,
                )
            except Exception as exc:
                st.error(f"Could not generate corrected ABAP: {exc}")

    if st.session_state.fixed_code:
        st.code(st.session_state.fixed_code, language="abap")

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "⬇ Download .ABAP",
                data=st.session_state.fixed_code,
                file_name="ABAP_Code_Doctor_Fixed.abap",
                mime="text/plain",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "⬇ Download .TXT",
                data=st.session_state.fixed_code,
                file_name="ABAP_Code_Doctor_Fixed.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.warning(
            "Review and test generated code in the target SAP environment before implementation."
        )

    # ========================================================
    # AUDIT REPORT
    # ========================================================

    st.subheader("📋 Audit Report")

    report = [
        "ABAP CODE DOCTOR",
        "ABAP Code Health Analysis Report",
        "=" * 60,
        "",
        f"Health: {score}/100",
        f"Assessment: {health_label(score)}",
        "",
        "SUMMARY",
        "-" * 60,
        diagnosis.get("summary", ""),
        "",
        "PRIMARY RISK",
        "-" * 60,
        diagnosis.get("primary_risk", ""),
        "",
        "FINDINGS",
        "-" * 60,
    ]

    for finding in findings:
        report.append(f"[{normalize_severity(finding.get('severity'))}] {finding.get('title', '')}")
        report.append(f"Location: {finding.get('location', '')}")

        if finding.get("existing_code"):
            report.append("Existing code:")
            report.append(finding["existing_code"])

        report.append("Problem: "         + finding.get("problem",         ""))
        report.append("Why it matters: "  + finding.get("why_it_matters",  ""))
        report.append("Recommendation: "  + finding.get("recommendation",  ""))

        if finding.get("recommended_code"):
            report.append("Recommended code:")
            report.append(finding["recommended_code"])

        report.append("Confidence: " + str(finding.get("confidence", "Medium")))
        report.append("")

    report_text = "\n".join(report)

    st.download_button(
        "⬇ Download Audit Report",
        data=report_text,
        file_name="ABAP_Code_Doctor_Audit.txt",
        mime="text/plain",
        use_container_width=True,
    )