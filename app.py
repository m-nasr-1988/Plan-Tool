import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import defaultdict

st.set_page_config(page_title="Goal Plan", layout="wide")

BASE_COLS = [
    "id", "parent", "type", "name",
    "progress_pct", "weight", "start", "end",
    "owner", "status", "resources", "impediments"
]

DEFAULT_DATA = [
    {"id":"L","parent":"","type":"Main Goal","name":"Learning","progress_pct":0,"weight":1,"start":"","end":"","owner":"Me","status":"In Progress","resources":"","impediments":""},
    {"id":"L-1","parent":"L","type":"Goal","name":"Learn Python","progress_pct":0,"weight":3,"start":"2025-08-01","end":"2025-10-01","owner":"Me","status":"In Progress","resources":"","impediments":""},
    {"id":"L-1-1","parent":"L-1","type":"Point","name":"Finish 'Automate the Boring Stuff'","progress_pct":60,"weight":2,"start":"2025-08-01","end":"2025-09-15","owner":"Me","status":"In Progress","resources":"https://automatetheboringstuff.com","impediments":"Limited evening time"},
    {"id":"L-1-2","parent":"L-1","type":"Point","name":"Build 3 small scripts","progress_pct":30,"weight":1,"start":"2025-09-10","end":"2025-10-01","owner":"Me","status":"Not Started","resources":"","impediments":""},
    {"id":"H","parent":"","type":"Main Goal","name":"Health","progress_pct":0,"weight":1,"start":"","end":"","owner":"Me","status":"In Progress","resources":"","impediments":""},
    {"id":"H-1","parent":"H","type":"Goal","name":"Run 5K","progress_pct":0,"weight":2,"start":"2025-08-10","end":"2025-10-10","owner":"Me","status":"In Progress","resources":"","impediments":""},
    {"id":"H-1-1","parent":"H-1","type":"Point","name":"Couch-to-5K Week 1-4","progress_pct":50,"weight":1,"start":"2025-08-10","end":"2025-09-10","owner":"Me","status":"In Progress","resources":"","impediments":""},
    {"id":"H-1-2","parent":"H-1","type":"Point","name":"Join local 5K event","progress_pct":10,"weight":1,"start":"2025-09-25","end":"2025-10-10","owner":"Me","status":"Not Started","resources":"","impediments":""},
]

def ensure_columns(df):
    for c in BASE_COLS:
        if c not in df.columns:
            df[c] = "" if c in ["id","parent","type","name","owner","status","resources","impediments"] else np.nan
    df = df[BASE_COLS].copy()
    # Coerce types
    df["id"] = df["id"].astype(str).str.strip()
    df["parent"] = df["parent"].fillna("").astype(str).str.strip()
    df["type"] = df["type"].astype(str)
    df["name"] = df["name"].astype(str)
    df["progress_pct"] = pd.to_numeric(df["progress_pct"], errors="coerce").fillna(0.0).clip(0,100)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0).clip(lower=0.0)
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["owner"] = df["owner"].astype(str)
    df["status"] = df["status"].astype(str)
    return df

def build_children_map(df):
    m = defaultdict(list)
    for _, r in df.iterrows():
        p = (r["parent"] or "").strip()
        if p:
            m[p].append(r["id"])
    return m

def compute_levels(df):
    parent_map = dict(zip(df["id"], df["parent"]))
    def level(nid):
        seen = set()
        l = 0
        p = parent_map.get(nid, "")
        while p and p not in seen:
            seen.add(p); l += 1
            p = parent_map.get(p, "")
        return l
    df["level"] = df["id"].map(level)
    return df

def compute_paths(df):
    parent_map = dict(zip(df["id"], df["parent"]))
    def path(nid):
        parts = [nid]
        p = parent_map.get(nid, "")
        guard = 0
        while p and guard < 100:
            parts.append(p)
            p = parent_map.get(p, "")
            guard += 1
        return ">".join(parts[::-1])
    df["path"] = df["id"].map(path)
    return df

def compute_rollups(df):
    children = build_children_map(df)
    weights = dict(zip(df["id"], df["weight"]))
    base_prog = dict(zip(df["id"], df["progress_pct"]))
    memo = {}
    visiting = set()

    def dfs(nid):
        if nid in memo:
            return memo[nid]
        if nid in visiting:
            raise ValueError(f"Cycle detected at {nid}")
        visiting.add(nid)
        childs = children.get(nid, [])
        if not childs:
            val = float(base_prog.get(nid, 0.0))
        else:
            ws, ww = 0.0, 0.0
            for cid in childs:
                cp = dfs(cid)
                w = float(weights.get(cid, 1.0))
                ws += cp * w
                ww += w
            val = (ws/ww) if ww > 0 else float(base_prog.get(nid, 0.0))
        visiting.remove(nid)
        memo[nid] = val
        return val

    vals = {}
    for nid in df["id"]:
        try:
            vals[nid] = dfs(nid)
        except ValueError:
            vals[nid] = np.nan
    df["progress_rollup"] = df["id"].map(vals)
    return df

def validate(df):
    msgs = []
    # Duplicates
    dups = df["id"][df["id"].duplicated()].unique().tolist()
    if dups:
        msgs.append(f"Duplicate IDs: {', '.join(dups)}")
    # Missing parents
    all_ids = set(df["id"])
    missing = sorted(set([p for p in df["parent"] if p]) - all_ids)
    if missing:
        msgs.append(f"Missing parent rows for: {', '.join(missing)}")
    # Self-parent
    selfp = df[df["id"] == df["parent"]]["id"].tolist()
    if selfp:
        msgs.append(f"Self-parent not allowed: {', '.join(selfp)}")
    return msgs

# Load data (plan.csv if exists)
@st.cache_data
def load_initial():
    try:
        df = pd.read_csv("plan.csv", dtype=str, keep_default_na=False)
        # fix types after load
        return ensure_columns(df)
    except Exception:
        return pd.DataFrame(DEFAULT_DATA)

if "df" not in st.session_state:
    st.session_state.df = ensure_columns(load_initial())

st.title("Goal Plan — hierarchy, rollups, and visuals")

with st.sidebar:
    st.subheader("Filters")
    # Filters (after computing)
    st.caption("Use table below to edit. Add new rows at the bottom.")
    if st.button("Save to plan.csv"):
        st.session_state.df.to_csv("plan.csv", index=False)
        st.success("Saved to plan.csv")

# Recompute derived columns each run
df = ensure_columns(st.session_state.df)
df = compute_levels(df)
df = compute_paths(df)
df = compute_rollups(df)

# Sidebar filters
with st.sidebar:
    owners = sorted([o for o in df["owner"].dropna().unique() if o])
    statuses = sorted([s for s in df["status"].dropna().unique() if s])
    sel_owners = st.multiselect("Owner", owners, default=owners)
    sel_status = st.multiselect("Status", statuses, default=statuses if statuses else [])
    sel_types = st.multiselect("Type", sorted(df["type"].unique()), default=sorted(df["type"].unique()))
    filt = pd.Series(True, index=df.index)
    if sel_owners:
        filt &= df["owner"].isin(sel_owners)
    if sel_status:
        filt &= df["status"].isin(sel_status)
    if sel_types:
        filt &= df["type"].isin(sel_types)
    df_view = df[filt].copy()

# Top metrics
roots = df[df["parent"] == ""].copy()
if not roots.empty:
    overall = (roots["progress_rollup"] * roots["weight"]).sum() / max(roots["weight"].sum(), 1)
else:
    overall = (df["progress_rollup"] * df["weight"]).sum() / max(df["weight"].sum(), 1)
col1, col2, col3 = st.columns(3)
col1.metric("Overall progress", f"{overall:.0f}%")
col2.metric("Items", f"{len(df):,}")
col3.metric("Leaf items (Points)", f"{(~df['id'].isin(df['parent'])).sum():,}")

# Validation messages
msgs = validate(df)
for m in msgs:
    st.warning(m)

tabs = st.tabs(["Table", "Treemap", "Timeline"])

with tabs[0]:
    st.subheader("Edit your plan")
    # Pretty indent for display-only
    disp = df_view.copy()
    disp["name (tree)"] = disp.apply(lambda r: ("  " * int(r["level"])) + ("↳ " if r["level"]>0 else "") + r["name"], axis=1)
    # Arrange columns for display
    show_cols = ["id","parent","type","name (tree)","progress_pct","weight","progress_rollup","start","end","owner","status","resources","impediments"]
    # Editor (computed cols disabled)
    edited = st.data_editor(
        disp[show_cols],
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "progress_pct": st.column_config.NumberColumn("Progress %", min_value=0, max_value=100, step=1),
            "weight": st.column_config.NumberColumn("Weight", min_value=0, step=0.1),
            "start": st.column_config.DateColumn("Start"),
            "end": st.column_config.DateColumn("End"),
            "status": st.column_config.SelectboxColumn("Status", options=["Not Started","In Progress","Blocked","Done"]),
            "type": st.column_config.SelectboxColumn("Type", options=["Main Goal","Goal","Point"]),
            "resources": st.column_config.TextColumn("Resources"),
            "impediments": st.column_config.TextColumn("Impediments"),
            "progress_rollup": st.column_config.NumberColumn("Rollup %", disabled=True, format="%.0f"),
            "name (tree)": st.column_config.TextColumn("Name (tree)", help="Auto-indented", disabled=True),
        }
    )
    # Push edits back into session_state (map columns)
    # Only update editable fields
    upd = edited.rename(columns={"name (tree)":"name"})
    for col in ["id","parent","type","name","progress_pct","weight","start","end","owner","status","resources","impediments"]:
        if col in upd.columns:
            st.session_state.df[col] = upd[col]

with tabs[1]:
    st.subheader("Treemap (size=Weight, color=Progress)")
    treedf = df.copy()
    treedf["parent_plot"] = treedf["parent"].replace({"": None})
    fig = px.treemap(
        treedf,
        ids="id",
        parents="parent_plot",
        values="weight",
        color="progress_rollup",
        color_continuous_scale="RdYlGn",
        range_color=(0,100),
        hover_data={"name": True, "type": True, "progress_rollup": ":.0f", "weight": True, "owner": True, "status": True}
    )
    fig.update_layout(margin=dict(t=30,l=10,r=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("Timeline (leaves only)")
    leaves_mask = ~df["id"].isin(df["parent"])
    leaves = df[leaves_mask].dropna(subset=["start","end"]).copy()
    if leaves.empty:
        st.info("Add Start and End dates to leaf items (Points) to see the timeline.")
    else:
        leaves = leaves.sort_values("start")
        tl = px.timeline(
            leaves,
            x_start="start", x_end="end",
            y="name",
            color="progress_rollup",
            color_continuous_scale="RdYlGn",
            range_color=(0,100),
            hover_data={"id":True, "type":True, "owner":True, "status":True, "progress_rollup":":.0f"}
        )
        tl.update_yaxes(autorange="reversed")
        tl.update_layout(margin=dict(t=30,l=10,r=10,b=10))
        st.plotly_chart(tl, use_container_width=True)

st.caption("Tip: Add new rows at the bottom of the table. Use IDs like L-1, L-1-1 to keep hierarchy readable.")
