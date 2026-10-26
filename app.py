import re
from itertools import combinations

import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Sus.exe | Narrative Radar", page_icon="🕵️", layout="wide")

st.markdown("""
<style>
.stApp {background: #07111f; color: #e7f0ff;}
[data-testid="stMetric"] {background:#0c1d32; border:1px solid #183b5b; padding:14px; border-radius:12px;}
h1,h2,h3 {color:#f5f8ff;}
.small-note {color:#9eb7d4;}
</style>
""", unsafe_allow_html=True)

st.title("🕵️ Sus.exe — Narrative Radar")
st.caption("PS-03 | Social Media Coordinated Disinformation Detection System")
st.warning("Research prototype: scores indicate patterns for human review, not proof that an account is deceptive.")

SAMPLE = [
    {"post_id":"P001","account":"@bluefox","timestamp":"2026-10-26 09:01","text":"Breaking: city water supply will be shut down tomorrow. Share this with everyone immediately!","platform":"SampleNet"},
    {"post_id":"P002","account":"@dailyflash","timestamp":"2026-10-26 09:03","text":"URGENT city water supply shut down tomorrow. Please share this with everyone now!","platform":"SampleNet"},
    {"post_id":"P003","account":"@truthwire","timestamp":"2026-10-26 09:04","text":"City water supply will stop tomorrow, forward this message to all your contacts.","platform":"SampleNet"},
    {"post_id":"P004","account":"@mango_dev","timestamp":"2026-10-26 09:10","text":"The college robotics club is hosting a workshop this Friday. Registration is open.","platform":"SampleNet"},
    {"post_id":"P005","account":"@campuslife","timestamp":"2026-10-26 09:12","text":"Join our campus robotics workshop on Friday; sign-ups are now open.","platform":"SampleNet"},
    {"post_id":"P006","account":"@filmfan","timestamp":"2026-10-26 10:00","text":"Watched a great mystery thriller last night. Any recommendations?","platform":"SampleNet"},
    {"post_id":"P007","account":"@bluefox","timestamp":"2026-10-26 09:30","text":"Share this now: banks will close tomorrow according to a secret notice. Forward urgently.","platform":"SampleNet"},
    {"post_id":"P008","account":"@dailyflash","timestamp":"2026-10-26 09:31","text":"Banks are closing tomorrow! Secret notice says share urgently with every contact.","platform":"SampleNet"},
    {"post_id":"P009","account":"@localwatch","timestamp":"2026-10-26 11:15","text":"Please check the official city website for updates about water services.","platform":"SampleNet"},
    {"post_id":"P010","account":"@truthwire","timestamp":"2026-10-26 09:32","text":"Urgent secret notice: banks close tomorrow. Forward to everyone immediately.","platform":"SampleNet"},
    {"post_id":"P011","account":"@studybuddy","timestamp":"2026-10-26 12:00","text":"Sharing my notes for the data structures exam. Good luck everyone!","platform":"SampleNet"},
    {"post_id":"P012","account":"@greenplanet","timestamp":"2026-10-26 12:10","text":"Community tree planting event at the park this weekend. Volunteers welcome.","platform":"SampleNet"},
]

def analyze(df, threshold, minutes):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["text"] = df["text"].fillna("").astype(str)
    df["account"] = df["account"].fillna("unknown").astype(str)
    df["post_id"] = df["post_id"].fillna("").astype(str)
    if len(df) < 2:
        df["cluster"] = -1
        df["coordination_score"] = 0
        df["review_flag"] = "Insufficient data"
        return df, nx.Graph(), pd.DataFrame()
    corpus = df["text"].tolist()
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=2500)
    try:
        matrix = vec.fit_transform(corpus)
        sim = cosine_similarity(matrix)
    except ValueError:
        sim = [[0.0]*len(df) for _ in range(len(df))]
    graph = nx.Graph()
    for _, row in df.iterrows():
        graph.add_node(row["account"])
    pairs = []
    scores = [0.0] * len(df)
    for i, j in combinations(range(len(df)), 2):
        similarity = float(sim[i][j])
        ti, tj = df.iloc[i]["timestamp"], df.iloc[j]["timestamp"]
        delta = abs((ti-tj).total_seconds())/60 if pd.notna(ti) and pd.notna(tj) else 999999
        same_window = delta <= minutes
        same_account = df.iloc[i]["account"] == df.iloc[j]["account"]
        if similarity >= threshold:
            pairs.append({"post_a":df.iloc[i]["post_id"],"account_a":df.iloc[i]["account"],
                          "post_b":df.iloc[j]["post_id"],"account_b":df.iloc[j]["account"],
                          "similarity":round(similarity,3),"minutes_apart":round(delta,1),
                          "within_time_window":same_window})
            scores[i] = max(scores[i], similarity * (1.0 if same_window else 0.65))
            scores[j] = max(scores[j], similarity * (1.0 if same_window else 0.65))
            if same_window and not same_account:
                graph.add_edge(df.iloc[i]["account"], df.iloc[j]["account"], weight=similarity)
    df["cluster"] = -1
    if pairs:
        # Connected components over similar-post pairs; cluster IDs are for exploration only.
        pg = nx.Graph()
        for p in pairs:
            pg.add_edge(p["post_a"], p["post_b"])
        comp_map = {}
        cid = 1
        for comp in nx.connected_components(pg):
            if len(comp) > 1:
                for pid in comp:
                    comp_map[pid] = cid
                cid += 1
        df["cluster"] = df["post_id"].map(comp_map).fillna(-1).astype(int)
    df["coordination_score"] = [round(x*100,1) for x in scores]
    df["review_flag"] = df["coordination_score"].apply(lambda x: "Review suggested" if x >= threshold*65 else "No strong signal")
    return df, graph, pd.DataFrame(pairs)

with st.sidebar:
    st.header("Analysis controls")
    threshold = st.slider("Text similarity threshold", 0.20, 0.90, 0.42, 0.01)
    window = st.slider("Coordination time window (minutes)", 5, 180, 45, 5)
    st.caption("Higher similarity thresholds reduce matches. Tune using your dataset.")
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    st.caption("Required columns: post_id, account, timestamp, text. Optional: platform.")

if uploaded:
    try:
        data = pd.read_csv(uploaded)
        missing = [c for c in ["post_id","account","timestamp","text"] if c not in data.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
            st.stop()
        if "platform" not in data.columns:
            data["platform"] = "Uploaded dataset"
        source_label = "Uploaded dataset"
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
else:
    data = pd.DataFrame(SAMPLE)
    source_label = "Built-in synthetic demo data"

result, network, pairs = analyze(data, threshold, window)
total = len(result)
flagged = int((result["review_flag"] == "Review suggested").sum())
clusters = int(result.loc[result["cluster"] >= 0, "cluster"].nunique())
edges = network.number_of_edges()

m1,m2,m3,m4 = st.columns(4)
m1.metric("Posts analyzed", total)
m2.metric("Posts needing review", flagged)
m3.metric("Narrative clusters", clusters)
m4.metric("Coordination links", edges)
st.caption(f"Data source: {source_label} · Similarity method: TF-IDF cosine similarity")

left,right = st.columns([1.25,1])
with left:
    st.subheader("📌 Posts for review")
    show_flagged = st.checkbox("Show only posts with review signals", value=False)
    view = result[result["review_flag"] == "Review suggested"] if show_flagged else result
    st.dataframe(view[["post_id","account","timestamp","text","cluster","coordination_score","review_flag"]], use_container_width=True, hide_index=True)
    st.download_button("Download analysis CSV", result.to_csv(index=False).encode("utf-8"), "sus_exe_analysis.csv", "text/csv")
with right:
    st.subheader("🧩 Narrative clusters")
    cluster_counts = result[result["cluster"] >= 0]["cluster"].value_counts().sort_index()
    if len(cluster_counts):
        st.bar_chart(cluster_counts.rename("Posts"))
    else:
        st.info("No clusters at this threshold. Try lowering the similarity threshold.")
    st.subheader("🕸️ Account coordination network")
    if network.number_of_edges():
        pos = nx.spring_layout(network, seed=7)
        edge_x, edge_y = [], []
        for u,v in network.edges():
            edge_x += [pos[u][0],pos[v][0],None]
            edge_y += [pos[u][1],pos[v][1],None]
        edge_trace = px.line(x=edge_x, y=edge_y).data[0]
        edge_trace.update(line=dict(width=1.5, color="#3b82f6"), hoverinfo="none")
        nodes = list(network.nodes())
        node_trace = px.scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], text=nodes).data[0]
        node_trace.update(mode="markers+text", textposition="top center", marker=dict(size=16, color="#fbbf24"), hovertemplate="%{text}<extra></extra>")
        import plotly.graph_objects as go
        fig = go.Figure(data=[edge_trace,node_trace])
        fig.update_layout(showlegend=False, margin=dict(l=5,r=5,t=10,b=5), height=360,
                          paper_bgcolor="#07111f", plot_bgcolor="#07111f",
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No cross-account links at this threshold and time window.")

st.subheader("🔎 Similar-post evidence")
if len(pairs):
    st.dataframe(pairs.sort_values(["within_time_window","similarity"], ascending=False), use_container_width=True, hide_index=True)
else:
    st.info("No pairs passed the current threshold.")
with st.expander("Methodology and limitations"):
    st.markdown("""
    - Text similarity uses TF-IDF word and bigram features with cosine similarity.
    - A potential coordination link is drawn when two different accounts post sufficiently similar text within the selected time window.
    - Narrative clusters are connected components of similar posts.
    - The coordination score is a heuristic for triage, not a calibrated probability.
    - Similarity does not establish falsehood, intent, bot activity, or coordination by itself.
    - The included posts are synthetic examples for demonstration; replace them with a licensed/public dataset before making real-world claims.
    """)
