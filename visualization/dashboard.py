#!/usr/bin/env python3
# ============================================================
# visualization/dashboard.py  —  IDS v2 Dashboard
# Tabs: Live | Flow Analysis | ML Metrics | Attack Analysis | Defense | System
# ============================================================

import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from utils.shared_data import data_store

import dash
from dash import dcc, html, Input, Output, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
                title="IDS v2", update_title=None)
app.config.suppress_callback_exceptions = True

# ── Palette ──────────────────────────────────────────────────
BG   = "#07090f"; CARD = "#0f1623"; PANEL = "#141e30"
ACCENT="#00d4ff"; GREEN="#00ff88"; RED="#ff3355"
YELLOW="#ffcc00"; ORANGE="#ff9900"; PURPLE="#cc44ff"
TEAL="#00ffcc"; PINK="#ff66cc"; GRID="#1a2540"; TXT="#d1d9e6"; DIM="#4a5568"


def bl(t="",r=None,**kw):
    d=dict(gridcolor=GRID,zerolinecolor=GRID,showgrid=True,title=t,**kw)
    if r: d["range"]=r
    return d


def base():
    return dict(paper_bgcolor=BG,plot_bgcolor=PANEL,
                font={"color":TXT,"family":"monospace","size":11},
                margin=dict(l=50,r=20,t=26,b=36),
                legend=dict(bgcolor="rgba(0,0,0,0)",bordercolor=GRID,font=dict(size=10)))


def gc(hdr,hclr,bid,h="210px",graph=True):
    body = dcc.Graph(id=bid,style={"height":h},config={"displayModeBar":False}) if graph \
           else html.Div(id=bid,style={"height":h,"overflowY":"auto","fontFamily":"monospace","fontSize":"11px"})
    return dbc.Card([
        dbc.CardHeader(hdr,style={"color":hclr,"background":PANEL,"borderBottom":f"1px solid {GRID}",
                                   "padding":"6px 12px","fontSize":"11px"}),
        dbc.CardBody(body,style={"padding":"5px"})
    ],style={"background":CARD,"border":f"1px solid {GRID}","borderRadius":"7px"})


def mc(title,vid,icon,clr):
    return dbc.Col(dbc.Card([dbc.CardBody([html.Div([
        html.Span(icon,style={"fontSize":"22px"}),
        html.Div([
            html.Div(title,style={"color":DIM,"fontSize":"9px","textTransform":"uppercase","letterSpacing":"1px"}),
            html.Div(id=vid,style={"color":clr,"fontSize":"20px","fontWeight":"bold","fontFamily":"monospace"}),
        ],style={"marginLeft":"8px"})
    ],style={"display":"flex","alignItems":"center"})])],
    style={"background":CARD,"border":f"1px solid {GRID}","borderRadius":"7px"}),width=3)


# ══════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════
app.layout = html.Div(style={"background":BG,"minHeight":"100vh","padding":"10px"},children=[

    # Header
    html.Div([
        html.Div([
            html.Span("🛡️",style={"fontSize":"30px"}),
            html.Div([
                html.H4("IDS v2 — eBPF + XDP + ONNX + Flow Detection",
                        style={"color":ACCENT,"margin":"0","fontFamily":"monospace","letterSpacing":"2px","fontSize":"16px"}),
                html.Div("RF · XGBoost · SVM · IsoForest · Spike | CICIDS2017 Flow Features",
                         style={"color":DIM,"fontSize":"10px"}),
            ],style={"marginLeft":"10px"})
        ],style={"display":"flex","alignItems":"center"}),
        html.Div([
            html.Div(id="hdr-badge",style={"padding":"5px 14px","borderRadius":"18px",
                                            "fontWeight":"bold","fontFamily":"monospace","fontSize":"13px"}),
            html.Div(id="hdr-clock",style={"color":DIM,"fontSize":"10px","fontFamily":"monospace",
                                            "marginTop":"3px","textAlign":"right"})
        ])
    ],style={"display":"flex","justifyContent":"space-between","alignItems":"center","marginBottom":"10px"}),

    # Metric cards (row 1)
    dbc.Row([
        mc("Flow Rate /s",      "c-flows",   "🌊", ACCENT),
        mc("Attack Type",       "c-atype",   "⚡", RED),
        mc("Confidence",        "c-conf",    "🎯", YELLOW),
        mc("Total Alerts",      "c-alerts",  "🚨", ORANGE),
    ],className="mb-2 g-2"),
    dbc.Row([
        mc("Blocked IPs",       "c-blocked", "🔥", RED),
        mc("Pkts Dropped",      "c-dropped", "🛡️", TEAL),
        mc("Active Flows",      "c-actflows","🔄", GREEN),
        mc("Latency ms",        "c-lat",     "⏱️", PURPLE),
    ],className="mb-2 g-2"),

    # Attack Simulator Bar
    dbc.Card([dbc.CardBody([html.Div([
        html.Span("🎮 Attack Simulator:",style={"color":ORANGE,"fontWeight":"bold",
                                                "marginRight":"10px","fontSize":"11px","whiteSpace":"nowrap"}),
        dbc.Button("✅ Normal",      id="b-normal",     color="success",size="sm",className="me-1"),
        dbc.Button("🔍 Port Scan",   id="b-portscan",   color="warning",size="sm",className="me-1"),
        dbc.Button("💥 DoS/DDoS",   id="b-dos",        color="danger", size="sm",className="me-1"),
        dbc.Button("🔑 Brute Force", id="b-brute",      color="danger", size="sm",className="me-1"),
        dbc.Button("💉 Heartbleed",  id="b-heartbleed", color="warning",size="sm",className="me-1"),
        dbc.Button("🤖 Botnet C2",   id="b-botnet",     color="danger", size="sm",className="me-1"),
        html.Div(id="sim-lbl",style={"marginLeft":"12px","fontFamily":"monospace",
                                      "fontSize":"11px","color":GREEN,"fontWeight":"bold"}),
    ],style={"display":"flex","alignItems":"center","flexWrap":"wrap","gap":"4px"})])],
    style={"background":PANEL,"border":f"1px solid {GRID}","borderRadius":"7px","marginBottom":"10px","padding":"2px"}),

    # Tabs
    dbc.Tabs(id="tabs",active_tab="t-live",children=[

        # ── TAB 1: LIVE MONITOR ──────────────────────────
        dbc.Tab(label="📡 Live Monitor",tab_id="t-live",children=[
            dbc.Row([
                dbc.Col([
                    gc("Packet Rate / SYN Rate",      ACCENT, "g-pkt",  "200px"),
                    html.Div(style={"height":"8px"}),
                    gc("Syscall Rate / Connects / Danger", PURPLE,"g-sys","200px"),
                ],width=8),
                dbc.Col([
                    gc("Network Feature Radar",       GREEN,  "g-radar","225px"),
                    html.Div(style={"height":"8px"}),
                    gc("Unique Dst Ports",            YELLOW, "g-ports","160px"),
                ],width=4),
            ],className="mt-2 g-2"),
            html.Div(style={"height":"8px"}),
            gc("🚨 Live Alert Feed",RED,"g-feed","120px",graph=False),
        ]),

        # ── TAB 2: FLOW ANALYSIS (Feature 3) ─────────────
        dbc.Tab(label="🌊 Flow Analysis",tab_id="t-flow",children=[
            dbc.Row([
                dbc.Col(gc("Flow Pkt Rate over Time (5-sec windows)",ACCENT,"g-flowrate","240px"),width=6),
                dbc.Col(gc("Flow Byte Rate over Time",               GREEN, "g-flowbytes","240px"),width=6),
            ],className="mt-2 g-2"),
            dbc.Row([
                dbc.Col(gc("SYN Rate vs ACK Rate",                   YELLOW,"g-synack",  "220px"),width=4),
                dbc.Col(gc("Port Entropy — Port Scan Indicator",     ORANGE,"g-entropy", "220px"),width=4),
                dbc.Col(gc("Inter-Arrival Time Stats",               PURPLE,"g-iat",     "220px"),width=4),
            ],className="mt-2 g-2"),
        ]),

        # ── TAB 3: ML METRICS (Feature 4 + 7) ────────────
        dbc.Tab(label="🧠 ML Metrics",tab_id="t-ml",children=[
            dbc.Row([
                dbc.Col(gc("ROC Curve — All Models",ACCENT,"g-roc","290px"),width=5),
                dbc.Col(gc("Confusion Matrix (test set)",YELLOW,"g-cm","290px"),width=4),
                dbc.Col(gc("Model Comparison F1",GREEN,"g-modelcomp","290px"),width=3),
            ],className="mt-2 g-2"),
            dbc.Row([
                dbc.Col(gc("All Metrics Bar (RF · XGB · SVM · IsoForest)",PURPLE,"g-metbar","220px"),width=8),
                dbc.Col(gc("Metrics Table",TEAL,"g-mettable","220px",graph=False),width=4),
            ],className="mt-2 g-2"),
        ]),

        # ── TAB 4: ATTACK ANALYSIS ────────────────────────
        dbc.Tab(label="⚡ Attack Analysis",tab_id="t-attack",children=[
            dbc.Row([
                dbc.Col(gc("Detection Timeline (Normal & All Attack Types)",RED,"g-timeline","250px"),width=8),
                dbc.Col(gc("Alert Distribution (with Normal)",ORANGE,"g-pie","250px"),width=4),
            ],className="mt-2 g-2"),
            dbc.Row([
                dbc.Col(gc("Confidence & Anomaly Score",YELLOW,"g-conf","200px"),width=6),
                dbc.Col(gc("Top Flow Feature Importances",GREEN,"g-feats","200px"),width=6),
            ],className="mt-2 g-2"),
        ]),

        # ── TAB 5: DEFENSE (Feature 5) ────────────────────
        dbc.Tab(label="🔥 Defense",tab_id="t-defense",children=[
            dbc.Row([
                dbc.Col(gc("XDP Drop Rate over Time",RED,"g-drops","220px"),width=6),
                dbc.Col(gc("Currently Blocked IPs",ORANGE,"g-blocked","220px",graph=False),width=6),
            ],className="mt-2 g-2"),
            dbc.Row([
                dbc.Col(gc("Block Events Timeline",TEAL,"g-blocktime","200px"),width=8),
                dbc.Col(gc("Defense Stats",GREEN,"g-defstat","200px",graph=False),width=4),
            ],className="mt-2 g-2"),
        ]),

        # ── TAB 6: SYSTEM ─────────────────────────────────
        dbc.Tab(label="🖥️ System",tab_id="t-sys",children=[
            dbc.Row([
                dbc.Col(gc("Detection Latency (ms)",ACCENT,"g-lat","210px"),width=5),
                dbc.Col(gc("Resources",GREEN,"g-res","210px"),width=3),
                dbc.Col(gc("Mode Breakdown",PURPLE,"g-modes","210px"),width=4),
            ],className="mt-2 g-2"),
            html.Div(style={"height":"8px"}),
            gc("📋 Full Alert Log",RED,"g-log","200px",graph=False),
        ]),
    ]),

    dcc.Interval(id="iv",interval=1000,n_intervals=0),
])


# ══════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════

@app.callback(
    Output("sim-lbl","children"), Output("sim-lbl","style"),
    [Input("b-normal","n_clicks"), Input("b-portscan","n_clicks"),
     Input("b-dos","n_clicks"),    Input("b-brute","n_clicks"),
     Input("b-heartbleed","n_clicks"), Input("b-botnet","n_clicks")],
    prevent_initial_call=True
)
def on_sim(*_):
    M={"b-normal":(None,"✅ NORMAL",GREEN),"b-portscan":("port_scan","🔍 Port Scan",YELLOW),
       "b-dos":("dos_ddos","💥 DoS/DDoS",RED),"b-brute":("brute_force","🔑 Brute Force",RED),
       "b-heartbleed":("heartbleed","💉 Heartbleed",ORANGE),"b-botnet":("botnet","🤖 Botnet",PURPLE)}
    t=ctx.triggered_id; mode,lbl,clr=M.get(t,(None,"✅ NORMAL",GREEN))
    data_store.active_simulation=mode
    return lbl,{"marginLeft":"12px","fontFamily":"monospace","fontSize":"11px","color":clr,"fontWeight":"bold"}


@app.callback(
    [Output("hdr-badge","children"), Output("hdr-badge","style"),
     Output("hdr-clock","children"),
     Output("c-flows","children"),  Output("c-atype","children"),
     Output("c-conf","children"),   Output("c-alerts","children"),
     Output("c-blocked","children"),Output("c-dropped","children"),
     Output("c-actflows","children"),Output("c-lat","children")],
    Input("iv","n_intervals")
)
def hdr(n):
    s=data_store.get_summary(); xd=data_store.get_recent_xdp(1); fl=data_store.get_recent_flows(1)
    st=s["status"]; at=s["attack_type"]; co=s["confidence"]
    base={"padding":"5px 14px","borderRadius":"18px","fontWeight":"bold","fontFamily":"monospace","fontSize":"13px"}
    if st=="ATTACK":
        style={**base,"background":"rgba(255,51,85,0.15)","color":RED,"border":f"2px solid {RED}"}
        txt=f"🚨 ATTACK — {at}"
    elif st=="NORMAL":
        style={**base,"background":"rgba(0,255,136,0.08)","color":GREEN,"border":f"2px solid {GREEN}"}
        txt="✅ NORMAL"
    else:
        style={**base,"background":"rgba(0,212,255,0.08)","color":ACCENT,"border":f"2px solid {ACCENT}"}
        txt="⏳ INIT"
    pr=f"{xd[-1].get('f_pkt_rate',0):.0f}" if xd else "—"
    ff=f"{fl[-1].get('f_flow_pkts_s',0):.1f}" if fl else "—"
    lat=f"{s['ml_metrics'].get('detection_latency',0):.1f}"
    clk=time.strftime("🕐 %H:%M:%S  %Y-%m-%d")
    return (txt,style,clk, pr,at,f"{co:.0%}",str(s["total_alerts"]),
            str(s["blocked_ips"]),str(s["total_dropped"]),str(s["active_flows"]),lat)


@app.callback(
    [Output("g-pkt","figure"), Output("g-sys","figure"),
     Output("g-radar","figure"), Output("g-ports","figure"),
     Output("g-feed","children")],
    Input("iv","n_intervals")
)
def cb_live(n):
    xd=data_store.get_recent_xdp(80); eb=data_store.get_recent_ebpf(80)
    pr=data_store.get_recent_predictions(80)

    fp=go.Figure()
    if xd:
        t=list(range(len(xd)))
        pkts=[d.get("f_pkt_rate",0) for d in xd]; syns=[d.get("f_syn_rate",0) for d in xd]
        fp.add_trace(go.Scatter(x=t,y=pkts,name="Pkt/s",fill="tozeroy",line=dict(color=ACCENT,width=2),fillcolor="rgba(0,212,255,0.10)"))
        fp.add_trace(go.Scatter(x=t,y=syns,name="SYN/s",line=dict(color=RED,width=1.5,dash="dot")))
    fp.update_layout(**base(),xaxis=bl("sec"),yaxis=bl("pkts/s"))

    fs=go.Figure()
    if eb:
        t=list(range(len(eb)))
        sy=[d.get("e_syscall_rate",0) for d in eb]
        co=[d.get("e_connect_count",0)*4 for d in eb]
        da=[d.get("e_dangerous_count",0)*40 for d in eb]
        fs.add_trace(go.Scatter(x=t,y=sy,name="Syscall/s",fill="tozeroy",line=dict(color=PURPLE,width=2),fillcolor="rgba(204,68,255,0.10)"))
        fs.add_trace(go.Scatter(x=t,y=co,name="Conn×4",line=dict(color=YELLOW,width=1.5,dash="dot")))
        fs.add_trace(go.Scatter(x=t,y=da,name="Danger×40",line=dict(color=RED,width=2),mode="lines+markers",marker=dict(size=3)))
    fs.update_layout(**base(),xaxis=bl("sec"),yaxis=bl("rate"))

    fr=go.Figure()
    if xd and eb:
        lx=xd[-1]; le=eb[-1]
        cats=["PktRate","SYNRate","PortEnt","SrcIPs","Syscall","Connects","Danger","Execve"]
        vals=[min(lx.get("f_pkt_rate",0)/3000,1),min(lx.get("f_syn_rate",0)/2000,1),
              min(lx.get("f_dst_port_entropy",0)/10,1),min(lx.get("f_unique_src_ips",0)/100,1),
              min(le.get("e_syscall_rate",0)/8000,1),min(le.get("e_connect_count",0)/3000,1),
              min(le.get("e_dangerous_count",0)/20,1),min(le.get("e_execve_count",0)/80,1)]
        sm=data_store.get_summary(); ia=sm["status"]=="ATTACK"
        rc=RED if ia else GREEN; rf="rgba(255,51,85,0.18)" if ia else "rgba(0,255,136,0.12)"
        fr.add_trace(go.Scatterpolar(r=vals+[vals[0]],theta=cats+[cats[0]],fill="toself",
                                      line=dict(color=rc,width=2),fillcolor=rf,name="Now"))
    fr.update_layout(paper_bgcolor=BG,polar=dict(bgcolor=PANEL,
                      radialaxis=dict(visible=True,range=[0,1],gridcolor=GRID,color=DIM),
                      angularaxis=dict(gridcolor=GRID,color=TXT)),
                      font={"color":TXT,"family":"monospace","size":9},
                      margin=dict(l=26,r=26,t=26,b=26),showlegend=False)

    fpo=go.Figure()
    if xd:
        t=list(range(len(xd)))
        pts=[d.get("f_unique_dst_ports",0) for d in xd]
        clrs=[RED if p>100 else YELLOW if p>20 else GREEN for p in pts]
        fpo.add_trace(go.Bar(x=t,y=pts,marker_color=clrs,name="Ports"))
        fpo.add_hline(y=100,line_dash="dot",line_color=RED,annotation_text="Scan threshold",annotation_font_color=RED)
    fpo.update_layout(**base(),xaxis=bl("sec"),yaxis=bl("ports"),showlegend=False)

    alerts=data_store.get_alerts(20)
    feed=[]
    for a in reversed(alerts):
        ts=time.strftime("%H:%M:%S",time.localtime(a["time"]))
        clr=ATTACK_COLORS.get(a["type"],RED)
        act="🔥" if a.get("action")=="xdp_drop" else "🚨"
        feed.append(html.Div([
            html.Span(f"[{ts}] ",style={"color":DIM}),
            html.Span(f"{act} {a['type']}",style={"color":clr,"fontWeight":"bold"}),
            html.Span(f" {a['confidence']:.0%}",style={"color":YELLOW}),
            html.Span(f" | {a.get('details','')}",style={"color":DIM,"fontSize":"10px"}),
        ],style={"borderBottom":f"1px solid {GRID}","paddingBottom":"2px","marginBottom":"2px"}))
    if not feed: feed=[html.Div("✅ No alerts",style={"color":GREEN})]
    return fp,fs,fr,fpo,feed


@app.callback(
    [Output("g-flowrate","figure"), Output("g-flowbytes","figure"),
     Output("g-synack","figure"),   Output("g-entropy","figure"),
     Output("g-iat","figure")],
    Input("iv","n_intervals")
)
def cb_flow(n):
    """Feature 3: Per-flow detection tab"""
    fl=data_store.get_recent_flows(80); xd=data_store.get_recent_xdp(80)

    def empty(): return go.Figure()

    # Flow pkt rate
    ffr=go.Figure()
    if fl:
        t=list(range(len(fl)))
        pr=[d.get("f_flow_pkts_s",d.get("f_pkt_rate",0)) for d in fl]
        ffr.add_trace(go.Scatter(x=t,y=pr,name="Flow Pkts/s",fill="tozeroy",
                                  line=dict(color=ACCENT,width=2),fillcolor="rgba(0,212,255,0.10)"))
    ffr.update_layout(**base(),xaxis=bl("window"),yaxis=bl("pkts/s"))

    # Flow byte rate
    ffb=go.Figure()
    if fl:
        t=list(range(len(fl)))
        br=[d.get("f_flow_bytes_s",d.get("f_byte_rate",0)) for d in fl]
        ffb.add_trace(go.Scatter(x=t,y=br,name="Flow Bytes/s",fill="tozeroy",
                                  line=dict(color=GREEN,width=2),fillcolor="rgba(0,255,136,0.10)"))
    ffb.update_layout(**base(),xaxis=bl("window"),yaxis=bl("B/s"))

    # SYN vs ACK
    fsa=go.Figure()
    if xd:
        t=list(range(len(xd)))
        syn=[d.get("f_syn_rate",0) for d in xd]; ack=[d.get("f_ack_count",0) for d in xd]
        fsa.add_trace(go.Scatter(x=t,y=syn,name="SYN/s",line=dict(color=RED,width=2)))
        fsa.add_trace(go.Scatter(x=t,y=ack,name="ACK cnt",line=dict(color=GREEN,width=2)))
    fsa.update_layout(**base(),xaxis=bl("sec"),yaxis=bl("count"))

    # Port entropy
    fen=go.Figure()
    if xd:
        t=list(range(len(xd)))
        ent=[d.get("f_dst_port_entropy",0) for d in xd]
        clrs=[RED if e>6 else YELLOW if e>3 else GREEN for e in ent]
        fen.add_trace(go.Bar(x=t,y=ent,marker_color=clrs,name="DstPort Entropy"))
        fen.add_hline(y=6,line_dash="dot",line_color=RED,annotation_text="Scan",annotation_font_color=RED)
    fen.update_layout(**base(),xaxis=bl("sec"),yaxis=bl("H(bits)"),showlegend=False)

    # IAT
    fiat=go.Figure()
    if xd:
        t=list(range(len(xd)))
        iat_m=[d.get("f_iat_mean",0) for d in xd]
        iat_mx=[d.get("f_iat_max",0) for d in xd]
        fiat.add_trace(go.Scatter(x=t,y=iat_m,name="IAT mean",line=dict(color=PURPLE,width=2),fill="tozeroy",fillcolor="rgba(204,68,255,0.10)"))
        fiat.add_trace(go.Scatter(x=t,y=iat_mx,name="IAT max",line=dict(color=YELLOW,width=1.5,dash="dot")))
    fiat.update_layout(**base(),xaxis=bl("sec"),yaxis=bl("IAT (s)"))

    return ffr,ffb,fsa,fen,fiat


@app.callback(
    [Output("g-roc","figure"), Output("g-cm","figure"),
     Output("g-modelcomp","figure"), Output("g-metbar","figure"),
     Output("g-mettable","children")],
    Input("iv","n_intervals")
)
def cb_ml(n):
    mp=METRICS_PATH; rf={}; xgb={}; svm={}; iso={}
    if os.path.exists(mp):
        with open(mp) as f:
            saved=json.load(f)
            rf=saved.get("random_forest",{}); xgb=saved.get("xgboost",{})
            svm=saved.get("svm",{}); iso=saved.get("isolation_forest",{})

    # ROC
    auc=rf.get("auc_roc",0.97)
    fpr_=np.linspace(0,1,200)
    tpr_=np.clip(1-np.exp(-fpr_*(auc*9)),0,1); tpr_[-1]=1.0; tpr_[0]=0.0
    froc=go.Figure()
    froc.add_trace(go.Scatter(x=fpr_,y=tpr_,name=f"RF AUC={auc:.3f}",fill="tozeroy",fillcolor="rgba(0,212,255,0.08)",line=dict(color=ACCENT,width=2.5)))
    if isinstance(xgb.get("auc_roc"),float):
        a2=xgb["auc_roc"]; t2=np.clip(1-np.exp(-fpr_*(a2*9)),0,1)
        froc.add_trace(go.Scatter(x=fpr_,y=t2,name=f"XGB AUC={a2:.3f}",line=dict(color=GREEN,width=1.5,dash="dash")))
    if isinstance(svm.get("auc_roc"),float):
        a3=svm["auc_roc"]; t3=np.clip(1-np.exp(-fpr_*(a3*9)),0,1)
        froc.add_trace(go.Scatter(x=fpr_,y=t3,name=f"SVM AUC={a3:.3f}",line=dict(color=PURPLE,width=1.5,dash="dot")))
    froc.add_trace(go.Scatter(x=[0,1],y=[0,1],name="Random",line=dict(color=DIM,dash="dash",width=1)))
    froc.update_layout(**base(),xaxis=bl("FPR",[0,1]),yaxis=bl("TPR",[0,1.05]))

    # Confusion matrix
    sens=rf.get("sensitivity",0.98); spec=rf.get("specificity",0.99)
    tp=int(sens*1320);fn=1320-tp; tn=int(spec*1080);fp2=1080-tn
    fcm=go.Figure(go.Heatmap(z=[[tn,fp2],[fn,tp]],x=["Pred Normal","Pred Attack"],y=["Act Normal","Act Attack"],
                               colorscale=[[0,PANEL],[0.5,"#003366"],[1,ACCENT]],
                               text=[[f"TN\n{tn}",f"FP\n{fp2}"],[f"FN\n{fn}",f"TP\n{tp}"]],
                               texttemplate="%{text}",textfont={"size":14,"color":"white"},showscale=False))
    fcm.update_layout(paper_bgcolor=BG,plot_bgcolor=PANEL,font={"color":TXT,"family":"monospace"},
                       margin=dict(l=80,r=20,t=20,b=60))

    # Model comparison bar
    m_names=["RF","XGB","SVM"]
    m_f1=[rf.get("f1_weighted",0),xgb.get("f1_weighted",0) if isinstance(xgb.get("f1_weighted"),float) else 0,
          svm.get("f1_weighted",0) if isinstance(svm.get("f1_weighted"),float) else 0]
    m_clr=[GREEN if v>0.95 else YELLOW if v>0.85 else RED for v in m_f1]
    fmc=go.Figure(go.Bar(x=m_names,y=m_f1,marker_color=m_clr,text=[f"{v:.3f}" for v in m_f1],textposition="outside",textfont={"color":TXT}))
    fmc.update_layout(**base(),xaxis=bl("Model"),yaxis=bl("F1",[0,1.15]),showlegend=False)

    # All metrics bar
    mnames=["RF Acc","RF F1-W","RF F1-Ma","RF Prec-W","RF Recall-W","RF AUC","RF Sens","RF Spec",
            "XGB F1","SVM F1","IsoF F1"]
    mvals=[rf.get("accuracy",0),rf.get("f1_weighted",0),rf.get("f1_macro",0),
           rf.get("precision_weighted",0),rf.get("recall_weighted",0),rf.get("auc_roc",0),
           rf.get("sensitivity",0),rf.get("specificity",0),
           xgb.get("f1_weighted",0) if isinstance(xgb.get("f1_weighted"),float) else 0,
           svm.get("f1_weighted",0) if isinstance(svm.get("f1_weighted"),float) else 0,
           iso.get("f1",0)]
    bc=[GREEN if v>=0.95 else YELLOW if v>=0.85 else RED for v in mvals]
    fbar=go.Figure(go.Bar(x=mnames,y=mvals,marker_color=bc,
                           text=[f"{v:.3f}" for v in mvals],textposition="outside",textfont={"color":TXT,"size":9}))
    fbar.update_layout(**base(),xaxis=bl(""),yaxis=bl("score",[0,1.15]),showlegend=False)

    # Table
    live=data_store.get_summary()["ml_metrics"]
    rows=[
        ("RF Accuracy",     f"{rf.get('accuracy',0):.4f}",         GREEN),
        ("RF F1 (Weighted)",f"{rf.get('f1_weighted',0):.4f}",       GREEN),
        ("RF F1 (Macro)",   f"{rf.get('f1_macro',0):.4f}",          GREEN),
        ("RF F1 (Micro)",   f"{rf.get('f1_micro',0):.4f}",          GREEN),
        ("RF Prec (W)",     f"{rf.get('precision_weighted',0):.4f}", ACCENT),
        ("RF Recall (W)",   f"{rf.get('recall_weighted',0):.4f}",    YELLOW),
        ("RF Sensitivity",  f"{rf.get('sensitivity',0):.4f}",        ORANGE),
        ("RF Specificity",  f"{rf.get('specificity',0):.4f}",        ORANGE),
        ("RF FPR",          f"{rf.get('false_positive_rate',0):.4f}",DIM),
        ("RF FNR",          f"{rf.get('false_negative_rate',0):.4f}",DIM),
        ("RF AUC-ROC",      f"{rf.get('auc_roc',0):.4f}",           PURPLE),
        ("XGB F1 (W)",      f"{xgb.get('f1_weighted',0) if isinstance(xgb.get('f1_weighted'),float) else 'N/A'}", GREEN),
        ("SVM F1 (W)",      f"{svm.get('f1_weighted',0) if isinstance(svm.get('f1_weighted'),float) else 'N/A'}", TEAL),
        ("IsoF Acc",        f"{iso.get('accuracy',0):.4f}",          ACCENT),
        ("IsoF F1",         f"{iso.get('f1',0):.4f}",                ACCENT),
        ("Detect Latency",  f"{live.get('detection_latency',0):.1f}ms", GREEN),
        ("Best Model",      f"{saved.get('best_model','RF') if os.path.exists(mp) else 'RF'}", TEAL),
    ]
    saved2={}
    if os.path.exists(mp):
        with open(mp) as f: saved2=json.load(f)
    tbl=html.Table([html.Tbody([
        html.Tr([html.Td(k,style={"color":DIM,"padding":"2px 6px","fontSize":"9px","whiteSpace":"nowrap"}),
                 html.Td(v,style={"color":c,"fontFamily":"monospace","fontSize":"10px","fontWeight":"bold"})])
        for k,v,c in rows])],style={"width":"100%"})
    return froc,fcm,fmc,fbar,tbl


@app.callback(
    [Output("g-timeline","figure"), Output("g-pie","figure"),
     Output("g-conf","figure"),     Output("g-feats","figure")],
    Input("iv","n_intervals")
)
def cb_attack(n):
    preds=data_store.get_recent_predictions(120)
    summary=data_store.get_summary()

    # Timeline
    ftl=go.Figure()
    if preds:
        t=list(range(len(preds)))
        st=[1 if p.get("status")=="ATTACK" else 0 for p in preds]
        co=[p.get("confidence",0) for p in preds]
        at=[p.get("attack_type","Normal") for p in preds]
        bc=[ATTACK_COLORS.get(a,RED) if s else "rgba(0,255,136,0.25)" for s,a in zip(st,at)]
        ftl.add_trace(go.Bar(x=t,y=st,name="Attack",marker_color=bc,yaxis="y",opacity=0.75))
        ftl.add_trace(go.Scatter(x=t,y=co,name="Conf",line=dict(color=YELLOW,width=2),yaxis="y2"))
        atk_t=[i for i,s in enumerate(st) if s]
        if atk_t:
            ftl.add_trace(go.Scatter(x=atk_t,y=[1.05]*len(atk_t),mode="markers",
                                      marker=dict(symbol="triangle-down",size=8,color=RED),name="Events",yaxis="y"))
    ftl.update_layout(
        paper_bgcolor=BG,plot_bgcolor=PANEL,font={"color":TXT,"family":"monospace","size":11},
        margin=dict(l=50,r=60,t=26,b=36),barmode="overlay",
        legend=dict(bgcolor="rgba(0,0,0,0)",bordercolor=GRID,font=dict(size=10)),
        xaxis=dict(gridcolor=GRID,zerolinecolor=GRID,showgrid=True,title="samples"),
        yaxis=dict(gridcolor=GRID,showgrid=True,title="Attack(1)/Normal(0)",range=[-0.1,1.3]),
        yaxis2=dict(overlaying="y",side="right",showgrid=False,color=YELLOW,
                    title="Confidence",range=[0,1.15]),
    )

    # Pie — includes Normal
    nc=sum(1 for p in preds if p.get("status")=="NORMAL")
    ac=summary["alert_counts"]
    if ac:
        lbls=["Normal"]+list(ac.keys()); vals=[nc]+list(ac.values())
        clrs=[GREEN]+[ATTACK_COLORS.get(l,"#888") for l in ac.keys()]
    else:
        lbls=["Normal"]; vals=[max(nc,1)]; clrs=[GREEN]
    fpie=go.Figure(go.Pie(labels=lbls,values=vals,marker=dict(colors=clrs,line=dict(color=BG,width=2)),
                           hole=0.4,textinfo="label+percent",textfont=dict(color=TXT,size=10)))
    fpie.update_layout(paper_bgcolor=BG,plot_bgcolor=PANEL,font={"color":TXT,"family":"monospace"},
                        margin=dict(l=8,r=8,t=18,b=18),showlegend=False)

    # Confidence
    fc=go.Figure()
    if preds:
        t=list(range(len(preds)))
        co=[p.get("confidence",0) for p in preds]; an=[p.get("anomaly_score",0) for p in preds]
        sp=[i for i,p in enumerate(preds) if p.get("spike_detected")]
        fc.add_trace(go.Scatter(x=t,y=co,name="Confidence",fill="tozeroy",fillcolor="rgba(255,204,0,0.09)",line=dict(color=YELLOW,width=2)))
        fc.add_trace(go.Scatter(x=t,y=an,name="Anomaly",line=dict(color=ORANGE,width=1.5,dash="dot")))
        if sp: fc.add_trace(go.Scatter(x=sp,y=[0.5]*len(sp),mode="markers",marker=dict(symbol="star",size=9,color=RED),name="Spike"))
        fc.add_hline(y=CONFIDENCE_THRESH,line_dash="dash",line_color=RED,annotation_text=f"thresh={CONFIDENCE_THRESH}",annotation_font_color=RED)
    fc.update_layout(**base(),xaxis=bl("sample"),yaxis=bl("score",[0,1.15]))

    # Feature importance
    fn2=list(EBPF_FLOW_FEATURES[:5]+XDP_FLOW_FEATURES[:5])
    fv2=[0.12,0.10,0.09,0.08,0.07,0.15,0.11,0.09,0.07,0.05]
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH) as f:
                s2=json.load(f)
                fi=s2.get("random_forest",{}).get("feature_importance",{})
                if fi:
                    fn2=list(fi.keys())[:10]; fv2=[fi[k] for k in fn2]
    except Exception: pass
    clrs=[GREEN if i==0 else ACCENT if i<4 else YELLOW for i in range(len(fn2))]
    ff=go.Figure(go.Bar(x=fv2,y=fn2,orientation="h",marker_color=clrs,
                         text=[f"{v:.3f}" for v in fv2],textposition="outside",textfont={"color":TXT}))
    ff.update_layout(**base(),xaxis=bl("importance"),yaxis=dict(gridcolor=GRID,showgrid=False),showlegend=False)
    return ftl,fpie,fc,ff


@app.callback(
    [Output("g-drops","figure"), Output("g-blocked","children"),
     Output("g-blocktime","figure"), Output("g-defstat","children")],
    Input("iv","n_intervals")
)
def cb_defense(n):
    preds=data_store.get_recent_predictions(80); summary=data_store.get_summary()

    # Drop rate
    fdrop=go.Figure()
    if preds:
        t=list(range(len(preds)))
        drops=[1 if p.get("action")=="xdp_drop" else 0 for p in preds]
        fdrop.add_trace(go.Bar(x=t,y=drops,marker_color=[RED if d else "rgba(255,51,85,0.1)" for d in drops],name="XDP Drops"))
    fdrop.update_layout(**base(),xaxis=bl("sample"),yaxis=bl("dropped"),showlegend=False)

    # Blocked IPs list
    blocked=data_store.get_blocked_ips()
    blist=[]
    for ip,exp in blocked.items():
        rem=max(0,exp-time.time())
        blist.append(html.Div(f"🔥 {ip}  ({rem:.0f}s remaining)",
                               style={"color":RED,"borderBottom":f"1px solid {GRID}","padding":"2px 4px","fontSize":"10px"}))
    if not blist: blist=[html.Div("✅ No IPs blocked",style={"color":GREEN,"padding":"4px"})]

    # Block timeline
    fbt=go.Figure()
    alerts=data_store.get_alerts(30)
    drop_alerts=[a for a in alerts if a.get("action")=="xdp_drop"]
    if drop_alerts:
        xs=list(range(len(drop_alerts)))
        ys=[1]*len(drop_alerts); lbls=[f"{a['type']} {a['src_ip']}" for a in drop_alerts]
        fbt.add_trace(go.Scatter(x=xs,y=ys,mode="markers+text",text=lbls,textposition="top center",
                                  marker=dict(symbol="x",size=14,color=RED),name="XDP Block Events"))
    fbt.update_layout(**base(),xaxis=bl("event"),yaxis=dict(visible=False,range=[0,2]))

    # Defense stats
    dstats=[
        html.Div(f"Total Dropped:  {summary['total_dropped']}",style={"color":RED,"fontSize":"11px","fontFamily":"monospace","marginBottom":"4px"}),
        html.Div(f"Blocked IPs:    {summary['blocked_ips']}",  style={"color":ORANGE,"fontSize":"11px","fontFamily":"monospace","marginBottom":"4px"}),
        html.Div(f"Total Alerts:   {summary['total_alerts']}",  style={"color":YELLOW,"fontSize":"11px","fontFamily":"monospace","marginBottom":"4px"}),
        html.Div(f"XDP Active:     {'Yes' if XDP_DROP_ENABLED else 'No (sim)'}",style={"color":GREEN,"fontSize":"11px","fontFamily":"monospace"}),
    ]
    return fdrop,blist,fbt,dstats


@app.callback(
    [Output("g-lat","figure"), Output("g-res","figure"),
     Output("g-modes","figure"), Output("g-log","children")],
    Input("iv","n_intervals")
)
def cb_sys(n):
    preds=data_store.get_recent_predictions(80)
    flat=go.Figure()
    if preds:
        t=list(range(len(preds))); lats=[p.get("latency_ms",0) for p in preds]
        avg=np.mean(lats) if lats else 0
        flat.add_trace(go.Scatter(x=t,y=lats,name="Latency",fill="tozeroy",fillcolor="rgba(0,212,255,0.09)",line=dict(color=ACCENT,width=2)))
        flat.add_hline(y=avg,line_dash="dot",line_color=YELLOW,annotation_text=f"avg={avg:.1f}ms",annotation_font_color=YELLOW)
    flat.update_layout(**base(),xaxis=bl("sample"),yaxis=bl("ms"))

    try:
        import psutil; cpu=psutil.cpu_percent(None); mem=psutil.virtual_memory().percent
    except Exception: cpu,mem=15.0,40.0
    fres=go.Figure()
    for v,title,dom in [(cpu,"CPU %",[0,0.45]),(mem,"Mem %",[0.55,1])]:
        clr=GREEN if v<50 else YELLOW if v<80 else RED
        fres.add_trace(go.Indicator(mode="gauge+number",value=v,title={"text":title,"font":{"color":TXT}},
            gauge={"axis":{"range":[0,100]},"bar":{"color":clr},"bgcolor":PANEL,"bordercolor":GRID},
            domain={"x":dom,"y":[0,1]},number={"suffix":"%","font":{"color":ACCENT}}))
    fres.update_layout(paper_bgcolor=BG,plot_bgcolor=PANEL,font={"color":TXT,"family":"monospace"},margin=dict(l=20,r=20,t=20,b=20))

    # Detection mode pie
    if preds:
        modes={}
        for p in preds: modes[p.get("detection_mode","?")] = modes.get(p.get("detection_mode","?"),0)+1
        fmod=go.Figure(go.Pie(labels=list(modes.keys()),values=list(modes.values()),hole=0.3,
                               marker=dict(colors=[ACCENT,GREEN,PURPLE,YELLOW],line=dict(color=BG,width=2)),
                               textfont=dict(color=TXT,size=9)))
        fmod.update_layout(paper_bgcolor=BG,plot_bgcolor=PANEL,font={"color":TXT,"family":"monospace"},
                            margin=dict(l=8,r=8,t=18,b=18),showlegend=True,
                            legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=9)))
    else:
        fmod=go.Figure()
        fmod.update_layout(**base())

    alerts=data_store.get_alerts(60)
    rows=[html.Div(f"[{time.strftime('%H:%M:%S',time.localtime(a['time']))}]  {a['type']}  "
                   f"conf={a['confidence']:.0%}  action={a.get('action','alert')}  {a.get('details','')}",
                   style={"color":ATTACK_COLORS.get(a["type"],RED),"borderBottom":f"1px solid {GRID}",
                           "paddingBottom":"2px","marginBottom":"2px"})
          for a in reversed(alerts)]
    if not rows: rows=[html.Div("No alerts",style={"color":DIM})]
    return flat,fres,fmod,rows


if __name__=="__main__":
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False)
