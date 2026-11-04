"""Add the extended visual edition to the content-consolidated manuscript.

All empirical plots use the retained manuscript snapshot or inspected original
figures. Conceptual diagrams are expressly illustrative, not new observations.
"""
from pathlib import Path


def apply_visual_expansion(dest: Path):
    def save(name, content):
        (dest / name).write_text(content.strip() + "\n", encoding="utf-8")

    def insert(name, anchor, addition, before=False):
        path = dest / name
        text = path.read_text(encoding="utf-8")
        assert text.count(anchor) == 1, (name, anchor)
        replacement = addition + "\n" + anchor if before else anchor + "\n" + addition
        path.write_text(text.replace(anchor, replacement), encoding="utf-8")

    save("figure_pipeline.tex", r"""
\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  >={Latex[length=2mm]},
  box/.style={draw=black!55, rounded corners=2pt, align=left, inner sep=6pt},
  wide/.style={box,text width=10.7cm,fill=black!3},
  half/.style={box,text width=4.8cm},
  arrow/.style={->,thick,black!65}]
\node[wide] (raw) {\textbf{College-basketball panel, 2009--2021}\\
61{,}061 source rows $\longrightarrow$ \pnPanelN{} eligible player-seasons};
\node[half,fill=blue!5,anchor=north] (exp) at (-2.85,-1.3)
{\textbf{Exploratory density analysis}\\
Broader statistical representation\\
PCA (94\%) $\to$ UMAP\\
HDBSCAN: population geometry};
\node[half,fill=teal!7,anchor=north] (fit) at (2.85,-1.3)
{\textbf{Factorization input}\\
23 features; median imputation\\
Exposure normalization and scaling\\
$X\in\mathbb R_+^{45{,}980\times23}$};
\node[half,fill=teal!7,anchor=north] (base) at (-2.85,-4.2)
{\textbf{Comparative baselines}\\
NMF weights $\to$ $K$-means,\\
Gaussian mixtures, fuzzy $C$-means\\
Graph-only dynamic NMF};
\node[half,fill=teal!12,anchor=north] (court) at (2.85,-4.2)
{\textbf{COURT-NMF}\\
Exposure weights $q_i$; 421 contexts\\
23{,}479 career edges; five roles\\
Constrained updates; ten starts};
\node[wide,anchor=north] (eval) at (0,-7.2)
{\textbf{Evaluation}\\
Reconstruction and trajectory diagnostics; four ablations\\
60 full-panel initializations and 8{,}000 player-subsample runs\\
195 position labels: joint association and five-fold label prediction};
\draw[arrow] (raw.south) -- ++(0,-3mm) -| (exp.north);
\draw[arrow] (raw.south) -- ++(0,-3mm) -| (fit.north);
\draw[arrow] (fit.south) -- (court.north);
\draw[arrow] (fit.south west) -- (base.north east);
\draw[arrow] (base.south) -- ++(0,-4mm) -| (eval.north);
\draw[arrow] (court.south) -- ++(0,-4mm) -| (eval.north);
\end{tikzpicture}
\caption{Study workflow. The exploratory density representation is distinct from the 23-feature factorization input. Labels enter evaluation after fitting; classifier folds do not hold out the full representation-learning pipeline.}
\label{fig:pipeline}
\end{figure}
""")
    insert("main.tex", r"\section{Model and Estimation}\label{sec:method}", r"""
Figure~\ref{fig:pipeline} summarizes the analysis and separates exploratory geometry from the controlled factorization comparisons. Both branches use the eligible player-season population, but their feature representations differ.
\input{tex/submission/figure_pipeline}
""")

    save("figure_graph.tex", r"""
\begin{figure}[htbp]
\centering
\begin{tikzpicture}[>={Latex[length=2mm]},
 season/.style={draw,rounded corners=2pt,minimum width=2.5cm,minimum height=1.4cm,align=center},
 edge/.style={thick,black!65}]
\node[season,fill=blue!7] (a) at (0,0) {Season $t$\\Team A\\Minutes: 60\%};
\node[season,fill=blue!7] (b) at (4.2,0) {Season $t+1$\\Team A\\Minutes: 60\%};
\node[season,fill=orange!12] (c) at (8.4,0) {Season $t+2$\\Team B\\Minutes: 60\%};
\draw[edge] (a) -- node[above,align=center]{same team\\$a_{ij}=1$} (b);
\draw[edge,dashed] (b) -- node[above,align=center]{transfer\\$a_{ij}=0.35$} (c);
\node[draw=black!40,align=left,text width=10.3cm,inner sep=7pt,below=12mm of b] (note)
{\textbf{Opportunity shock:} a 35-point minutes-share change multiplies either edge by $e^{-1}\approx0.368$. The corresponding weights become 0.368 within a team and 0.129 after a transfer.\\
\textbf{No edge:} observations separated by a missing calendar season are not linked.};
\end{tikzpicture}
\caption{Illustration of the career graph in Eq.~\ref{eq:edge}. Each node carries a five-component mixture $w_i$. The penalty weights squared distances between linked mixtures; the numbers shown are an illustrative scenario, not an observed player's trajectory.}
\label{fig:career-graph}
\end{figure}
""")
    insert("main.tex", r"\paragraph{Solver and stopping rule.}", r"""
Figure~\ref{fig:career-graph} illustrates how the edge rule distinguishes roster continuity from a transfer. A weaker edge permits greater mixture change; it does not require a discontinuity or replace the reconstruction evidence for the two seasons.
\input{tex/submission/figure_graph}
""", before=True)

    save("figure_ablation.tex", r"""
\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=13mm},
width=5.3cm,height=5.7cm,xbar,/pgf/bar width=9pt,
xmin=0,ymin=-0.6,ymax=3.6,ytick={0,1,2,3},
yticklabels={No separation,No trajectory,No context,Full model},
yticklabel style={font=\small},tick label style={font=\small},
axis lines=left,xmajorgrids=true,grid style={black!10},
nodes near coords,point meta=x,nodes near coords style={font=\small,text=black,
/pgf/number format/fixed,/pgf/number format/precision=5},
every axis plot/.append style={fill=blue!65!black,draw=none}]
\nextgroupplot[title={Reconstruction MSE},xmax=0.0047,xtick={0,0.002,0.004}]
\addplot[fill=blue!65!black,draw=none] coordinates {(0.00271,0) (0.00268,1) (0.00288,2) (0.00272,3)};
\nextgroupplot[title={Trajectory MSE},xmax=0.0095,yticklabels={},xtick={0,0.004,0.008}]
\addplot[fill=blue!65!black,draw=none] coordinates {(0.00203,0) (0.00656,1) (0.00262,2) (0.00246,3)};
\end{groupplot}
\end{tikzpicture}
\caption{Ablation diagnostics from Table~\ref{tab:court-benchmark}. All plotted variants retain opportunity weighting; removing context nevertheless changes the reconstruction specification. The full-versus-no-trajectory comparison retains both context and weighting, and shows the largest drift increase. The full fit uses ten starts; ablations use seed 42.}
\label{fig:court-ablation}
\end{figure}
""")
    insert("main.tex", r"\input{tex/submission/montecarlo}", r"""
Figure~\ref{fig:court-ablation} makes the different scales of reconstruction and trajectory error explicit. Removing trajectory regularization changes reconstruction little while allowing a much larger movement in consecutive-season memberships. The separation ablation, by contrast, combines lower drift with concentrated dominant-role assignments, illustrating why small drift alone is not a sufficient quality criterion.
\input{tex/submission/figure_ablation}
""", before=True)

    insert("montecarlo.tex", r"\input{tex/submission/table_montecarlo}", r"""
Figure~\ref{fig:mc-distribution} restores the distributional view behind the intervals. Mixture reconstruction has a concentrated upper mode and a lower tail, whereas position association varies differently across starts. This is why one endpoint or one selected fit does not summarize every aspect of stability.
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig24_mc_distribution.png}
\caption{Distribution across 60 full-panel initializations of mixture reconstruction $R^2$ and CLR-based position-association $R^2$. Shading marks the empirical 2.5--97.5\% range, the colored line the mean, and the dashed line seed 42. These are run distributions, not population confidence intervals.}
\label{fig:mc-distribution}
\end{figure}
""")

    insert("appendix.tex", r"\section{Supporting Position Validation}\label{app:validation-tables}", r"""
\paragraph{Interpreting the archetypes.}
The five names refer to combinations of tendencies rather than a ranking of player quality. The foul-drawing rebounding forward combines rebounding and foul generation with relatively limited playmaking. The perimeter specialist combines high minutes with low usage and strong shooting percentages. The combo guard contributes assists and steals with greater ball-handling responsibility and turnover exposure. The bench scorer combines limited minutes with high normalized activity; its rate interpretation depends on the exposure construction described in Appendix~\ref{app:variables}. The two-way glass-cleaning scorer combines efficient finishing, rebounding, and rim-protection indicators. These descriptions restore the domain interpretation while keeping cluster identifiers specific to the exploratory fit.

The low maximum memberships and high normalized entropies show why these names need not become exclusive labels: a descriptive role vocabulary can remain useful even when the population contains many hybrids.
""", before=True)

    save("figure_softroles.tex", r"""
\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\begin{axis}[width=11.5cm,height=5.8cm,ybar,/pgf/bar width=7pt,
ymin=0,ymax=0.45,ytick={0,0.1,0.2,0.3,0.4},
xtick={0,1,2,3,4},xticklabels={Role 1,Role 2,Role 3,Role 4,Role 5},
ylabel={Mean membership},axis lines=left,ymajorgrids=true,grid style={black!10},
tick label style={font=\small},xmin=-0.5,xmax=4.5,
legend image code/.code={\draw[#1] (0cm,-0.05cm) rectangle (0.25cm,0.12cm);},
legend style={draw=none,at={(0.5,1.04)},anchor=south,legend columns=3,/tikz/column sep=2mm}]
\addplot[fill=blue!65!black,draw=none] coordinates {(0,0.386) (1,0.048) (2,0.109) (3,0.140) (4,0.317)};
\addplot[fill=orange!80,draw=none] coordinates {(0,0.350) (1,0.243) (2,0.149) (3,0.065) (4,0.193)};
\addplot[fill=teal!70,draw=none] coordinates {(0,0.379) (1,0.372) (2,0.143) (3,0.031) (4,0.075)};
\legend{Guard,Forward,Center}
\end{axis}
\end{tikzpicture}
\caption{Mean COURT-NMF membership by position from Table~\ref{tab:court-soft-validation}. Roles 3 and 4 vary by position despite never being the arg-max in this labeled sample. Role 1 shows the weakest marginal separation. Bars show descriptive means, without uncertainty intervals.}
\label{fig:soft-role-profiles}
\end{figure}
""")
    insert("appendix.tex", r"\input{tex/submission/table_softmembership}", r"""
Table~\ref{tab:court-soft-validation} shows that the large center share on Role 2 and guard shares on Roles 4 and 5 coexist with substantial common weight on Role 1. Thus position differences concern the composition of the mixture, rather than a one-to-one correspondence between five roles and three labels.
""")

    # Number and place the Monte Carlo displays in order of first mention.
    path = dest / "montecarlo.tex"
    text = path.read_text(encoding="utf-8")
    start = text.index(r"\begin{figure}[htbp]")
    end = text.index(r"\end{figure}", start) + len(r"\end{figure}")
    distribution_figure = text[start:end]
    assert r"\label{fig:mc-distribution}" in distribution_figure
    text = text[:start] + text[end:]
    path.write_text(text.rstrip() + "\n" + distribution_figure + "\n", encoding="utf-8")
