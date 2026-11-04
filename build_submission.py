"""Assemble the editorial submission without altering the original main/supplement.

Run from any directory with Python. Numerical table bodies are copied from the
supplied TeX snapshot, not regenerated from the inconsistent results archive.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "tex" / "submission"
DEST.mkdir(parents=True, exist_ok=True)
(ROOT / "output" / "pdf").mkdir(parents=True, exist_ok=True)
(ROOT / "tmp" / "pdfs" / "submission").mkdir(parents=True, exist_ok=True)
main = (ROOT / "COURT_NMF_ICMLC2027_main.tex").read_text(encoding="utf-8")
supp = (ROOT / "COURT_NMF_ICMLC2027_supplement_1.tex").read_text(encoding="utf-8")
mc = (ROOT / "tex" / "section_montecarlo.tex").read_text(encoding="utf-8")
smc = (ROOT / "tex" / "supplement_montecarlo.tex").read_text(encoding="utf-8")


def table(source, label):
    for block in re.findall(r"\\begin\{table\}.*?\\end\{table\}", source, re.S):
        if r"\label{" + label + "}" in block:
            return block
    raise ValueError(label)


def caption(block, text):
    start = block.index(r"\caption{") + len(r"\caption{")
    depth, end = 1, start
    while depth:
        depth += (block[end] == "{") - (block[end] == "}")
        end += 1
    return block[:start] + text + block[end - 1:]


def save(name, text):
    (DEST / name).write_text(text.strip() + "\n", encoding="utf-8")


# Isolate this edition's numeric snapshot from the legacy generators.
save("paper_numbers.tex", (ROOT / "tex" / "paper_numbers.tex").read_text(encoding="utf-8"))

save("main.tex", r"""
\mainmatter
\title{COURT-NMF: Contextual, Opportunity-Weighted, Role-Trajectory Factorization for Longitudinal Panels}
\titlerunning{COURT-NMF}
\author{Haoran Li}
\authorrunning{H. Li}
\tocauthor{Haoran Li}
\institute{Capital University of Economics and Business, Beijing, China}
\maketitle
\begin{abstract}
Longitudinal role discovery must distinguish changes in latent roles from unequal exposure, contextual shifts, and discontinuities between observations. COURT-NMF combines simplex-constrained role mixtures, opportunity weighting, bounded context offsets, a transfer-aware temporal graph, and prototype separation in one non-negative factorization objective. An alternating solver uses projected-gradient updates with backtracking and selects the lowest-objective fit from multiple initializations. On a college-basketball panel of \pnPanelN{} player-seasons and \pnFeatures{} style features, removing the trajectory penalty increases adjacent-season mixture drift by a factor of \pnTrajRiseFactor{}, while weighted reconstruction error changes by \pnWMSEChange{}. Validation against \pnValN{} independently sourced position labels yields joint mixture association ($R^2=\pnPermR$, permutation $p=\pnPermP$) and five-fold classifier accuracy of \pnAcc{} against a \pnBaseline{} majority baseline. Sixty full-panel initializations characterize optimization variability; \pnMCSubR{} player-subsample runs additionally assess sensitivity to sample composition. These are descriptive, full-panel results: label cross-validation does not establish temporal generalization. The internal appendices provide configuration, validation, and sensitivity details within this single document.
\keywords{NMF, longitudinal panels, graph regularization, role mixtures}
\end{abstract}

\section{Introduction}\label{sec:intro}
An observation in a longitudinal panel may change because its context or exposure changed, even when its underlying role did not. In basketball, a player's statistics depend on playing time and team environment as well as playing style. Standard non-negative matrix factorization (NMF) approximates $X$ by $WH$, but treats rows independently, leaves the scale of their memberships unconstrained, and does not distinguish continuity within an organization from change after a transfer. Row-normalizing a fitted $W$ produces proportions after estimation; it does not impose that interpretation on the reconstruction objective.

COURT-NMF places role proportions on the probability simplex during estimation and combines reliability weighting, context adjustment, and transfer-aware continuity in the same objective. A prototype-separation penalty discourages redundant role directions. The contribution is this joint constrained formulation and its implemented estimation procedure, evaluated through baselines, component ablations, and external-label validation. The individual ingredients have established precedents; the empirical question is what their combination contributes on a panel exhibiting all three sources of variation.

College basketball provides a useful application because playing time varies widely, conference-season conditions change, and players can transfer between teams. It also raises a substantive question: whether traditional positions adequately describe a population of overlapping skill profiles \cite{alagappan2012,terner2021,oliver2004}. We use five roles as an interpretable modeling choice, rather than claim that the population contains five naturally separated groups.

\paragraph{Related work.}
NMF learns additive, non-negative parts \cite{lee1999,lee2001}. Graph-regularized NMF introduces neighborhood dependence \cite{cai2011}, while online and state-space approaches allow temporal variation \cite{mroueh2007,mohammadiha2015}. These motivate the factorization and continuity components considered here. In basketball, foundational box-score metrics \cite{kubatko2007}, tracking-based possession models \cite{cervone2016}, and lineup analysis using clustered player tendencies \cite{kalman2020} support complementary descriptions of performance. Position-specific clustering and data-driven archetypes \cite{duman2024,guo2025,penner2025} emphasize role heterogeneity; longitudinal role mapping and transfer research \cite{tempo2026,cole2026} motivate explicit treatment of continuity. Our evaluation focuses on the combined estimation problem rather than a broad comparison with unrelated match-outcome or revenue models.

\section{Model and Estimation}\label{sec:method}
Let $X\in\mathbb R_+^{n\times f}$ contain the scaled style features. The parameters are role mixtures $W\in\mathbb R_+^{n\times K}$, prototypes $H\in\mathbb R_+^{K\times f}$, and offsets $B\in\mathbb R^{G\times f}$. Observation $i$ belongs to context $g(i)$; each row $w_i$ sums to one. We use $K=5$ and $G=\pnContexts{}$ conference-season groups. With $h_r$ denoting prototype $r$, the objective is
\begin{equation}
\begin{aligned}
\mathcal L(W,H,B)={}&\sum_i q_i\|x_i-b_{g(i)}-w_iH\|_2^2\\
&+\lambda_T\sum_{(i,j)\in\mathcal E}a_{ij}\|w_i-w_j\|_2^2\\
&+\lambda_D\sum_{r\ne s}(h_r^\top h_s)^2
+\lambda_S\sum_i\|w_i\|_1,
\end{aligned}\label{eq:court-objective}
\end{equation}
subject to $W,H\geq0$, $W\mathbf1=\mathbf1$, and $B\in[-0.30,0.30]^{G\times f}$. The implementation centers the estimated offsets before clipping; Appendix~\ref{app:court} specifies that operation. The $\ell_1$ term is constant on the simplex, $\sum_i\|w_i\|_1=n$, and therefore does not induce sparse memberships.

\paragraph{Exposure and continuity.}
Writing $[z]_a^b=\min(b,\max(a,z))$, the implemented reliability weight is
\begin{equation}
q_i=\sqrt{[\mathrm{Min\%}_i/100]_{0.05}^{1}
                  [\mathrm{GP}_i/35]_{0.25}^{1}}.
\end{equation}
This reduces the influence of low-exposure rows that pass the eligibility filter. The graph $\mathcal E$ links consecutive calendar seasons of the same player. Its weights are
\begin{equation}
a_{ij}=\exp\!\left(-\frac{|\mathrm{Min\%}_i-\mathrm{Min\%}_j|}{35}\right)
\begin{cases}1,&\text{same team},\\0.35,&\text{different teams}.
\end{cases}\label{eq:edge}
\end{equation}
Thus comparable opportunity within a team favors continuity, while transfers and minutes shocks weaken the penalty. The graph encodes this assumption; it does not identify whether a particular transition caused a role change. The separation term discourages overlapping prototype directions without imposing orthogonality.

\paragraph{Solver and stopping rule.}
Each iteration computes a weighted residual-mean update for $B$, centers and clips it, then updates $W$ by projection onto the row simplex and $H$ by projection onto non-negative values (a numerical floor of $10^{-8}$). For the $W$ and $H$ updates, backtracking accepts a candidate when the objective does not increase and otherwise halves the step. This is a non-increase test, rather than a formal Armijo sufficient-decrease condition. It applies to these two blocks; the separate centered-and-clipped context update is not covered by that acceptance test.

We record the objective every 25 iterations and stop when its relative change is below $\pnTol$ for \pnPatience{} consecutive checks, subject to a \pnIterCap{}-iteration cap. Objective stabilization is an operational convergence criterion, not a certificate of stationarity or global optimality. With $\lambda_T=0.35$, $\lambda_D=0.12$, and $\lambda_S=0.008$, the reported full model is the lowest-objective fit among \pnRestarts{} starts with seeds $42,\ldots,51$; it stops at iteration \pnStopIter{}. Baselines and ablations use the configurations in Appendix~\ref{app:court}. For fixed backtracking effort, the main matrix operations cost $O(nKf+|\mathcal E|K+K^2f)$ per iteration. Appendix~\ref{app:solver} compares alternative step rules.

\section{Data and Application Baselines}\label{sec:data}
The source panel contains 61{,}061 player-season rows from 2009--2010 through 2020--2021. Requiring minutes share above 5\%, at least 10 games, and at least one recorded point leaves \pnPanelN{} observations. The factorization uses \pnFeatures{} non-negative scaled style variables, \pnContexts{} context groups, and \pnEdges{} consecutive-season career edges. Appendix~\ref{app:variables} lists the fitted features and distinguishes the current implementation from the exploratory preprocessing described in the source study.

\subsection{Baseline structure}\label{sec:baselines}
The exploratory PCA--UMAP--HDBSCAN pipeline produces one cluster containing 98.5\% of observations and three small peripheral clusters. This supports a continuous role description in that embedding, although an embedding-based density result cannot establish the geometry of every alternative representation. Appendix~\ref{app:baselines} gives the configuration and cluster diagnostics.

Five-component NMF followed by $K$-means supplies an interpretable static vocabulary. The exploratory silhouette is 0.218 at $K=5$, compared with 0.345 at $K=2$; the latter mainly separates interior and perimeter play. Five clusters distinguish additional usage and bench-role patterns, but their names are descriptive prototypes, not ground-truth classes. Row-normalizing the NMF weights gives mean maximum membership 0.386 and normalized entropy above 0.75 for 89.9\% of seasons, consistent with substantial overlap.

The comparative benchmark also includes Gaussian mixtures and fuzzy $C$-means on the five-dimensional NMF weights. Dynamic NMF is the graph-only variant of Eq.~\ref{eq:court-objective}: context adjustment, opportunity weights, and separation are removed, while the simplex constraint and temporal term remain. These comparisons separate a hard role vocabulary, soft static assignments, and longitudinal regularization. The exploratory density pipeline uses a broader feature representation, so its silhouette is not a controlled comparison of factorization objectives.

\section{Results}\label{sec:results}
\subsection{Benchmark and ablations}\label{sec:benchmark}
Table~\ref{tab:court-benchmark} reports the benchmark and four term-removal ablations. These are full-panel diagnostics, not temporal hold-out estimates. The full COURT-NMF row uses multi-start selection; the ablations use seed 42 and the same stopping criterion, so the comparisons do not average over initialization variability. Silhouette assesses hard-assignment compactness in the representation used by each method; the canonical HDBSCAN value is measured in its UMAP embedding with noise excluded.
\input{tex/submission/table_benchmark}

Removing the trajectory term increases adjacent-season mixture MSE from \pnTrajMSE{} to \pnTrajMSENoTraj{} (\pnTrajRise{}, or \pnTrajRiseFactor{} times), while weighted reconstruction MSE changes by \pnWMSEChange{}. This matched comparison supports the intended continuity effect; it does not by itself establish better forecasts. Removing separation produces a 0.957 modal-role share, indicating that one role dominates the hard assignments even though memberships remain soft. Removing opportunity weighting gives trajectory MSE \pnNoOppTrajMSE{}, near the graph-only baseline's \pnDynTrajMSE{}.

Reconstruction values use each variant's own weighting and context choice. Consequently, differences involving removed opportunity weights or context are descriptive rather than matched-loss estimates. Negative silhouettes for several COURT-NMF variants indicate weak hard-cluster separation; the model is evaluated principally through reconstruction, continuity, and validation of its mixtures. Appendix~\ref{app:court} adds membership and stopping diagnostics, and Appendix~\ref{app:transfer-sensitivity} reports the transfer-gate sweep.

\input{tex/submission/montecarlo}

\section{External Validation}\label{sec:validation}
Position labels enter only after fitting. The validation join contains \pnValN{} independently sourced guard/forward/center labels (72/80/43). On centered-log-ratio (CLR) transformed mixtures, a permutation MANOVA tests joint association with position. A stratified five-fold multinomial classifier evaluates label prediction against a \pnBaseline{} majority baseline, with \pnPerms{} label permutations for both tests (Table~\ref{tab:court-multivariate}). CLR uses a $10^{-9}$ floor before logarithms. The factorization and scaling are fit on the full unlabeled panel; only the supervised classifier is cross-validated. This is out-of-fold label prediction within a fixed representation, not an end-to-end held-out or future-season experiment.
\input{tex/submission/table_validation}

The reported joint association is $R^2=\pnPermR$, with classifier accuracy \pnAcc{} and macro-$F_1$ \pnMacroF{}. Marginal Kruskal--Wallis tests also detect position differences in \pnRolesSurvivingWord{} of five memberships after Bonferroni correction; Role~\pnBestRole{} has the largest statistic ($H=\pnBestH$, adjusted $p=\pnBestAdjP$). Appendix~\ref{app:validation-tables} retains all marginal results and dominant-role counts. The static baseline's position association ($\chi^2=247.52$, 8 degrees of freedom, $p=5.78\times10^{-49}$) provides an additional descriptive comparator.

\section{Discussion and Limitations}\label{sec:discussion}\label{sec:limitations}
A density continuum and a five-role factorization answer different questions. Density clustering seeks separated populations; factorization approximates observations with a small set of shared directions. Overlap between scoring, rebounding, and defensive activity need not produce distinct density islands. Silhouette remains a useful compactness diagnostic \cite{rousseeuw1987,mcinnes2017}, but it is insufficient to judge a longitudinal mixture model.

The five memberships have four independent degrees of freedom because they sum to one. Joint validation therefore complements the marginal tests. Hardening to an arg-max discards information: \pnEmptyRolesWord{} roles are never dominant in the labeled sample, although their soft weights vary with position. Neither the significant joint tests nor an association with three legacy positions proves that the five roles are uniquely identified, or that they improve scouting or lineup decisions. The validation establishes association and label-predictive information within this panel.

The study is restricted to U.S. men's college basketball, historical box-score data, rotation players, and a small external-label sample. It lacks off-ball and defensive-positioning measurements, and the eligibility filter limits generalization to low-exposure players. The context offsets are descriptive adjustments. The number of roles, graph discounts, and regularization weights are modeling choices; stability under the transfer-gate sweep does not make them optimal. Full-panel and subsample Monte Carlo experiments quantify different sources of variation and do not replace temporal validation. The full model's additional restarts also warrant matched multi-start ablations before attributing every gap to a single objective term.

The main next steps are expanding-window evaluation with preprocessing fit only on training data, uncertainty intervals for matched ablation gaps, and tests of next-season usage, transfer outcomes, or lineup fit. Stronger optimization guarantees would require analysis of the complete context-update procedure and a stationarity-based stopping check.

\section{Conclusion}\label{sec:conclusion}
COURT-NMF integrates simplex role mixtures, exposure weighting, bounded context adjustment, and transfer-aware continuity in one factorization framework. The trajectory ablation supports its intended smoothing effect, while joint and marginal label validation support the descriptive relevance of the mixtures. Multiple initializations expose variation hidden by a single fit. These results motivate the model as a longitudinal representation, with prospective utility and stronger convergence guarantees left to further evaluation.

\paragraph{Reproduction materials.}
The internal appendices contain the supporting methods and tables. Analysis scripts, input data, and result files are maintained in the project archive. A public repository link is not provided in this version.
""")

save("montecarlo.tex", r"""
\subsection{Monte Carlo convergence and stability}\label{sec:montecarlo}
Non-convexity makes a single fit an incomplete description of the estimator. We report two designs: 60 independent initializations on the complete panel and \pnMCSubR{} runs on subsamples of 600 players, retaining all seasons of each selected player. The subsample design anchors the players contributing the 195 validation observations and randomly selects the remaining players. Position values remain excluded from estimation. Full-panel runs isolate initialization effects; subsample runs additionally vary the fitted population.

All 60 full-panel runs meet the objective-change criterion within 4,000 iterations, with a median stopping iteration of 1038. Final objectives span 5.26\% from best to worst; 23.3\% of runs stop more than 0.1\% above the best value observed. These are empirically worse stabilized solutions, not certified local minima. Figure~\ref{fig:mc-convergence} shows the reconstruction and trajectory diagnostics during optimization; Table~\ref{tab:montecarlo} reports the resulting variation. Its intervals are empirical 2.5--97.5\% ranges across runs, not confidence intervals for population performance.

After optimal role matching, mean prototype cosine similarity to the seed-42 reference is 0.9679 [0.8519, 0.9997]. Dominant-role adjusted Rand agreement is 0.6958 [0.3185, 0.8953], showing that prototype similarity can coexist with substantial hard-assignment variation. The seed-42 mean top-role weight lies at the eighth percentile of the restart distribution. It is retained as a common comparator here; the benchmark uses the best of ten starts and therefore need not match it exactly.

The subsample experiment converges in 99.4\% of runs, with mean label accuracy 0.761 [0.703, 0.810], mixture reconstruction $R^2$ 0.4470 [0.4078, 0.4726], and matched prototype cosine similarity 0.8203 [0.7241, 0.9674]. The weaker recovery in this design reflects both sample composition and initialization. Its objective values cannot be ranked across different samples as if all runs optimized the same data.

Across full-panel runs, objective per observation correlates negatively with mixture $R^2$ (Spearman $-0.350$) and classifier accuracy ($-0.389$). Resampling 4,000 sets of existing runs at each restart count estimates mean selected accuracy of 0.8206 at one start and 0.8285 at 50 starts; corresponding mixture $R^2$ values are 0.6146 and 0.6200. These empirical gains motivate the ten-start protocol, but do not guarantee improvement on every metric or new dataset. Appendix~\ref{app:solver} supplies seeds and the step-rule comparison.
\input{tex/submission/table_montecarlo}
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig23_mc_convergence.png}
\caption{Optimization diagnostics over 60 full-panel initializations. The solid line is the median, darker shading the interquartile range, and lighter shading the 5--95\% range; the dashed line is seed 42. Each diagnostic uses a separate logarithmic axis.}
\label{fig:mc-convergence}
\end{figure}
""")

save("appendix.tex", r"""
\clearpage
\appendix
\renewcommand{\theHsubsection}{appendix.\Alph{section}.\arabic{subsection}}
\section{Data and Baseline Details}\label{app:variables}\label{app:baselines}
\subsection{Feature construction and missingness}
The exploratory source describes 62 raw columns and a missingness-based preparation scheme: drop fields missing in more than 30\% of rows (draft pick 98\%, recruit rank 70\%, dunk percentage 55\%); use chained-equation imputation for shot-location and ratio fields missing in 8--17\%; and use median or mode for sparse missingness below 0.1\%. Identifiers, opportunity measures, and redundant made/attempted pairs are not intended as style coordinates.

The factorization script applies the eligibility filter first, median-fills the selected numeric fields, and normalizes seven count variables by exposure. For count $c_i$, the columns named \texttt{per100} are calculated as
\[
z_i=100c_i/\mathrm{Min\%}_i.
\]
The denominator is minutes share, not an observed possession count. These are therefore features normalized by exposure; their values do not establish a rate per 100 possessions. This distinction also limits the interpretation of the exploratory bench-scorer rate.

The fitted 23 features include seven normalized counts: points, total rebounds, assists, steals, blocks, offensive rebounds, and defensive rebounds. The remaining variables are usage; effective field goal and true shooting percentages; offensive rebound, defensive rebound, assist, and turnover percentages; free throw, two point, and three point percentages; block and steal percentages; free throw rate; and overall, offensive, and defensive box plus/minus. RobustScaler followed by MinMaxScaler produces the non-negative matrix. The implementation uses the full analytic panel for both scaling and factorization.

\subsection{Density configuration}\label{app:umap-hdbscan}
The exploratory density pipeline retains 16 PCA components (94\% explained variance), applies two-dimensional UMAP, and then HDBSCAN \cite{mcinnes2018umap,campello2015}; 45{,}285 of 45{,}980 observations (98.5\%) fall in one main cluster, with an overall embedding silhouette of 0.380 after excluding 12 noise points. This core--periphery result persists for minimum cluster sizes 5--30 (the two smallest peripheral clusters are absorbed at 50), but it describes density geometry rather than the five-role factorization.

\subsection{Static factors and alternative assignments}\label{app:nmf-archetypes}
The direct-NMF baseline uses five components, \texttt{init=nndsvd}, 500 maximum iterations, and seed 42 on the scaled indicators \cite{pedregosa2011}. It avoids shifting signed PCA scores solely to satisfy non-negativity. The exploratory $K$-means sweep yields silhouettes 0.345, 0.235, 0.228, 0.218, 0.208, 0.206, 0.193, 0.181, and 0.172 for $K=2,\ldots,10$. The benchmark rerun in Table~\ref{tab:court-benchmark} reports 0.217 at $K=5$; the exploratory value and its cluster shares belong to the earlier configuration and should not be equated with the rerun's modal-role share.

The exploratory cluster names are foul-drawing rebounding forward (27.8\%), high-minute perimeter shooting specialist (34.7\%), efficient floor-spacing combo guard (24.8\%), high-volume bench scorer (3.8\%), and two-way glass-cleaning scorer (8.8\%). They summarize profile contrasts; their indices do not identify COURT-NMF factors. In particular, the bench group combines high exposure-normalized scoring with limited minutes, illustrating why raw totals and normalized profiles can yield different interpretations.

Normalizing NMF activations as $p_{ir}=W_{ir}/\sum_sW_{is}$ gives mean maximum membership 0.386 (median 0.378); 89.9\% of seasons have normalized entropy above 0.75, the effective archetype count is 3.65, and only 0.85\% exceed 0.60 in one component. These summaries come from the exploratory static NMF fit, not the selected COURT-NMF model. Gaussian mixtures and fuzzy $C$-means \cite{bezdek1981} supply soft alternatives. Fuzzy $C$-means has silhouette 0.193, ARI 0.627 against $K$-means, entropy 0.798, and Cramer's $V=0.707$. Additional exploratory spectral clustering \cite{luxburg2007} on 8,000 observations and a local sparse-subspace approximation \cite{elhamifar2013} on 1,500 give silhouettes 0.197 and 0.177. These sample-based checks do not establish superiority over methods fitted on the complete panel.

\section{Supporting Position Validation}\label{app:validation-tables}
Tables~\ref{tab:court-role-position} and~\ref{tab:court-soft-validation} retain the direct COURT-NMF validation details. Arg-max assignment leaves \pnEmptyRoleList{} empty. Removing those empty rows gives $\chi^2=\pnDomChi$, $\mathrm{df}=\pnDomDf$, $p=\pnDomP$, and Cramer's $V=\pnDomV$; the minimum expected count is 8.82. This is a valid contingency calculation on the nonempty rows, but a lossy description of the five-coordinate mixture. The soft-membership tests demonstrate variation that hardening would conceal.
\input{tex/submission/table_roleposition}
\input{tex/submission/table_softmembership}

For comparison, the static baseline has the following counts in guard, forward, center order: $(0,38,0)$, $(39,1,0)$, $(0,26,14)$, $(0,9,29)$, and $(33,6,0)$. All expected counts exceed five. Their association is reported in Section~\ref{sec:validation}. Cluster indices are specific to their fitted model.

\section{Implementation and Sensitivity}\label{app:court}
\subsection{Update details and diagnostics}
Initialize $W$ by projecting a seeded uniform random matrix onto the simplex, set $H$ to positive random values, and set $B=0$. Let $Q=\operatorname{diag}(q_i)$, $R=WH+B_g-X$, $L$ be the weighted graph Laplacian, and $M=HH^\top$ with its diagonal set to zero. The implemented gradients are
\begin{align}
\nabla_W\mathcal L&=2QRH^\top+2\lambda_TLW+\lambda_S\mathbf1,\\
\nabla_H\mathcal L&=2W^\top QR+4\lambda_DMH.
\end{align}
The candidate steps divide these gradients by $f$ and $n$, respectively. Initial step sizes are 0.08 for $W$ and 0.04 for $H$; each unsuccessful attempt halves the step, with at most 20 attempts. An accepted step is multiplied by 1.15 for the next iteration. If all attempts fail, the block remains unchanged and its initial step size is restored.

For context $g$, form the weighted residual mean
\begin{equation}
\widetilde b_g=\frac{\sum_{i:g(i)=g}q_i(x_i-w_iH)}{\sum_{i:g(i)=g}q_i},
\qquad
b_g=\operatorname{clip}_{[-0.30,0.30]}
\left(\widetilde b_g-\frac1G\sum_h\widetilde b_h\right).
\end{equation}
Clipping enforces the box constraint but can alter the zero mean obtained by centering. This implementation is not an exact projection onto the intersection of the box and centered subspace. The reported stopping rule assesses the recorded objective, without a gradient or KKT-residual test.

Full-model selection uses seeds 42--51; ablations and the graph-only baseline use seed 42. Table~\ref{tab:court-diagnostics} supplies the stopping iterations and soft-membership diagnostics omitted from the main benchmark. Reconstruction and trajectory MSEs are already reported in Table~\ref{tab:court-benchmark}. Weighted MSE is $\sum_iq_i\|R_i\|^2/(f\sum_iq_i)$; trajectory MSE averages squared membership-coordinate differences over all career edges without the edge weights used in the training penalty.
\input{tex/submission/table_diagnostics}

\subsection{Transfer-gate sensitivity}\label{app:transfer-sensitivity}
Table~\ref{tab:transfer-gate-sensitivity} compares transfer gates of 0.10, 0.35, 0.60, and 0.90 at the same seed, holding all other settings fixed. Weighted MSE ranges from \pnGateWMSELo{} to \pnGateWMSEHi{}. Cross-team trajectory MSE decreases from \pnGateCrossLo{} to \pnGateCrossHi{}. Larger gates therefore impose more continuity as intended, with little change in reconstruction on this panel. This is a fixed-sample sensitivity check, not optimization of the gate. The 0.35 row is a seed-42 fit and differs slightly from the selected full model.
\input{tex/submission/table_sensitivity}

\section{Monte Carlo Protocol and Solver Comparison}\label{app:solver}\label{sup:montecarlo}
\subsection{Design and selection}
The full-panel design uses seeds $100000+r$ for $r=0,\ldots,59$, distinct from the reference and selected-fit seeds. All runs hold $K$, the regularization weights, the graph gate, and the 4,000-iteration cap fixed. The reference seed-42 fit stops at iteration 650; the best-of-ten fit in the benchmark stops at \pnStopIter{}. The mean full-panel run time reported in the study is 67.0 seconds; timing is implementation- and hardware-dependent.

The 8,000-run design samples players rather than independent rows, preserving their consecutive-season edges. All players contributing the 195 validation observations are retained among the 600 selected players. Consequently, variation in the resulting validation metrics is conditional on this anchored evaluation set. A best-objective comparison across these changing samples has no common objective scale; no local-minimum fraction is inferred for that design.

For restart selection, sets of $N$ full-panel fits are resampled 4,000 times and the lowest-objective member is retained. At $N=1$, selected accuracy is 0.8206 [0.7846, 0.8308]; at $N=50$, it is 0.8285 [0.8256, 0.8308]. These resampling ranges describe selection from the available runs. They are distinct from the direct run quantiles in Table~\ref{tab:montecarlo}, and they do not establish an optimal restart count.

\subsection{Alternative step rules}
With the same seed-42 initialization, projected-gradient backtracking attained objective 2169.1, whereas the extended multiplicative rule---which requires row normalization to restore the simplex constraint and therefore lacks the classical unconstrained-NMF monotonicity argument---attained 2583.6, or 19.1\% higher; this is an empirical comparison, not an impossibility result for multiplicative methods.

\paragraph{Artifact organization.}
The project archive separates benchmark and ablation outputs, memberships, prototypes, offsets, position validation, and Monte Carlo summaries. Per-run CSVs and objective traces preserve individual runs and solver comparisons; Table~\ref{tab:montecarlo} presents the principal distributional results. The summary statistics in this document are sufficient to assess the stated empirical claims without a separate supplementary PDF.
""".replace("\\begin{lrrr}\n\\end{lrrr}\n", ""))

# Supporting numerical tables: retain values, shorten captions, use class float font.
for name, label, cap in [
    ("table_roleposition", "tab:court-role-position", r"Dominant COURT-NMF role by position, $N=\pnValN$. Daggers mark empty rows excluded from the contingency test ($\mathrm{df}=\pnDomDf$)."),
    ("table_softmembership", "tab:court-soft-validation", r"Mean soft membership by position and per-role Kruskal--Wallis statistics. Adjusted $p$ values use a five-role Bonferroni correction."),
    ("table_validation", "tab:court-multivariate", r"Joint mixture validation, $N=\pnValN$, with \pnPerms{} permutations. The classifier uses five folds and a \pnBaseline{} majority baseline."),
    ("table_sensitivity", "tab:transfer-gate-sensitivity", r"Transfer-gate sweep at seed 42. Other settings and the stopping criterion are held fixed."),
]:
    block = table((ROOT / "tex" / (name + ".tex")).read_text(encoding="utf-8"), label)
    block = re.sub(r"\\centering\\(?:small|footnotesize|scriptsize)", r"\\centering", block)
    if name == "table_validation":
        block = block.replace(r"\setlength{\tabcolsep}{0pt}", "")
        block = block.replace(r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l l r@{}}", r"\begin{tabular}{p{4cm}lr}").replace(r"\end{tabular*}", r"\end{tabular}")
        block = block.replace("PERMANOVA on CLR mixtures", "PERMANOVA (CLR)").replace("5-fold multinomial classifier", "Multinomial classifier")
        block = block.replace(r"acc.\ $=\pnAcc$, macro-$F_1=\pnMacroF$", r"accuracy $=\pnAcc$")
    if name == "table_softmembership":
        # Raw and corrected p convey the same five-test correction; retain corrected p.
        block = block.replace("{lrrrrrr}", "{lrrrrr}")
        block = block.replace(r" & \textbf{$p$}", "")
        lines = []
        for line in block.splitlines():
            if re.search(r"Role [1-5] &", line):
                cells = line.split(" & ")
                del cells[5]
                line = " & ".join(cells)
            lines.append(line)
        block = "\n".join(lines)
    save(name + ".tex", caption(block, cap))

bench = table((ROOT / "tex" / "table_benchmark.tex").read_text(encoding="utf-8"), "tab:court-benchmark")
rows = []
for line in bench.splitlines():
    if " & " in line and line.rstrip().endswith(r"\\") and any(s in line for s in ("NMF", "Gaussian", "Fuzzy", "HDBSCAN", "without")):
        c = line.strip().removesuffix(r"\\").strip().split(" & ")
        rows.append(" & ".join([c[0], c[1], c[3], c[5], c[6]]) + r" \\")
save("table_benchmark.tex", r"""
\begin{table}[htbp]
\centering
\caption{Benchmark and ablations. Modal share is the proportion of seasons sharing the most frequent arg-max role. It differs from mean top-role weight (Table~\ref{tab:court-diagnostics}). Reconstruction MSE uses each variant's weights and context specification.}
\label{tab:court-benchmark}
\begin{tabular}{>{\raggedright\arraybackslash}p{4.1cm}rrrr}
\toprule
Method & Silhouette & Modal & Weighted & Trajectory\\
 & & share & MSE & MSE\\\midrule
""" + "\n".join(rows).replace("UMAP + HDBSCAN (canonical)", "UMAP + HDBSCAN").replace("Dynamic NMF (graph term only)", "Dynamic NMF (graph only)") + r"""
\bottomrule
\end{tabular}
\end{table}
""")

mcrows = []
for line in table(mc, "tab:montecarlo").splitlines():
    if " & " in line and "[" in line and line.rstrip().endswith(r"\\"):
        c = line.strip().removesuffix(r"\\").strip().split(" & ")
        mcrows.append(" & ".join([c[0], c[1], c[3], c[4]]) + r" \\")
save("table_montecarlo.tex", r"""
\begin{table}[htbp]
\centering
\caption{Sixty full-panel initializations. Intervals are 2.5--97.5\% empirical run quantiles. Seed 42 is the reference fit, whereas Table~\ref{tab:court-benchmark} selects the best of ten starts.}
\label{tab:montecarlo}
\begin{tabular}{lrrr}
\toprule Quantity & Mean & 95\% run interval & Seed 42\\\midrule
""" + "\n".join(mcrows).replace("Weighted reconstruction MSE", "Weighted MSE").replace("Mixture reconstruction $R^2$", "Mixture $R^2$").replace("External PERMANOVA $R^2$", "PERMANOVA $R^2$") + r"""
\bottomrule
\end{tabular}
\end{table}
""")

diag = (ROOT / "tex" / "table_diagnostics.tex").read_text(encoding="utf-8")
diagrows = []
entropy = ["0.815", "0.848", "0.841", "0.849", "0.803", "0.839"]
for line in diag.splitlines():
    if " & " in line and any(s in line for s in ("NMF", "Without")):
        c = line.strip().removesuffix(r"\\").strip().split(" & ")
        diagrows.append(" & ".join([c[0], c[3], entropy[len(diagrows)], c[4]]) + r" \\")
save("table_diagnostics.tex", r"""
\begin{table}[htbp]
\centering
\caption{Membership and stopping diagnostics. Mean top-role weight averages each season's largest membership. Entropy is normalized by $\log K$; stopped denotes the iteration satisfying the objective-change criterion.}
\label{tab:court-diagnostics}
\begin{tabular}{lrrr}
\toprule Variant & Mean top-role & Entropy & Stopped\\
 & weight & & \\\midrule
""" + "\n".join(diagrows).replace("Dynamic NMF (graph term only)", "Dynamic NMF (graph only)") + r"""
\bottomrule
\end{tabular}
\end{table}
""")

solverrows = []
for line in table(smc, "tab:solver-comparison").splitlines():
    if " & " in line and any(s in line for s in ("gradient", "Multiplicative")):
        c = line.strip().removesuffix(r"\\").strip().split(" & ")
        solverrows.append(" & ".join([c[0], c[1], c[3], c[4]]) + r" \\")
save("table_solver.tex", r"""
\begin{table}[htbp]
\centering
\caption{Step rules from seed 42. Failure to converge refers to the 20,000-iteration cap for the fixed-step rule.}
\label{tab:solver-comparison}
\begin{tabular}{>{\raggedright\arraybackslash}p{4.1cm}rrr}
\toprule Step rule & Stopped & Objective & Above best\\\midrule
""" + "\n".join(solverrows).replace("Projected gradient, Armijo backtracking (adopted)", "Projected gradient, backtracking").replace("Exponentiated gradient (entropic mirror descent)", "Exponentiated $W$, multiplicative $H$").replace("Multiplicative updates (Lee--Seung, extended)", "Extended multiplicative updates").replace("did not converge", "not converged") + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# Extended visual edition requested after the initial 15-page consolidation.
from submission_visuals import apply_visual_expansion
apply_visual_expansion(DEST)

# One bibliography, retaining only citations that survive the content revision.
alltext = "\n".join(p.read_text(encoding="utf-8") for p in DEST.glob("*.tex") if p.name != "references.tex")
cited = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", alltext) for key in group.split(",")}
refs = {}
for source in (main, supp):
    bib = source.split(r"\begin{thebibliography}{99}")[1].split(r"\end{thebibliography}")[0]
    for key, entry in re.findall(r"\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\Z)", bib, re.S):
        refs.setdefault(key, entry.strip())
assert not cited - refs.keys(), cited - refs.keys()
save("references.tex", "\\begin{thebibliography}{99}\n" + "\n\n".join("\\bibitem{" + k + "}\n" + refs[k] for k in refs if k in cited) + "\n\\end{thebibliography}")

(ROOT / "COURT_NMF_ICMLC2027_submission.tex").write_text(r"""% Single-document editorial edition. Original main/supplement sources preserved.
% Body, headings, captions and spacing use svproc.cls defaults.
\documentclass{svproc}
\usepackage{url}
\def\UrlFont{\rmfamily}
\usepackage{amsmath,amssymb}
\usepackage{booktabs,array,graphicx}
\usepackage{tikz,pgfplots}
\usetikzlibrary{arrows.meta,positioning,calc}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\usepackage{cite}
\usepackage[hidelinks]{hyperref}
\newcommand{\hdr}[2]{\begin{tabular}[b]{@{}r@{}}#1\\#2\end{tabular}}
\input{tex/submission/paper_numbers}
\begin{document}
\input{tex/submission/main}
\input{tex/submission/appendix}
\input{tex/submission/references}
\end{document}
""", encoding="utf-8")
print(f"Wrote submission source, {len(cited)} references, and isolated editorial fragments.")
