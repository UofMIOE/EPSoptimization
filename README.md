# EPSoptimization
Optimization Routines for the USAI EPS System

\documentclass{article}
\usepackage{amsmath}
\usepackage{algorithmic}
\makeatletter
\makeatother
\begin{document}
\begin{description}  
\item Problem Size = $P_n$ 
\item Number of Bees = $B$ 
\item Number of Other Bees = $B_o$
\item Number of Recruited Bees = $B_R$
\item Number of Leftover Bees = $B_\text{left}$
\item Number of Elite Bees = $B_E$
\item Patch Size = $Q_p$
\item Initial Patch Size = $Q_{\textbf{init},p}$
\item Patch Reduction Factor = $Q_{\textbf{-}\Delta}$
\item Number of Sites = $S$
\item Best Sites = $S^*$
\item Number of Elite Sites = $S_E$
\item Next Generation = $G_next$
\item Best Bee = $B^*$
\\
\end{description}
\begin{algorithmic}
\STATE{\textbf{Algorithm I: Bee Colony}}
\STATE{Input: $P_n$, $B$, $B_o$, $B_E$, $S$, $S_E$}
\STATE{Output: $B^*$} 
\STATE{$Population \gets$ \textbf{init} $Population(B, P_n)$}
\WHILE {\NOT $Stopping Condition()$} 
        \STATE $Evaluate Population(Population)$
        \STATE $B^* \gets BestSolution(Population)$
        \STATE $G_\text{next} \gets \emptyset$
        \STATE $Q_p \gets (Q_{\textbf{init},p}\times Q_{\textbf{-}\Delta})$
        \STATE $S^* \gets SelectBestSites(Population, S)$
\FOR{$i \in S^*$}
    \STATE $B_R \gets \emptyset$
        \IF {$i < S_E$}
        \STATE $B_R \gets B_E$
        \ELSE $B_R \gets B_o$
        \ENDIF
        \STATE $Neighborhood \gets \emptyset$
    \FOR{$j \textbf{ to } B_R$}
    \STATE $Neighborhood \gets GenerateNeighborhoodBee(i,Q_p)$
\ENDFOR 
\STATE $G_\text{next} \gets BestSolution(Neighborhood)$
\ENDWHILE
\STATE $B_\text{left} \gets (B - S)$
\FOR{$j \textbf{ to } B_\text{left}$}
\STATE $G_\text{next} \gets GenerateRandomBee()$
\ENDFOR
\STATE $Population \gets G_\text{next}$ 
\RETURN $(B^*)$
\end{algorithmic}
\end{document}
