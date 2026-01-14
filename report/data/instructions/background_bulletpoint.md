# Instructions for Expanding Background Bulletpoints

## Task Overview
You are given a section from a background chapter with high-level concept bulletpoints. Your task is to expand each bulletpoint by finding relevant academic papers and extracting concise sentences that relate to the concept. If multiple papers share similar findings or concepts, merge them into a single sentence with multiple citations.

## Input Format
You will receive:
- A section heading (e.g., "Background Subtraction and Motion Detection")
- Multiple bulletpoints describing high-level concepts

## Output Format Requirements

### Part 1: Expanded LaTeX Content
For each bulletpoint, structure your output as follows:

```
\subsection{[Original Bulletpoint Concept]}
\begin{itemize}
    \item [High-level concept statement from original bulletpoint]
    \begin{itemize}
        \item \cite{authorYear} [ONE sentence extracted from this paper that relates to the concept in the context of multi-camera drone/bird detection and 3D reconstruction]
        \item \cite{authorYear, authorYear2} [ONE merged sentence covering overlapping concepts from multiple papers]
    \end{itemize}
\end{itemize}
```

### Citation Guidelines
- Find 3-5 relevant academic papers per bulletpoint concept
- Prioritize recent high quality (HIGHLY CITED, FOUNDATIONAL) papers (2021-2025) and seminal works
- Focus on papers related to: drone detection, bird detection, multi-view geometry, 3D reconstruction, Gaussian splatting, small object detection
- Each sentence should be directly usable in the final thesis
- Sentences should be concise (max 25 words) and self-contained
- Merge overlapping self-contained sentences into one statement with multiple citations (e.g., \cite{paper1, paper2})
- Use citation keys in format: firstAuthorLastNameYear (e.g., smith2023, zhang2024)

### Part 2: BibTeX Entries
Provide complete BibTeX entries for all cited papers:

```bibtex
@inproceedings{authorYear,
    title={Full Paper Title},
    author={Author Names},
    booktitle={Conference/Journal Name},
    url = {},
	doi = {},
	abstract = {xxx},
    year={YYYY},
    pages={XX--XX},
    organization={Publisher/Organization}
}
```

## Quality Criteria
1. **Relevance**: Papers must be directly relevant to the specific concept and the thesis context (drone/bird detection, 3D reconstruction)
2. **Accuracy**: Extracted sentences must accurately represent the paper's contribution
3. **Diversity**: Cover different aspects/approaches to the concept across multiple papers
4. **Recency**: Prefer recent work but include foundational papers where appropriate
5. **Clarity**: Sentences should be understandable without reading the full paper

## Search Strategy
- Use specific technical terms from the bulletpoint
- Combine with keywords: "drone detection", "UAV", "bird classification", "multi-view", "3D Gaussian", "sparse view reconstruction"
- Look for papers in top-tier venues: CVPR, ICCV, ECCV, SIGGRAPH, IEEE Transactions on Pattern Analysis and Machine Intelligence

## Example

**Input Bulletpoint:**
"Motion-based detection cues (optical flow, frame differencing) and their utility for detecting small moving objects that lack appearance details."

**Expected Output:**

```latex
\item Motion-based detection cues (optical flow, frame differencing) and their utility for detecting small moving objects that lack appearance details.
\begin{itemize}
    \item \cite{zhao2023} Optical flow estimation enables detection of micro-UAVs by capturing motion signatures even when visual features are insufficient due to distance.
    \item \cite{liu2022, chen2024} Frame differencing and temporal consistency checks effectively segment small aerial objects against complex sky backgrounds, improving robustness in surveillance scenarios.
\end{itemize}
```
