# Instructions for Expanding Background Bulletpoints from Provided Papers

## Task Overview
You are given a section from a background chapter with high-level concept bulletpoints, along with the Introduction and Problem Setting of the thesis for context. You are also provided with the text content of several academic papers (PDFs). Your task is to expand each bulletpoint by extracting concise sentences from the **provided papers** that relate to the concept. If multiple provided papers share similar findings or concepts, merge them into a single sentence with multiple citations.

## Input Format
You will receive:
- **Thesis Context**: The Introduction and Problem Setting chapters of the thesis.
- **Target Section**: A section heading and multiple bulletpoints describing high-level concepts to be expanded.
- **Source Papers**: The text content of several academic papers.

## Output Format Requirements

### Expanded LaTeX Content
For each bulletpoint, structure your output as follows:

```latex
\subsection{[Original Bulletpoint Concept]}
\begin{itemize}
    \item [High-level concept statement from original bulletpoint]
    \begin{itemize}
        \item \cite{AuthorYear} [ONE sentence extracted from this paper that relates to the concept in the context of the provided thesis background]
        \item \cite{AuthorYear1, AuthorYear2} [ONE merged sentence covering overlapping concepts from multiple papers]
    \end{itemize}
\end{itemize}
```

### Extraction Guidelines
- **Source Material**: Only use the papers provided in the prompt. Do not search for external papers.
- **Contextual Relevance**: Use the provided "Introduction" and "Problem Setting" to determine which statements in the papers are most relevant to the thesis theme (e.g., multi-camera drone/bird detection, 3D reconstruction).
- **Citation Format**: Use the citation key provided with the paper text, or generate a standard `AuthorYear` key based on the paper's metadata.
- **Conciseness**: Sentences should be concise (max 25 words) and self-contained.
- **Synthesis**: Merge overlapping self-contained sentences into one statement with multiple citations.

## Quality Criteria
1. **Relevance**: Extracted statements must be directly relevant to the specific bulletpoint concept and the thesis context.
2. **Accuracy**: Extracted sentences must accurately represent the paper's contribution.
3. **Clarity**: Sentences should be understandable without reading the full paper.

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
