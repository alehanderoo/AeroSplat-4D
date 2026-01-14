# Instructions for Improving Background & Related Work Sections

## Purpose
These instructions guide the expansion of a thesis section with appropriate breadth (coverage of relevant work) and depth (technical detail and critical analysis).

---

## Pre-Writing Requirements

### 1. Context Review
Before writing, read and internalize:
- **Thesis Introduction & Problem Setting**: Understand the research objectives, hypotheses, and contributions
- **Full Background & Related Work Chapter**: Ensure narrative coherence with preceding and following sections
- **Current Section Draft**: Identify gaps, shallow treatments, and missing connections

### 2. Materials Provided
- Summary of thesis introduction and problem setting
- Current background and related work chapter
- Section to improve
- XML topic diagram (non-exhaustive guide to coverage)

---

## Writing Guidelines

### A. Structure & Flow

1. **Opening**: Connect to the previous section's conclusion; establish why this topic matters for the thesis
2. **Body**: Progress logically—typically chronological, simple→complex, or general→specific
3. **Closing**: Bridge to the next section; highlight remaining gaps that motivate the thesis

### B. Breadth: Topic Coverage

1. Consult the XML diagram for required topics; treat it as a minimum, not exhaustive list
2. For each topic, cover:
   - Foundational/seminal works
   - Key methodological developments
   - State-of-the-art approaches
   - Relevant benchmarks and datasets

3. **Citation Consolidation**: When multiple papers share findings or concepts, merge into a single sentence:
   > *"Several works have demonstrated that [concept] improves performance [1, 2, 3]."*  
   > rather than separate sentences per paper.

### C. Depth: Technical Detail

1. **Mathematical Foundations**: Include key equations when they:
   - Define a concept central to the thesis
   - Are referenced later in methodology
   - Illustrate a fundamental principle

   Format: Introduce the equation, present it, then explain notation and significance.

2. **Critical Analysis**: Don't merely describe methods—evaluate them:
   - Strengths and limitations
   - Assumptions and failure cases
   - Computational/data requirements
   - How limitations connect to thesis contributions

3. **Comparative Discussion**: When appropriate, contrast approaches:
   - Performance trade-offs
   - Applicability to the thesis problem domain
   - Historical evolution showing why newer methods emerged

### D. Coherence & Narrative

1. Maintain consistent terminology with other sections
2. Cross-reference related concepts discussed elsewhere in the chapter:
   > *"As discussed in Section 2.2, neural radiance fields provide..."*
3. Use transitional sentences between subsections
4. Ensure each paragraph has a clear purpose advancing the narrative

---

## Quality Checklist

Before finalizing, verify:

- [ ] All XML diagram topics addressed (at minimum)
- [ ] Smooth transitions from previous section and to next section
- [ ] Similar findings consolidated with grouped citations
- [ ] Foundational equations included where appropriate
- [ ] Critical analysis present (not just descriptive summaries)
- [ ] Clear connection to thesis problem established
- [ ] Consistent terminology with rest of chapter
- [ ] SOTA methods and recent developments included
- [ ] Limitations of existing work highlighted (supporting Research Gap)

---

## Formatting Notes

- Use consistent citation style matching thesis requirements
- Number equations if referenced elsewhere
- Maintain figure/table numbering scheme of the chapter
- Subsection hierarchy should align with chapter structure

---

## Section-Specific Considerations

| Section | Key Depth Areas |
|---------|-----------------|
| Detection & Tracking | Multi-view geometry, association algorithms, occlusion handling |
| 3D Reconstruction | Representation trade-offs (explicit vs. implicit), optimization objectives |
| Point Cloud Classification | Permutation invariance, local/global feature extraction |
| Temporal Modeling | Sequence encoding strategies, attention complexity |
| SE(3) Equivariance | Group theory foundations, equivariant layer construction |

---

*These instructions assume familiarity with computer vision, 3D reconstruction, and neural classification literature. Focus effort on critical synthesis rather than basic concept explanation.*

---

## Writing Style

- **Succinct**: Eliminate unnecessary words; prefer direct statements
- **B2 English level**: Clear, professional academic writing without overly complex constructions
- **Professional tone**: Objective, analytical, appropriate for a Master's thesis

---

## Citation Format

Citations in text use: `\cite{authorYear}`

### BibTeX Entries
Provide complete BibTeX entries for all cited papers at the end of the section:

```bibtex
@inproceedings{authorYear,
    title={Full Paper Title},
    author={Author Names},
    booktitle={Conference/Journal Name},
    year={YYYY},
    url={},
    doi={},
}
```

Ensure:
- Consistent `authorYear` key format (e.g., `smith2023`, `jonesEtAl2022`)
- Complete author lists
- DOI included when available
