## Standard structure for each SUCCINCTLY WRITTEN component

- Target length of component: 2-3 pages TOPS!
- Your answer should be written succinct, human readable, English level B2 (professional) and directly useable
- don't use hyphens or em dash in your text

For every component in the pipeline, use:

> **3.k Component X: [Name]**
> 3.k.1 Role and problem definition
> 3.k.2 Taxonomy of approaches
> 3.k.3 Classical / pre-learning methods
> 3.k.4 Modern learning-based / SOTA methods
> 3.k.5 Evaluation metrics and benchmarks
> 3.k.6 Critical comparison & motivation for your design

---

### 3.k.1 Role and problem definition

**Goal:** Clearly state what the component does and why it is needed in the overall system.

Write:

* defining the problem the component solves in isolation.
* explaining how its inputs come from previous components and how its outputs are used by later components.
* narrowing to the specific variant of the problem you focus on (e.g. constraints, operating conditions, or special requirements).

Emphasise:

* How your setting differs from common assumptions in the literature (data characteristics, supervision, environment, constraints).
* What makes this component particularly challenging or critical in your pipeline.

---

### 3.k.2 Taxonomy of approaches

**Goal:** Organise prior work into a small, clear structure rather than listing papers chronologically.

Write:

* introducing 2–3 **axes** along which methods differ (for example: representation, supervision, use of domain knowledge, model architecture, etc.).
* defining a compact taxonomy of method groups based on these axes.
* per group describing:
  * The core idea of the group.
  * Typical strengths and weaknesses, without going into individual papers yet.

Emphasise:
* The main **competing paradigms** and how they contrast.
* For each paradigm, briefly state when it is usually preferred and where it tends to struggle.

---

### 3.k.3 Classical / pre-learning methods

**Goal:** Show that you understand the non-learning or early-learning methods and their relevance and limitations.

Write:

* summarize how the component was traditionally addressed before modern learning-based methods.
* representative classical approaches. For each:

  * Assumptions (about data, noise, prior knowledge, etc.).
  * Core algorithmic idea.
  * One main strength and one main weakness.

Compare methods along:

* Computational complexity and memory footprint.
* Robustness to noise, data scarcity, or adverse conditions relevant to your problem.
* Suitability for real-time or resource-constrained deployment, if applicable.

End with:

* answering the question: **Why are classical methods alone not sufficient for your problem setting?**

---

### 3.k.4 Modern learning-based / SOTA methods

**Goal:** Position your work within the current state of the art.

Write:

* introductory describing how modern learning-based methods changed the way this component is approached.
* For each group in your taxonomy (from 3.k.2):
  * summarising the typical approach of that group.
  * Then 3–5 key references (summarised in very short form):
    * method’s central idea.
    * on what is demonstrated empirically (type of data, main result).
    * limitations in relation to your problem constraints.

Emphasise:

* Trends in accuracy and robustness.
* Data and supervision requirements.
* Training and inference cost, model size, and runtime behaviour.
* Compatibility with your system assumptions (e.g. known/unknown parameters, number of views, online/offline setting).

Maintain **synthesis**: instead of listing methods, regularly summarise patterns:

* What these methods achieve collectively.
* Where they systematically fall short for your use case.

---

### 3.k.5 Evaluation metrics and benchmarks

**Goal:** Clarify how this component is usually evaluated and how you will evaluate it.

Write:

* describe the standard metrics used for this type of component (what they measure and why they are used).
* describe standard datasets or benchmarks (data modality, difficulty, scale, typical scenarios).
* explain the gap between these benchmarks and your setting (differences in data characteristics, constraints, or objectives).

Emphasise:

* If there are competing metrics, briefly explain trade-offs and why you prioritise certain metrics.
* Clearly state which metrics and datasets you will use in your own experiments and how that follows from the literature.

---

### 3.k.6 Critical comparison & motivation for your design

**Goal:** Turn the review into a concise argument that justifies your chosen design for this component.

Write:

1. **Summary table (strongly recommended)**
   A small table comparing 5–10 key methods or method groups across 4–6 criteria

2. **Textual synthesis**

   * Start with a short “bird’s-eye” summary paragraph describing how classical and modern approaches compare overall.
   * For each major design choice you make for this component (e.g. representation, supervision regime, type of architecture, inference strategy):

     * Define the alternatives.
     * Give bullet points or short sentences contrasting them (pros, cons, typical use, main failure modes).
     * Conclude with 1 sentence explicitly linking your choice to your problem constraints and objectives.

3. **Research gap and contribution hook**

   * 1 final paragraph stating:

     * The **gap**: what is missing or insufficient in existing approaches with respect to your specific problem setting.
     * The **promise of your design**: how your chosen approach for this component is intended to address that gap.

This subsection should directly prepare the reader for your methods chapter: after reading it, the reader should understand why your chosen design is a logical, well-justified choice given the existing literature and your constraints.

---

## Checklist per component (for a strong “9-level” section)

Before moving on, ensure:

* [ ] The component’s problem is clearly defined and tied to its role in the overall pipeline.
* [ ] Prior work is grouped using a clear, small taxonomy rather than listed chronologically.
* [ ] Both classical and modern methods are covered, with explicit pros/cons relative to your setting.
* [ ] Metrics and benchmarks are described and connected to your planned evaluation.
* [ ] A concise comparison table summarises key methods or method groups.
* [ ] The section ends with a clear statement of the research gap and a convincing motivation for your chosen design.
* [ ] Length is 2-3 pages maximum
* [ ] Writing is succinct, B2 English level, professional
* [ ] No hyphens or em dashes used in text
* [ ] Response is consists of 2 parts: 1) .tex text including inline references 2) .bib references