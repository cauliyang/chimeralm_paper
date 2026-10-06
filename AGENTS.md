# Task: ChimeraLM manuscript revision (Round 1, 3 reviewers, NAR)

 You are helping me revise the ChimeraLM manuscript (a genomic language model for
 detecting WGA chimeric artifacts in single-cell long-read sequencing) in response
 to the first round of peer review. Work carefully: accuracy and traceability
 matter more than speed.

 Environment

### Local machine (where you run)

- Manuscript:            /home/yangli/Projects/writing_projects/chimeralm_paper
- Point-by-point response:
   /home/yangli/Projects/writing_projects/chimeralm_paper/review/point-by-point-response-for-chimeralm
- Code (local copy):     /home/yangli/Projects/coding_project/ChimeraLM

### Quest HPC (ssh quest)

- Code:                  /gpfs/projects/b1171/ylk4626/project/Chimera
- Data:                  /gpfs/projects/b1171/ylk4626/project/Chimera/data
- Raw chimera BAM/FASTA/FASTQ: /gpfs/projects/b1171/ylk4626/project/Chimera/data/raw
- GPU node (already allocated, 2× A100): qgpu0517 — reach it with ssh quest then ssh qgpu0517
- Python env: use uv to run command and add deps for example uv run chimera

### HPC rules

 1. Never run training, inference, or heavy I/O on the Quest login node; only on qgpu0517.
 2. Run every long job inside tmux or with nohup ... > logs/<name>.log 2>&1 & so it
    survives SSH disconnects. Tell me the session name / log path.
 3. Run nvidia-smi before launching; use both GPUs when the job supports it.
 4. Do not delete or overwrite anything under data/raw or existing checkpoints.
    Write new outputs to revision/<experiment_name>/ with a timestamp.
 5. Keep local and HPC code in sync via git (<BRANCH, e.g. revision-r1>). Commit
    locally, push, then git pull on Quest. Never edit the same file on both sides.
 6. If the allocation is about to expire, warn me before starting a long job.

 Workflow — follow these phases in order

### Phase 1: Understand (read-only, no edits)

- Read the full manuscript, the reviewer comments, and any existing response draft.
- Skim the code structure (model, training, evaluation, figure scripts) locally and on HPC.
- Produce review/revision_tracker.md: a table with one row per reviewer comment:
   | ID (R1.1, R2.3, …) | Summary | Type (text / new analysis / new experiment / figure / clarification /
   disagree) | Effort (S/M/L) | GPU needed? | Proposed action | Status |
- Flag comments that overlap across reviewers so we address them once.
- STOP and wait for my approval of the tracker before doing anything else.

### Phase 2: Plan experiments

- For each comment needing new analysis or training, write a short plan: hypothesis,
   data, command, expected runtime, output location, and which figure/table it feeds.
- Prioritize: launch the longest GPU jobs first so they run while we do text edits.
- STOP and wait for my approval.

### Phase 3: Execute

- Launch approved jobs on qgpu0517; log the exact command, git commit hash, and
   config for each run in revision/RUNLOG.md.
- While jobs run, draft responses for text-only comments.
- When results come in, summarize them honestly, including results that don't
   support our claims. Never fabricate or round numbers favorably.

### Phase 4: Write

- Manuscript edits: make changes in LaTeX and mark every change
   (<e.g. \textcolor{blue}{...} or tracked changes>) so reviewers can find them.
- Response letter, for each comment:
   1. Quote the reviewer's comment verbatim.
   2. Thank them briefly (once per reviewer, not every point).
   3. Explain what we did, with concrete numbers.
   4. State exactly where the change is (section, page/line, figure, supplementary item).
   5. Quote the revised manuscript text.
- Tone: professional, respectful, confident; where we disagree, give evidence, not defensiveness.
- Keep numbers consistent between the manuscript, figures, and response letter.

 General rules

- Ask me when a reviewer comment is ambiguous or when an action could change a main conclusion.
- Do not invent citations; if you suggest a reference, give enough detail for me to verify it.
- Update revision_tracker.md status as you go.
- At the end of each working session, give me a short summary: done / running / blocked.

 Start with Phase 1.
