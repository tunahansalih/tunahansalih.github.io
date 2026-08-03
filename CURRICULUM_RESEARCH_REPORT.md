# Evidence-Based Interview Curriculum Research Report

- **Candidate:** Tuna Han Salih Meral
- **Market target:** Spring 2027
- **Research cutoff:** 2 August 2026
- **Curriculum version:** `vision-generative-2026-08-rev12-coding-refresher`

## Executive conclusion

Research Scientist interviews differ by company and team. The current roles reviewed for this plan fall into five groups:

1. frontier generative image/video and world-model research;
2. research engineering, large-scale training, data, and evaluation;
3. applied/perception/3D science;
4. visual post-training, rewards, and evaluation;
5. coding-heavy generalist ML.

The curriculum covers:

- an **always-on core**: ML coding, debugging, general coding, math, research defense, experiment design, systems, behavioral stories, and mocks;
- a **main technical area**: controllable image/video generation and world models;
- a **secondary area**: research engineering, JAX/TPU, data, evaluation, and visual post-training;
- a **short 3D unit** based on the Waymo work;
- five **72-hour review plans** chosen after the recruiter or job description identifies the interview topics.

Topics outside the job description remain outside the weekly workload. This keeps the backlog bounded.

## Candidate-to-market fit

The CV documents work in:

- controllable and interpretable image/video generation;
- diffusion, flow matching, DiT, autoregressive visual generation, personalization, and distillation;
- JAX/Flax, TPU training, and distributed ML;
- camera-based 3D detection and the Waymo Open Dataset;
- large-scale production ML, including 5M+ daily inference requests;
- research publication defense across seven exposed papers.

A concise description of that background is:

> Research Scientist / Research Engineer working on controllable generative image and video models, with experience in JAX/TPU training, production systems, and 3D perception.

The RL unit covers visual-model post-training and world models. The weekly plan does not include a full classical-control or robotics syllabus because the CV and target roles do not support one.

## Startup and neo-lab target expansion

The company tracker includes 16 startups and independent labs alongside large employers. The groups reflect the current match between each role and the CV.

**Apply first:** Black Forest Labs, fal, World Labs, Luma AI, Runway, Pika, Genmo, and Moonvalley. Black Forest Labs works on flow-based visual models, image/video generation, pretraining, post-training, VLMs, and training infrastructure. fal hires across research science, applied ML, inference performance, and distributed systems; the Lyrebird work supplies relevant production evidence.

**Second group:** Midjourney, Krea, Higgsfield, and SpAItial. Check the role and location before spending preparation time.

**World-model and robotics labs:** Odyssey, Pantograph, Mirage, and Basis Research Institute. Apply when the role asks for video/world modeling, visual evaluation, JAX-scale training, or post-training. State the robotics gap directly.

Keep no more than five Dream/High targets, two practice companies, and one watch relationship active at once. Company-specific review begins after a recruiter or job description identifies the interview topics.

Primary verification pages include [Black Forest Labs careers](https://job-boards.greenhouse.io/blackforestlabs), [fal careers](https://fal.ai/careers/4f910e21-48de-48b9-bfde-aa7f474f73c1), [Moonvalley careers](https://www.moonvalley.com/careers), [Midjourney careers](https://www.midjourney.com/careers), [Odyssey world-model research](https://jobs.ashbyhq.com/odysseyml/18ffb449-dcc4-42b2-a547-fb021fc1177b), and [Pantograph research](https://jobs.ashbyhq.com/pantograph/dfe5cf92-33f3-4ab7-8dc8-f8aa75693ac7). Hiring status is dated 1 August 2026 and must be rechecked before applying.

## Evidence standard

| Grade | Evidence type | How it is used |
|---|---|---|
| A | Official employer interview guide or current role description | Interview formats and current work expectations |
| B | Named, dated firsthand interview account | Recurring interview machinery and preparation lessons |
| C | Anonymous firsthand or high-quality secondary account | Corroboration only |
| I | Inference from current role duties or the candidate's CV | Domain weighting; never presented as question frequency |

The report does **not** assign invented probabilities such as “40% diffusion” or “20% RL.” No credible public source supports such precision for generative-vision interviews.

## What official interview guides establish

### OpenAI

The [OpenAI Interview Guide](https://openai.com/interview-guide/) describes skills-based assessments that may include pair coding, take-homes, and technical tests, followed by a final stage commonly lasting four to six hours with four to six people. Engineering evaluation includes well-designed solutions, high-quality code, performance, tests, communication, and collaboration.

**Curriculum consequence:** repeated ML coding, general coding, debugging, system design, testing, narration, and full-loop stamina remain core.

### Anthropic

[Anthropic Careers](https://www.anthropic.com/careers) states that technical interviews may use Colab or CodeSignal and permits documentation lookup while expecting fluency with basic syntax and standard libraries. It also explicitly blurs the researcher/engineer boundary.

**Curriculum consequence:** Python/PyTorch implementation, debugging, and research-engineering judgment are not optional for a Research Scientist candidate.

### Amazon

The [Amazon Applied Scientist interview guide](https://amazon.jobs/content/en/how-we-hire/applied-scientist-interview-prep) describes one or two technical phone screens and a loop of four 55-minute interviews covering technical topics, problem solving and coding, depth/breadth, a tech talk, and repeated behavioral questions.

**Curriculum consequence:** project defense, coding, technical breadth, job-talk preparation, and a rehearsed story bank all need measured gates.

### NVIDIA

[NVIDIA's hiring guide](https://www.nvidia.com/en-us/about-nvidia/careers/how-we-hire/) describes several 30–60 minute interviews and possible coding exercises using HackerRank, a whiteboard, or a laptop.

**Curriculum consequence:** general coding cannot vanish after an early gate, and practice must tolerate different tools.

### Google DeepMind

[Google DeepMind Careers](https://deepmind.google/careers/) distinguishes Research Scientists, who formulate hypotheses and conduct novel research, from Research Engineers, who combine engineering, mathematics, implementation, optimization, and distributed infrastructure. Its public process includes recruiter/hiring-manager stages, skills interviews, and a final stage.

**Curriculum consequence:** the candidate should keep one research narrative but prepare both research-depth and research-engineering evidence.

## What current roles establish

Current descriptions are evidence of work expectations, not exact interview questions.

### Generative image/video and world models

- [World Labs: Generative Modeling](https://job-boards.greenhouse.io/worldlabs/jobs/4089324009) spans diffusion image/video/3D, large-scale training, pre/post-training data curation, tokenizers/VAEs, long context, 3D vision, controls, preference, distillation, evaluation, and deployment.
- [Luma careers](https://lumalabs.ai/careers) includes world-model, foundation-model, training-infrastructure, RL-infrastructure, controllability/personalization, and evaluation work.
- [Runway careers](https://runway.com/careers) includes research and engineering roles around video and world models.
- [Pika: Research Scientist](https://jobs.ashbyhq.com/pika/0228ca18-c0e9-4a3b-87bf-6be2f3f7efb0) names controllable image/video, multimodal models, RLHF/DPO, inference efficiency, data pipelines, and distributed training.
- [Mirage: Research Scientist](https://jobs.ashbyhq.com/mirage/06fe4d02-2b36-40b0-94a3-8f816e8826c1) lists temporal consistency, controllability, multimodal alignment, scaling, evaluation, and real-time efficiency.

**Curriculum consequence:** the core vision sequence must include latent representations/tokenizers, DiT, diffusion/flow, autoregressive alternatives, video temporal modeling, controllability, evaluation, long-horizon design, and candidate-specific paper defense.

### Data and evaluation

- [Runway: Data Foundations](https://jobs.ashbyhq.com/runway-ml/f74afbd9-b4e8-4b87-835a-5e22dcc556ae) covers multimodal data design, data composition experiments, synthetic pipelines, filtering, quality control, evaluation, and PyTorch/JAX infrastructure.
- World Labs also names data curation before and after training.
- [Anthropic: Visual Knowledge Work](https://job-boards.greenhouse.io/anthropic/jobs/5074217008) names evaluation environments, rewards, data quality, reward hacking, and held-out generalization for visual tasks.

**Curriculum consequence:** “experiment design” cannot mean only architecture ablations. The revised task independently controls method, data mixture, compute, leakage, quality slices, and human evaluation.

### Visual post-training and RL

- [Genmo: Video Post-Training](https://jobs.ashbyhq.com/genmo/f09286ef-c08d-49c4-8635-c1d952440865) directly covers SFT/RLHF and alignment for large video-diffusion models.
- Pika names RLHF/DPO for controllable image/video work.
- Anthropic's visual role names rewards, evaluation, reward hacking, and held-out generalization.
- [Anthropic: Production Model Post-Training](https://job-boards.greenhouse.io/anthropic/jobs/4613592008) lists post-training pipelines, evaluation, debugging, distributed systems, reliability, and reproducibility.

**Curriculum consequence:** tabular Q-learning is no longer a default task. The core now includes Diffusion-DPO, VADER, policy gradients, PPO/GRPO/DPO comparisons, reward/control diagrams, and held-out reward-hacking tests. Classical environment RL remains only a role-triggered expansion.

### Perception and modern 3D

- The candidate has direct Waymo camera-based 3D detection evidence.
- Current NVIDIA roles span 3D vision, neural rendering, generative image/video, and world reconstruction.
- [World Labs: 3D Reconstruction](https://job-boards.greenhouse.io/worldlabs/jobs/4113005009) expects deep specialization in modern representations, multiview geometry, rendering, and performance.

**Curriculum consequence:** detection, camera geometry, AP/APH, and Waymo defense stay in the core. 3D Gaussian Splatting and DUSt3R are a bounded activation-pack comparison, not a default specialist curriculum.

## What firsthand evidence establishes

### High-confidence named account

[Alisa Liu's 2026 account](https://alisawuffles.github.io/blog/job-search/) is the strongest recent, named, detailed account found. Across a large interview search, she reports:

- ML coding as the most common technical form;
- PyTorch fluency and implementations from scratch;
- general algorithmic coding;
- experiment-design and rapid-fire technical discussions;
- research discussions on a candidate's work and papers;
- behavioral interviews that punish improvisation;
- mathematical derivations involving probability, linear algebra, and calculus;
- transformer implementation/debugging practice;
- concentrated company-specific preparation;
- the cost of sleep deprivation and poor post-round recovery.

Its domain is LLM/NLP, so it is used only for common interview machinery. Vision-domain content comes from current roles and the candidate's CV.

[Silvia Sapora's ML interview notes](https://silviasapora.github.io/blog/ml-interviews.html) corroborate the variability of topic breadth, the need for company-specific preparation, and personal flashcards.

### Lower-confidence corroboration

Anonymous Meta and OpenAI candidate reports corroborate general coding, ML/system design, behavioral evaluation, debugging, statistics, and transformer breadth. They are not used to set curriculum weights because role, level, and process variants are difficult to verify.

### Important negative finding

The search found no strong, dated, detailed firsthand account that establishes the frequency of diffusion, video-generation, or 3D questions in Research Scientist interviews. A 2026 discussion asking specifically for diffusion-interview experiences did not provide the missing evidence.

**Consequence:** the curriculum labels generative-vision modules as a reasoned role/CV inference, not a known interview-frequency fact.

## Learning-sequence rules

Every task was reviewed against these rules:

1. no timed cold test before assigned review and supported practice;
2. one primary artifact per session;
3. exact source or named problem;
4. explicit timing mode: flexible learning window or strict simulation limit;
5. observable completion criterion;
6. low-capacity fallback that still moves the skill forward without requiring a successful full block;
7. prerequisite stated when the task depends on earlier material;
8. gates measure something the curriculum actually introduced;
9. recurring work appears in the tracker;
10. failed work is classified and repaired, not accumulated.

## General-coding program: Grind 75 integration

The previous plan did not contain enough deliberate general-coding practice. It assigned an early two-Medium gate, followed by one problem per week in topic blocks. That arrangement measured prior LeetCode exposure and allowed long gaps between related patterns.

The replacement uses the current [Grind 75 question data](https://www.techinterviewhandbook.org/grind75/) and the author's [ordering and timing guidance](https://www.techinterviewhandbook.org/grind75/faq):

- Start from the 14-active-week, 2-hours-per-week, Easy-and-Medium, all-topic, all-rounded filter.
- Omit Two Sum from required work. For this candidate it is an editor-fluency check, not useful interview preparation.
- Start with a 30-minute, four-cue recognition refresher covering arrays and strings, hash maps, stacks, and linked lists. It produces one retrieval sheet and contains no problem solving.
- Keep filtered questions 2–34 as the ordered Core sequence: 17 Easy and 16 Medium across 12 topics.
- The source estimate is about 24 hours including solution review and logging; the rounded task reservations total 24.5 hours. Weekly coding reservations range from about 1h20 to 2h10.
- Leave Weeks 8 and 12 empty instead of pulling future problems into recovery time.
- Keep the next three filtered problems—Rotting Oranges, Search in Rotated Sorted Array, and Combination Sum—as replacement-only repair problems.
- Run the first two-problem Medium checkpoint on questions 20 and 21, after 19 preparations. A failed checkpoint replaces one later problem; it never adds work to the week.
- Keep Hard problems outside Core until two consecutive two-Medium simulations pass.

Each problem has one countdown for the attempt and one larger calendar reservation for the full block. The problem remains complete even when the attempt is unfinished, provided the approach, elapsed time, failure point, and review are logged. This prevents a difficult problem from becoming an open-ended session.

## Whole-plan audit

### Week 0: Supported preparation

**Verdict:** dependency-correct. The first timed assessment comes after guided preparation.

- General coding begins with the four-cue recognition sheet. Valid Parentheses and Merge Two Sorted Lists then retrieve stack and pointer ideas that were activated first. Two Sum is not required or counted.
- ML coding uses a guided attention adapter and objective tests.
- Math uses explicit worked examples and a later fixed gate.
- The CV review comes before paper-defense sampling.

### Week 1: Narrative and supported repair

**Verdict:** coherent.

- The repair task branches from the actual Week 0 blocker.
- Three more Easy problems introduce array, string, and tree patterns without a timer-based pass criterion.
- Three paper cards exist before later random defense.
- The story bank begins before behavioral gates.
- Mock partners, references, and outreach are started early enough to have external latency.

### Week 2: Supported coding and first ML/math gates

**Verdict:** corrected and measurable.

- Valid Anagram, Binary Search, and Flood Fill complete the first nine coding preparations. G1 does not run yet.
- G2 follows guided attention work and repair.
- Shape/dtype/device debugging precedes later planted-bug mocks.
- The paper defense occurs only after source review.

### Week 3: Debugging, JAX, vision retrieval, first mock

**Verdict:** coherent.

- Maximum Subarray is the first Medium, after nine Easy problems; Lowest Common Ancestor revisits trees with a different invariant.
- The first debugging drill is supported and hypothesis-first.
- JAX study is introduced before its later gate.
- Vision fundamentals are tested only after a bounded review.
- The first ML-coding mock occurs early enough to change the plan and has a fixed prompt/check sheet for solo use.

### Week 4: Detection, 3D, JAX sharding, Waymo defense

**Verdict:** coherent and candidate-specific.

- IoU/NMS has an objective oracle.
- Camera geometry stays deliberately bounded.
- JAX sharding builds on Week 3.
- The architecture/evaluation tasks lead into the Waymo defense.
- Insert Interval and 01 Matrix continue the mixed-topic sequence within a combined reservation of about 1h50.

### Week 5: Diffusion, flow, controls, 3D evaluation

**Verdict:** coherent.

- DDPM/CFG precede implementation and controlled-generation defense.
- Flow matching is studied and derived before the broader comparison.
- Detection experiment design follows the Week 4 architecture.
- K Closest Points, Balanced Binary Tree, and Linked List Cycle mix heap, tree, and linked-list patterns.

### Week 6: Modern visual stack, video, evaluation, debugging

**Verdict:** materially improved.

- The revised architecture task now covers latent diffusion, visual tokenizers/VAEs, DiT, raster AR, and next-scale AR.
- Video factorization follows diffusion foundations.
- VBench-style evaluation precedes long-horizon stress tests.
- The second debugging mock is separated from the design block.
- Queue using Stacks, First Bad Version, and Ransom Note revisit stack, binary-search, and hash-table patterns.

### Week 7: Systems, scale, JAX/TPU

**Verdict:** coherent and strongly CV-grounded.

- Parameter/FLOP/memory arithmetic precedes parallelism decisions.
- Training and inference design are separately gated.
- JAX/TPU has its own measured gate.
- Longest Substring Without Repeating Characters and 3Sum form G1 only after 19 earlier preparations. Each has a 30-minute attempt limit and its own review block.

### Week 8: Slack and adaptation

**Verdict:** preserved.

- No new coding or content task is generated.
- Recovery, gate review, scope cuts, and outreach remain the only obligations.

### Week 9: Visual post-training and RL

**Verdict:** rebuilt around role evidence.

- MDP/Bellman concepts precede policy gradients.
- Visual preference/reward mapping is introduced open-note before formal PPO/GRPO/DPO comparison.
- Diffusion-DPO and VADER replace default tabular Q-learning.
- REINFORCE and GAE/PPO implementations remain bounded.
- G7 now includes a falsifiable held-out reward-hacking test.
- Binary Tree Level Order Traversal and Clone Graph restart the sequence after the recovery week.

### Week 10: Breadth, data/evaluation design, mocks

**Verdict:** improved.

- LLM breadth stays compact because current frontier labs may test it. It is not the main technical area.
- Experiment design now separates method, data mixture, compute, leakage, metrics, seeds, slices, and human evaluation.
- Rapid-fire breadth uses fixed prompts and answer cues.
- Experiment-design and ML-coding mocks are separate; the ML mock has a fixed sealed prompt and separate check sheet.
- Evaluate Reverse Polish Notation and Course Schedule provide a two-Medium retest pair if G1 failed.

### Week 11: Research agenda and random defense

**Verdict:** coherent.

- Remaining paper cards are created before random selection.
- The research agenda unifies the CV instead of adding a new topic.
- Random defense has a fixed question bank.
- Implement Trie and Coin Change add trie and dynamic-programming practice without increasing the weekly count after a miss.

### Week 12: Slack

**Verdict:** preserved.

- No new content or generated coding task.
- Retrieval, stories, and adaptation only.

### Week 13: Job talk and responsible research

**Verdict:** coherent.

- Slides reuse existing figures and assign one remembered sentence plus one piece of evidence per slide.
- Responsible-research work is grounded in the candidate's actual projects.
- Company variants remain deferred until a real role exists.
- Product of Array Except Self and Climbing Stairs revisit array and dynamic-programming patterns.

### Week 14: Behavioral and full-loop simulation

**Verdict:** coherent.

- Behavioral preparation precedes the onsite simulation.
- The full loop combines research, ML coding, design, and behavioral performance. A second fixed prompt makes the ML round possible without a partner.
- Longest Palindrome, Min Stack, and Reverse Linked List revisit string, stack, and linked-list patterns.
- Company-specific content is activated only by a scheduled interview.

### Week 15: Launch and stamina

**Verdict:** intentionally exceptional workload.

- Validate Binary Search Tree and Number of Islands close the 33-problem Core sequence before live interviewing.
- Application launch, offer-model preparation, negotiation criteria, stamina, and logistics are completed before live interviewing.
- The workload exception is explicit and should not be treated as the weekly norm.

### Live phase

**Verdict:** coherent.

- Weekly outreach, retrieval, miss repair, and post-round debriefs reset each week.
- The 72-hour protocol suspends the generic curriculum.
- Five exact role packs prevent company-specific preparation from becoming an unbounded reading list.

## Curriculum changes made by this research

1. Added a visible Evidence tab with technical areas, source grades, conclusions, limitations, and source links.
2. Replaced the improvised coding queue with a 33-problem Grind 75 sequence selected from a 14-week, 2-hour, Easy-and-Medium, all-topic filter. Two Sum is excluded from required work.
3. Moved G1 from Week 2 to Week 7, after 19 preparations, and defined a fixed W10 retest pair.
4. Added replacement-only repair problems so remediation changes the sequence without adding backlog.
5. Added latent diffusion, visual tokenizer/VAE, and DiT coverage to the modern visual architecture sequence.
6. Replaced default tabular Q-learning with Diffusion-DPO, VADER, rewards, preference optimization, and reward-hacking evaluation.
7. Expanded experiment design to separate method effects from data-mixture effects and leakage.
8. Added a short 3DGS/DUSt3R comparison to the perception/3D review plan.
9. Replaced broad company cramming with five role-specific review plans.
10. Added two fixed ML-coding mock prompts with separate check sheets so solo simulation is executable.
11. Updated gates, prerequisites, completion criteria, fallbacks, workload, and progress migration.
12. Replaced fixed learning deadlines with ADHD-calibrated work windows, calendar reserves, a count-up focus clock, and saved actual duration.
13. Replaced the cold Two Sum start with a bounded pattern-recognition refresher; Two Sum is now only an optional editor check.

## Timing reassessment: 31 July 2026

The original timing model was too optimistic and too rigid. It treated a nominal 45-minute content estimate as both a calendar reservation and a performance target. That estimate omitted start-up time, opening materials, attention transitions, breaks, debugging, recording, and closure. Multi-paper synthesis, JAX setup, implementation, systems arithmetic, and mathematical derivation can each require 60–90 minutes of work.

The revised model separates **content work**, **calendar reservation**, and **strict interview timing**:

| Profile | Expected content work | Calendar reservation | Clock rule |
|---|---:|---:|---|
| Brief | 15–25 minutes | 30 minutes | Count up; finish the named small artifact |
| Flexible | 35–50 minutes | 60 minutes | Two short focus cycles with a visible break |
| Deep | 50–70 minutes | 80 minutes | Two or three cycles; stop at a safe checkpoint |
| Simulation | Prompt limit plus review | 75 minutes | Countdown only for the interview prompt; count up for the whole block |
| Stamina simulation | Four hours of rounds/breaks | 5 hours 30 minutes | Run only when rested |

After the Grind 75 integration, a normal full week reserves approximately 9h10–12h05 on the calendar. Slack weeks still reserve 1h30, and Week 15 reserves about 13h because it combines the final coding problems with the full-loop simulation. The default across Weeks 0–15 is about 154 calendar hours. This is deliberately larger than the content total and should be replaced by the candidate's observed median after several tasks.

| Week | Default calendar reserve |
|---|---:|
| W0 | 5h30 |
| W1 | 10h15 |
| W2 | 9h10 |
| W3 | 9h30 |
| W4 | 11h25 |
| W5 | 11h45 |
| W6 | 11h40 |
| W7 | 12h05 |
| W8 | 1h30 |
| W9 | 11h45 |
| W10 | 11h45 |
| W11 | 10h50 |
| W12 | 1h30 |
| W13 | 10h50 |
| W14 | 11h15 |
| W15 | 13h00 |

The tracker now:

- hides minute-by-minute splits for flexible learning and presents the steps in order;
- displays one calendar reservation on each task and keeps the detailed timing model in a collapsed explanation;
- strips obsolete duration labels from session headings;
- provides a count-up clock for study sessions;
- saves actual focus duration per task;
- checkpoints elapsed time locally every 15 seconds, so opening a source in another tab does not stop the clock;
- shows weekly calendar reserve and a running median after three recorded tasks;
- keeps strict countdown language only for gates, mocks, and full-loop simulations.
- gives the Coding page a visible 14-week sequence, weekly problem checkboxes, direct problem links, exact attempt limits, and a pre-filled result log.

## ADHD-oriented execution design

The implementation follows [NICE guidance on environmental modifications and daily structure](https://www.nice.org.uk/guidance/ng87/chapter/recommendations), [NHS guidance on written instructions and help structuring tasks](https://www.nhs.uk/conditions/adhd-adults/), and [W3C cognitive-accessibility guidance](https://www.w3.org/WAI/WCAG2/supplemental/objectives/o5-user-focus/):

- written step-by-step tasks;
- one visible action at a time;
- external count-up time cues for learning and strict countdowns only for simulations;
- flexible learning windows plus a larger calendar reservation;
- automatic recording of actual duration for later calibration;
- exact links and problem names;
- a low-capacity fallback for initiation failure;
- visible completion criteria;
- no hidden recurring obligations;
- no catch-up backlog;
- slack weeks and explicit scope cuts.

These are interface and workflow choices, not medical treatment.

## Human-use review: 2 August 2026

The curriculum data was coherent, but the route to starting a task was too long. The dashboard placed countdowns, percentages, checkpoints and review-card reminders before the next action. Twelve primary navigation items competed for attention. Task cards showed two different time estimates, several controls and technical metadata at once. The coding schedule opened a separate Markdown instruction sheet before sending the candidate to the real exercise.

The revised interface now:

- places the next task before progress statistics and reminders;
- moves secondary tools into one More page;
- shows one calendar reservation on the task card;
- places the finish line and smaller starting version next to the action;
- keeps skip controls, rationale, prerequisites and saved notes under Options and context;
- puts each Grind 75 problem directly in its own task and keeps the ordered schedule visible on the Coding page;
- uses `plan.json` as the curriculum source while Firebase stores progress only;
- stops displaying weekly work while the interview-specific protocol is active.

The full curriculum remains demanding. Its calendar reservations include breaks and transition time, so they should not be compressed to make the plan look easier. If completion falls below 70% for two weeks, the existing scope-cut rule should remove later tasks. The plan should not create catch-up days or borrow from sleep.

## Remaining uncertainties and update policy

This curriculum should be treated as a versioned preparation system, not a permanent syllabus.

Update it when:

- a recruiter provides an actual round format;
- a mock or real interview produces a repeated miss;
- a target role materially changes;
- an official employer guide changes;
- the candidate's weekly completion stays below 70% for two weeks.

Do not update it because a new paper is fashionable. A new topic enters only if it is supported by a target role, attacks a CV claim, or repairs an observed miss.
