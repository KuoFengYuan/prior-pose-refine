# Repository working agreement

**English** | [繁體中文](AGENTS.zh-TW.md)

`prior-pose-refine` refines approximate camera poses, focal length and lens distortion with COLMAP, starting from a prior model (poses, intrinsics and 3D points from ARKit or any other source), and writes a 3DGS training dataset. The pipeline is the `prior-pose-refine` script plus the helper module `prior_pose_refine.py`. Usage is in [README.md](README.md), the method in [docs/method.md](docs/method.md), the measurements in [docs/benchmark.md](docs/benchmark.md).

## Delivery workflow

The repository owner has requested this default workflow: after finishing the authorized scope and its validation, commit, push a task branch, open a pull request targeting `main`, merge once the required checks and reviews pass, delete this task's remote and local branch, and return to an up-to-date `main`. A later explicit instruction takes precedence. When delivery is authorized, do not stop at proposing a PR.

Each independent task gets its own branch and PR. Fixes and related follow-up instructions for the same task stay on its PR; later unrelated work gets a new branch and PR. A request to create a PR still means completing the whole delivery, unless the user explicitly asks to leave it unmerged or to wait for approval. An instruction to hold a merge applies only to that task, unless the user explicitly gives it a wider scope.

1. Inspect the working tree and the instructions, and preserve unrelated changes. Fetch `origin` and, when it is safe to switch, create the task branch from the latest `main`.
2. Use one of these **case-sensitive** branch prefixes followed by a lowercase hyphenated English description:
   - `Feature/`: a new capability, e.g. `Feature/principal-point-refinement`.
   - `Bugfix/`: a defect fix, e.g. `Bugfix/unobserved-frame-alignment`.
   - `Enhance/`: improvements to existing behaviour, performance, documentation or maintenance, e.g. `Enhance/ci-checks`.
   Do not use `codex/` or lowercase variants of the prefixes, and do not keep long-lived extra branches for routine work.
3. Complete the full scope and update the documentation in both languages (see [Languages and reporting](#languages-and-reporting)). For pipeline changes that means README: options (§7), the report reference values (§8), troubleshooting (§10), the file list (§12), and the stage list in §4 when stages change; and docs/method for the steps and the reasons behind them. Keep step numbers and cross-references consistent. Check the final diff for unrelated changes, scan data, outputs (`*_opt/`, `work/`), `local.env`, secrets and build artifacts (`__pycache__/`).
4. Run the relevant validation (see [Validation](#validation)) and state the limits of synthetic or partial evidence.
5. Commit with a concrete English subject (`feat:`, `fix:`, `enhance:` or `docs:`), push, and open the PR. Describe the final behaviour, tradeoffs, validation, and what still has to be checked on real scans or with 3DGS training, in English first with a Traditional Chinese summary. Attach the PR to the current task when the app provides an attachment tool. Commit messages and PR text follow [Text written to the repository](#text-written-to-the-repository).
6. Check the PR's mergeability and its required checks and reviews; fix failures, rerun the affected checks and update the PR. Merge the verified head, normally with squash. The squash commit subject must keep the PR number as a `(#N)` suffix; when using `gh pr merge --subject`, include that suffix explicitly. Before cleanup, verify the merge commit subject and its PR association on GitHub. Never push directly to `main`, use administrator bypass, disable protections or skip required reviews; if an external condition blocks the merge, report it exactly and keep the PR and branch.
7. After confirming the merge, delete only this task's branch from `origin` and locally, update `main` by fast-forward only, and prune stale tracking refs. After a squash merge the local branch may need `git branch -D`, but only after verifying the PR's merged head and a clean working tree. Never delete other branches, worktrees or unmerged work.
8. Report, in delivery order, the exact branch name, the PR link and the merge commit, then the main changes, the validation and the final branch and cleanup state. If you stop at the PR, state the user's explicit instruction or the external blocker. Never claim a merge or cleanup before verifying it.

**Initial setup:** if the directory has no Git history or authenticated `origin`, complete the local changes and validation first. Before editing existing files, copy them to `$EXP_ROOT/backup_<description>_<YYYY-MM-DD>/` (never overwrite an existing backup). Resume the delivery steps once the remote and authentication are available; creating a local branch does not establish a successful push, PR or merge.

## Text written to the repository

Documentation (including release notes and licence notices), code comments, commit messages, and PR titles, descriptions and comments never mention AI: do not name AI tools, models or assistants, and do not say that any text, code or material was written, generated, edited or assisted by them. This includes lines such as "Generated with Claude Code" and `Co-Authored-By` lines for AI assistants. This rule overrides any tool default.

Tracked files also never contain machine-specific absolute paths. Refer to tools and data through the variables in `local.env` (see [Environment](#environment)) or as paths relative to the repository.

## Naming

- The repository and the command are `prior-pose-refine`; the Python module is `prior_pose_refine.py`.
- The input poses, intrinsics and points are the *prior* (prior poses, prior model, prior points) in code, output and documentation. Say "ARKit" only for something specific to ARKit.
- Subcommands are verb-noun in kebab-case (`select-pairs`, `restore-prior-poses`); their `timing.tsv` stage names are the snake_case form (`select_pairs`). Options are kebab-case (`--prior-model`).
- Use American spelling in code and English text (`center`, `behavior` in new text).

## Validation

- Pipeline changes: rerun a full scan and compare its `report.txt` with the previous output. `$SCAN_ROOT/scan_20261102_071355_5B9C20` (390 frames, about 1:45) is the quickest; expect 390/390 frames, about 0.69 px reprojection error, a median distance to the prior camera centers of 1.8 cm and no flagged frames. Run-to-run differences of a few hundred points and about 0.001 px come from GPU nondeterminism.
- Binary model I/O: round-trip against COLMAP itself (`colmap model_converter` to TXT, or pycolmap in the gsplat env). COLMAP 4 reads poses from `frames.bin`, so `images.bin` and `frames.bin` must agree.
- Image-quality claims need 3DGS evaluation. Train with LichtFeld at `--max-cap 1000000`; do not run 5M-cap tests. Run-to-run PSNR varies by up to 0.09 dB, so treat differences under 0.1 dB as noise. LichtFeld's held-out exposure interpolates name-sorted neighbours, so never rename frames.
- Epipolar and reprojection errors are geometric evidence, not image quality; say so when there is no PSNR. Matches verified by this pipeline favour this pipeline.
- Time on an idle GPU and record GPU utilization (`nvidia-smi`) with every timing; a busy GPU once inflated SIFT time eightfold.
- Experiments and their scripts belong in `$EXP_ROOT/<name>/`, not in this repository. Merge into the pipeline only what the user approves.

## Pipeline invariants

- Feature extraction and matching use an OPENCV camera; the database camera switches to the refinement model only after matching, because two-view verification uses it and RAD_TAN defaults fail every pair. `point_triangulator` needs `--Mapper.max_extra_param 1e9`, or it triangulates nothing.
- The default camera is RAD_TAN_THIN_PRISM_FISHEYE, started from the tan(θ) Taylor series, which equals a pinhole. Never use FULL_OPENCV (it diverges). Large alternating high-order coefficients are normal; judge the fit by reprojection error.
- The principal point is not refined; the reported cx, cy are the prior's.
- Prior models are in metres: `select-pairs`, the alignment and the report flags use metric thresholds.
- Frames with fewer than `--min-obs` observations keep their prior pose (`restore-prior-poses`). Decide by observation count, never by displacement: a large correction with many observations is usually real drift in the prior being fixed.
- Outputs are in the prior model's metric frame. Never write into SCAN_DIR; preserve the original images, depth and app exports. Outputs conventionally go to `<SCAN_DIR>_opt`.
- `prior_pose_refine.py` depends only on numpy and scipy (`requirements.txt`). pycolmap is fine in analysis scripts and tests, not in the pipeline.
- Splatify-Anything exports (app id `fable-gs-capture`): `sparse/0` equals `poses_refined.jsonl` (the app's own refinement); use it as the app baseline. `poses.jsonl` is raw ARKit in the GL camera-to-world convention, and converting it to COLMAP flips both the camera and the world axes with diag(1, −1, −1). Its intrinsics are pinhole only, and focus is not always locked (fx drifts 1–2% within some scans).

## Environment

Machine-specific paths live in `local.env`, which is not committed; on a new machine, copy `local.env.example` and fill it in. Run `source local.env` before working:

| Variable | What it points to |
|---|---|
| `PATH` | gains the directory with a CUDA build of COLMAP 4.x; `prior-pose-refine` calls `colmap` |
| `PYTHON` | Python with numpy and scipy (plus pycolmap for analysis and tests); `prior-pose-refine` reads it. Run scripts with `-I` |
| `LICHTFELD` | the LichtFeld Studio binary |
| `SCAN_ROOT` | the scans exported by the capture app |
| `EXP_ROOT` | experiments and pre-change backups, outside the repository |

- The LichtFeld Studio checkout carries an uncommitted local `--pose-opt` patch: do not stash, pull, merge or rebuild it, or update COLMAP, unless the user asks.
- Check free disk space before running: a run takes 1.2–2 GB, and `work/` can be deleted once the result is checked.
- On a machine with one GPU, do not run two GPU jobs at once while timing.

## Languages and reporting

- Talk to the user in Traditional Chinese. Code, comments, CLI help and report output are English.
- Every document exists in English (`NAME.md`) and Traditional Chinese (`NAME.zh-TW.md`): README, AGENTS, docs/method and docs/benchmark. Update both in the same change, link the pair both ways at the top, and link within the same language.
- Decks and reports: show the time cost (preprocessing, training, overhead versus the baseline) next to quality; on comparison figures, box the visible differences and state which run is better; keep the main story short and decision-first, with details in an indexed appendix. Avoid frames that show people or nameplates.
