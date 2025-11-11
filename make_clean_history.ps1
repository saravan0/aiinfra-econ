# make_clean_history.ps1
# Run from project root (S:\aiinfra-econ). Creates a clean linear history of 11 phase commits.
# WARNING: This will rewrite main and force-push. Intended for personal repo.
# It creates a local backup branch called "backup-before-cleanup".

$ErrorActionPreference = "Stop"

Write-Host "=== make_clean_history.ps1 START ==="

# 0. Ensure we are in a git repo
if (-not (Test-Path ".git")) {
    Write-Error "This directory is not a git repository (no .git folder). cd to repo root and retry."
    exit 1
}

# 0.5 Ensure working tree state is visible
Write-Host "Current branch:"
git rev-parse --abbrev-ref HEAD

# 0.6 Create a temporary backup commit if working tree has changes
$porcelain = git status --porcelain
if ($porcelain) {
    Write-Host "Working tree has changes. Creating a temporary backup commit..."
    git add -A
    git commit -m "tmp: backup current working tree before history rewrite"
} else {
    Write-Host "Working tree clean — no temporary backup commit needed."
}

# 1. Create a safety backup branch of current HEAD
$backupBranch = "backup-before-cleanup"
Write-Host "Creating/updating backup branch: $backupBranch"
git branch -f $backupBranch HEAD
Write-Host "Backup branch created/updated: $backupBranch"

# 2. If a previous clean-history branch exists locally, remove it to avoid confusion
$cleanBranch = "clean-history"
if (git show-ref --verify --quiet "refs/heads/$cleanBranch") {
    Write-Host "Local branch '$cleanBranch' already exists — deleting it first to start fresh."
    git branch -D $cleanBranch
}

# 3. Create orphan branch and clear index (preserve files on disk)
Write-Host "Creating orphan branch: $cleanBranch"
git checkout --orphan $cleanBranch

# Remove all from index (keeps files on disk). Use --cached to keep files in working tree.
git rm -rf --cached .

# Stage entire working tree (we will create cumulative commits)
git add -A

# Ordered commit messages (11 canonical phases)
$messages = @(
    "foundation: setup project skeleton and environment configs",
    "data: add raw global trade and climate datasets",
    "clean: preprocess and harmonize multi-source datasets",
    "feature: engineer economic exposure and tariff variables",
    "eda: exploratory analysis and visualization notebooks",
    "model: implement baseline models and training pipeline",
    "eval: benchmark models and add interpretability reports",
    "infra: containerize pipeline and add CI/CD config",
    "report: generate automated reporting and executive summary",
    "research: add paper-ready analyses and reproducibility notes",
    "final: integrate full AI infra-economic pipeline with reproducible research artifacts"
)

# 4. Commit cumulative snapshots for each phase
Write-Host "Committing cumulative snapshots for each phase..."
foreach ($msg in $messages) {
    # ensure everything staged (cumulative)
    git add -A
    git commit -m $msg
    Write-Host "Committed: $msg"
}

# 5. Rename new branch to main locally (temporarily)
$newMain = "main"
Write-Host "Renaming branch $cleanBranch -> $newMain locally..."
git branch -M $newMain

# 6. Force push new main to origin (overwrites remote main)
Write-Host "Force pushing new main branch to origin (this will overwrite remote main)."
git push origin $newMain --force

Write-Host ""
Write-Host "=== CLEAN HISTORY CREATION COMPLETE ==="
Write-Host "Local backup branch: $backupBranch"
Write-Host "Please verify remote repository on GitHub now."
