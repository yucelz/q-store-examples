# GitHub Commands for `main` and `dev` Branch Workflows

This document covers the **most common GitHub/Git commands** used when working with a two-branch workflow:

- **`main`** → stable, production-ready code
- **`dev`** → active development and integration branch

It assumes Git is used locally and GitHub is the remote repository host.

---

## 1. Repository Setup

### Clone a Repository
```bash
git clone https://github.com/username/repository.git
cd repository
```

### Check Remote URLs
```bash
git remote -v
```

---

## 2. Branch Basics

### List Branches
```bash
# Local branches
git branch

# Remote branches
git branch -r

# All branches
git branch -a
```

### Create Branches
```bash
# Create dev branch from current branch
git branch dev

# Create and switch to dev
git checkout -b dev
```

### Switch Branches
```bash
git checkout main
git checkout dev

# Modern alternative
git switch main
git switch dev
```

---

## 3. Keeping Branches Up to Date

### Fetch Latest Changes (No Merge)
```bash
git fetch origin
```

### Pull Changes
```bash
# Pull latest main
git checkout main
git pull origin main

# Pull latest dev
git checkout dev
git pull origin dev
```

---

## 4. Typical Development Workflow (Feature → dev)

### Create a Feature Branch from dev
```bash
git checkout dev
git pull origin dev
git checkout -b feature/my-feature
```

### Stage and Commit Changes
```bash
git status
git add .
git commit -m "Add new feature"
```

### Push Feature Branch to GitHub
```bash
git push origin feature/my-feature
```

### Merge Feature into dev (Local)
```bash
git checkout dev
git merge feature/my-feature
```

### Delete Feature Branch
```bash
# Local
git branch -d feature/my-feature

# Remote
git push origin --delete feature/my-feature
```

---

## 5. Merging `dev` into `main`

### Ensure Branches Are Updated
```bash
git checkout dev
git pull origin dev

git checkout main
git pull origin main
```

### Merge dev into main
```bash
git merge dev
```

### Push main to GitHub
```bash
git push origin main
```

---

## 6. Handling Merge Conflicts

### When a Conflict Occurs
```bash
git status
```

### Resolve Conflicts
- Open conflicted files
- Fix sections marked with:
```text
<<<<<<<
=======
>>>>>>>
```

### Mark Conflict as Resolved
```bash
git add conflicted_file.ext
git commit
```

---

## 7. Rebasing (Optional, Cleaner History)

### Rebase Feature Branch onto dev
```bash
git checkout feature/my-feature
git rebase dev
```

### Rebase dev onto main (Advanced)
```bash
git checkout dev
git rebase main
```

⚠️ Avoid rebasing shared branches unless you know the impact.

---

## 8. Comparing Branches

### View Differences
```bash
git diff main..dev
git diff dev..main
```

### View Commit Differences
```bash
git log main..dev --oneline
git log dev..main --oneline
```

---

## 9. Resetting and Undoing Changes

### Undo Uncommitted Changes
```bash
git checkout -- file.txt
```

### Undo Last Commit (Keep Changes)
```bash
git reset --soft HEAD~1
```

### Undo Last Commit (Discard Changes)
```bash
git reset --hard HEAD~1
```

---

## 10. Tagging Releases on `main`

### Create a Release Tag
```bash
git tag v1.0.0
git push origin v1.0.0
```

### Annotated Tag (Recommended)
```bash
git tag -a v1.0.0 -m "Production release"
git push origin v1.0.0
```

---

## 11. Branch Protection (GitHub UI)

Common protections for `main`:
- Require pull requests
- Require reviews
- Require status checks
- Disallow force pushes

Recommended:
- Merge **feature → dev → main** via Pull Requests

---

## 12. Common GitHub Flow Summary

```text
main  ← stable, production-ready
  ↑
 dev   ← integration branch
  ↑
feature/* ← short-lived development branches
```

---

## 13. Useful Shortcuts

```bash
git status -sb        # Short status
git log --oneline     # Compact log
git branch -vv        # Branch tracking info
git clean -fd         # Remove untracked files
```

---

## 14. Best Practices

- Never commit directly to `main`
- Keep `dev` deployable
- Use clear commit messages
- Delete merged branches
- Tag releases on `main`

---

**End of Document**

