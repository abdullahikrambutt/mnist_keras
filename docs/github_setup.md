# GitHub setup guide

Use these commands from the folder that contains this project.

## 1. Initialize Git

```bash
cd mnist-keras-github-repo
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

git init
git add .
git commit -m "Initial MNIST Keras project"
```

## 2. Create an empty repo on GitHub

Create a new GitHub repository named something like:

```text
mnist-keras-classifier
```

Do not initialize it with a README, license, or `.gitignore` if you are pushing this recreated project, because those files already exist locally.

## 3. Connect local repo to GitHub

Replace `YOUR_USERNAME` and `YOUR_REPO_NAME`:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

## 4. Future updates

```bash
git status
git add .
git commit -m "Describe your change"
git push
```
