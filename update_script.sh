# conda activate hugo_blog venv
conda activate hugo_blog

# hugo debug
hugo server -D

# hugo publish
hugo

# git add new files, modified and deleted files and commit
git add --all
git commit -m "update message..."

git push origin main

# hugo and firebase deployment
hugo && firebase deploy