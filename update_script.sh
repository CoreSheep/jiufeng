# conda activate hugo_blog venv
conda activate hugo_blog

# hugo debug
hugo server -D

# hugo publish
hugo

# git add new files, modified and deleted files and commit
git add --all
git commit -m "Remove Tags section and add Blogs section to website"

git push origin main

# hugo and firebase deployment
hugo && firebase deploy