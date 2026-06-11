# tac_infra
A private infra for VTLA

# quick start
    conda create -n vtla python=3.10 -y
    conda activate vtla
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements.txt

# git useage
    # 开始写代码前
    git pull --rebase origin main

    # 写完后
    git add .
    git commit -m "1"
    git push origin main

    # 另一台服务器同步
    git pull origin main

    # 强制同步
    git fetch origin
    git reset --hard origin/main