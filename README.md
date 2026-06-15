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

# deployment
## pre-test
### 1) 存在性 (默认, 最安全, 不连硬件)
    python -m deployment.tools.hardware_check

### 2) 图像 (实际连相机/触觉抓一帧, 可存图)
    python -m deployment.tools.hardware_check --stage camera --show

### 3) 主从同步 (⚠️ 会驱动从臂, 必须显式确认)
    python -m deployment.tools.hardware_check --stage teleop --confirm-move

## 数据采集 
    python -m deployment.collect \
        --robot.type=realman_ugripper_dual \
        --teleop.type=bi_realman_ugripper_leader \
        --dataset.repo_id=<dataset_id> \
        --robot.use_tactile=true \
        --dataset.single_task="抓笔" \
        --dataset.num_episodes=50

## 模型推理
    python -m deployment.inference \
        --robot.type=realman_ugripper_dual \
        --policy.path=<path to pretrained model>\
        --dataset.repo_id=<record_id> \
        --dataset.single_task=<task description> \
        --dataset.num_episodes=10

    python -m deployment.inference \
        --robot.type=realman_ugripper_dual \
        --policy.path=<path to pretrained model> \
        --dataset.repo_id=eval_pen \
        --match-policy
