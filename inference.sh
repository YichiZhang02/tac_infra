

python -m deployment.inference \
        --robot.type=realman_ugripper_dual \
        --policy.path=/home/a/yichi/tac_infra/playground/results/models/rm_umi_dual_pen_open_starvla_groot_wristonly_false_tactile_none_state_joint/checkpoints/005000/pretrained_model \
        --dataset.repo_id=/eval_pen2 \
        --match_policy=true