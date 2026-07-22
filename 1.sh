# prepare data
# merge 6
# rm_umi_dual_260706_pen_in_case_notac_undist_256
# rm_umi_dual_260706_pen_in_case_1_notac_undist_256
# rm_umi_dual_260707_pen_in_case_notac_undist_256
# rm_umi_dual_260707_pen_in_case_1_notac_undist_256
# rm_umi_dual_260708_pen_in_case_notac_undist_256
# rm_umi_dual_260711_pen_in_case_undist_256


# base
# rm_umi_dual_260707_pen_in_case_1_notac_undist_256





sh train.sh waic_base_6

sh train.sh <data> starvla_groot 8 16 40_000 true none joint joint strong

scp -O -P 1026 -i /home/dm/sshkey -r root@121.89.91.224:/mnt/data/xidong_data/tac_infra/playground/results/models/20260718_095809_waic_base_6_starvla_groot_wristonly_true_tactile_none_state_absolute_ee_action_relative_ee_aug_strong/checkpoints/050000 /home/dm/lerobot_tactile_ws/tac_infra/playground/results/models
