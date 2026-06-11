# ACT
sh train.sh rm_nist_260320_strawberry act \
    8 32 5000 \
    false as_image joint

sh train.sh rm_nist_260320_strawberry act \
    8 32 5000 \
    false none joint

# DP
sh train.sh rm_nist_260320_strawberry diffusion \
    8 32 5000 \
    false as_image joint

sh train.sh rm_nist_260320_strawberry diffusion \
    8 32 5000 \
    false none joint

# StarVLA-Groot
sh train.sh rm_nist_260320_strawberry starvla_groot \
    8 8 5000 \
    false as_image joint

sh train.sh rm_nist_260320_strawberry starvla_groot \
    8 8 5000 \
    false none joint

# Pi05
sh train.sh rm_nist_260320_strawberry pi05 \
    8 6 6500 \
    false as_image joint

sh train.sh rm_nist_260320_strawberry pi05 \
    8 8 5000 \
    false none joint
