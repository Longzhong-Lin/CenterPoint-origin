val_mini()
{
python tools/dist_test.py \
configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_mini.py \
--work_dir work_dirs/centerpoint_pillar_pretrain/val-mini \
--checkpoint work_dirs/centerpoint_pillar_pretrain/latest.pth
}

testing()
{
python tools/dist_test.py \
configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_mini.py \
--work_dir work_dirs/centerpoint_pillar_pretrain/test \
--checkpoint work_dirs/centerpoint_pillar_pretrain/latest.pth \
--testset
}

# val_mini
testing
