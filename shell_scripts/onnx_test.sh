val_mini()
{
python tools/test_onnx.py \
configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_mini.py \
--work_dir work_dirs/centerpoint_pillar_pretrain/val-mini/onnx \
--checkpoint work_dirs/centerpoint_pillar_pretrain/onnx/centerpoint.onnx
}

testing()
{
python tools/test_onnx.py \
configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_mini.py \
--work_dir work_dirs/centerpoint_pillar_pretrain/test/onnx \
--checkpoint work_dirs/centerpoint_pillar_pretrain/onnx/centerpoint.onnx \
--testset \
--save_data
}

# val_mini
testing
