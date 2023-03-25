val_mini()
{
python tools/test_onnx.py \
configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_mini.py \
--work_dir work_dirs/centerpoint_pillar_pretrain/val-mini/onnx/MatMul_Max_om \
--checkpoint work_dirs/centerpoint_pillar_pretrain/onnx/MatMul_Max_om/centerpoint.onnx
}

testing()
{
python tools/test_onnx.py \
configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_mini.py \
--work_dir work_dirs/centerpoint_pillar_pretrain/test/onnx/MatMul_Max_om \
--checkpoint work_dirs/centerpoint_pillar_pretrain/onnx/MatMul_Max_om/centerpoint.onnx \
--testset \
--save_data
}

# val_mini
testing
