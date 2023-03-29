val_mini()
{
python tools/nusc_tracking/pub_test.py \
--work_dir work_dirs/centerpoint_pillar_pretrain/val-mini/tracking \
--checkpoint work_dirs/centerpoint_pillar_pretrain/val-mini/infos_val_10sweeps_withvelo_filter_True.json \
--version v1.0-mini \
--root data/nuScenes-mini
}

testing()
{
python tools/nusc_tracking/pub_test.py \
--work_dir work_dirs/centerpoint_pillar_pretrain/test/tracking \
--checkpoint work_dirs/centerpoint_pillar_pretrain/test/infos_test_10sweeps_withvelo.json \
--version v1.0-test \
--root data/nuScenes-mini/v1.0-test
}

onnx_val_mini()
{
python tools/nusc_tracking/pub_test.py \
--work_dir work_dirs/centerpoint_pillar_pretrain/val-mini/onnx/MatMul_Max_om/tracking \
--checkpoint work_dirs/centerpoint_pillar_pretrain/val-mini/onnx/MatMul_Max_om/infos_val_10sweeps_withvelo_filter_True.json \
--version v1.0-mini \
--root data/nuScenes-mini
}

onnx_testing()
{
python tools/nusc_tracking/pub_test.py \
--work_dir work_dirs/centerpoint_pillar_pretrain/test/onnx/MatMul_Max_om/tracking \
--checkpoint work_dirs/centerpoint_pillar_pretrain/test/onnx/MatMul_Max_om/infos_test_10sweeps_withvelo.json \
--version v1.0-test \
--root data/nuScenes-mini/v1.0-test
}

om_val_mini()
{
python tools/nusc_tracking/pub_test.py \
--work_dir work_dirs/centerpoint_pillar_pretrain/val-mini/om/tracking \
--checkpoint work_dirs/centerpoint_pillar_pretrain/val-mini/om/infos_val_10sweeps_withvelo_filter_True.json \
--version v1.0-mini \
--root data/nuScenes-mini
}

om_testing()
{
python tools/nusc_tracking/pub_test.py \
--work_dir work_dirs/centerpoint_pillar_pretrain/test/om/tracking \
--checkpoint work_dirs/centerpoint_pillar_pretrain/test/om/infos_test_10sweeps_withvelo.json \
--version v1.0-test \
--root data/nuScenes-mini/v1.0-test
}

# val_mini
# testing

# onnx_val_mini
# onnx_testing

# om_val_mini
om_testing
