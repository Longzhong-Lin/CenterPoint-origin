import os
import argparse
import numpy as np
from copy import deepcopy

import onnx
from onnx import numpy_helper, helper
from onnxsim import simplify


def simplify_original_model(encoder_path, decoder_path):
    
    def simplify_model(model_path):
        model = onnx.load(model_path)
        if model is None:
            print("File %s is not find! "%model_path)
        return simplify(model)
    
    model, check = simplify_model(encoder_path)
    if not check:
        print("[ERROR]:Simplify %s error!"% encoder_path)    
    onnx.save(model, encoder_path)

    model, check = simplify_model(decoder_path)
    if not check:
        print("[ERROR]:Simplify %s error!"% decoder_path)    
    onnx.save(model, decoder_path)


def delete_init(model):
    init_len = len(model.graph.initializer)
    for i in range(init_len):
        model.graph.initializer.pop()

def convert_input_nhwc_nchw(model):
    batch_dim = 1
    dim_list = [dim_val.dim_value for dim_val in model.graph.input[0].type.tensor_type.shape.dim]
    dim_list.insert(0, batch_dim)
    dim_list = np.array(dim_list)[[0,3,1,2]]
   
    input_node = onnx.helper.make_tensor_value_info('input.1', \
                                                    onnx.TensorProto.FLOAT, dim_list.tolist())
    model.graph.input.pop()
    model.graph.input.append(input_node)
    
    dim_list = [dim_val.dim_value for dim_val in model.graph.output[0].type.tensor_type.shape.dim]
    dim_list.insert(0, batch_dim)
    dim_list.insert(2, 1)
    dim_list = np.array(dim_list)[[0,3,1,2]]
   
    out_node = onnx.helper.make_tensor_value_info(model.graph.output[0].name, \
                                                  onnx.TensorProto.FLOAT, dim_list.tolist())
    model.graph.output.pop()
    model.graph.output.append(out_node)

def matmul_to_conv2d(node, model, init_dict):
    weight_name = node.input[1]
    weight_tensor = init_dict[weight_name]
    weight = numpy_helper.to_array(weight_tensor)
    weight = np.expand_dims(weight.transpose(1,0),[2,3])
    weight_tensor = numpy_helper.from_array(weight, name=weight_name)
    init_dict[weight_name] = weight_tensor
    node = helper.make_node(
        op_type="Conv", inputs=node.input,
        outputs=node.output, name=node.name,
        dilations = [1, 1], group = 1,
        kernel_shape = [1, 1], pads = [0,0,0,0],
        strides=[1,1]
    )
    model.graph.node.append(node)
    
def reducemax_to_maxpool(node, model):
    node = helper.make_node(op_type="MaxPool", inputs=node.input, \
                            outputs=node.output, name=node.name,  \
                            ceil_mode = 0, kernel_shape = [1,20], \
                            pads = [0,0,0,0], strides=[1,1])
    model.graph.node.append(node)

def convert_tile(node, init_dict):
    arr_name = node.input[1]
    arr = np.array([1,1,1,20],np.int64)
    tensor = numpy_helper.from_array(arr, name=arr_name)
    init_dict[arr_name] = tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplify onnx")
    parser.add_argument(
        "--work_dir", help="the dir to save logs and models",
        default='work_dirs/centerpoint_pillar_pretrain/onnx'
    )
    args = parser.parse_args()
    
    encoder_path = os.path.join(args.work_dir, "encoder.onnx")
    encoder_sim_path = os.path.join(args.work_dir, "encoder_sim.onnx")
    decoder_path = os.path.join(args.work_dir, "decoder.onnx")
  
    simplify_original_model(encoder_path, decoder_path)
    print("=========Firstly Simplified=========")
    
    ## modify encoder
    encoder = onnx.load(encoder_path)
    init_dict = {}
    for init_node in encoder.graph.initializer:
        init_dict[init_node.name] = init_node
    
    # delete nodes
    delete_dict = {}
    for node in encoder.graph.node:
        if node.op_type in {"Transpose", "Expand", "Squeeze"}:
            delete_dict[node.output[0]] = node

    val_len = len(encoder.graph.value_info)
    for idx in range(val_len):
        encoder.graph.value_info.pop()

    delete_init(encoder)

    # convert to NCHW
    convert_input_nhwc_nchw(encoder)
    
    # convert ops
    rm_list = []
    for node in encoder.graph.node:
        if node.input[0] in delete_dict.keys():
            node.input[0] = delete_dict[node.input[0]].input[0]
        
        if node.op_type == "MatMul":
            # node.op_type = "Conv"
            # matmul_to_conv2d(node, init_dict)
            rm_list.append(node)
            matmul_to_conv2d(node, encoder, init_dict)
            # transpose_node_1 = helper.make_node(
            #     op_type="Transpose", inputs=[node.input[0]], \
            #     outputs=[node.name+'_transpose_1'], name=node.name+"_Transpose_1", \
            #     perm=[0,3,2,1]
            # )
            # node.input[0] = transpose_node_1.output[0]
            # transpose_node_2 = helper.make_node(
            #     op_type="Transpose", inputs=[node.name+'_transpose_2'], \
            #     outputs=node.output, name=node.name+"_Transpose_2", \
            #     perm=[0,3,2,1]
            # )
            # node.output[0] = transpose_node_2.input[0]
            # encoder.graph.node.append(transpose_node_1)
            # encoder.graph.node.append(transpose_node_2)
            
        if node.op_type == "ReduceMax":
            rm_list.append(node)
            reducemax_to_maxpool(node, encoder)
        
        if node.op_type == "Tile":
            convert_tile(node, init_dict)
        
        if node.op_type == "Concat":
            node.attribute[0].i = 1

    for node in encoder.graph.output:
        if node.name in delete_dict.keys():
            node.name = delete_dict[node.name].input[0]

    for name,tensor in init_dict.items():
        encoder.graph.initializer.append(tensor)

    for keys,node in delete_dict.items():
        encoder.graph.node.remove(node)

    for node in rm_list:
        encoder.graph.node.remove(node)

    onnx.save(encoder, encoder_sim_path)
    
    print("=========Encoder Modified=========")