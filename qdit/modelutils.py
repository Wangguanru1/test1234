import gc
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
from qdit.qLinearLayer import find_qlinear_layers
from qdit.qBlock import QuantDiTBlock,QuantAttention,QuantMlp
from qdit.gptq import GPTQ, Quantizer_GPTQ
from qdit.quant import TimestepPermutationQuantizer
from functools import partial
from models.models import DiTBlock
from .quant import PermutationQuantizer

from .quant import quantize_activation_wrapper, quantize_attn_v_wrapper, quantize_attn_k_wrapper, quantize_attn_q_wrapper, optimize_timestep_groups_assignment

def add_permutation_matrix(model,device,args):
    minmax_data = torch.load(args.minmax_path)
    for name, module in model.named_modules():
        if isinstance(module, QuantAttention):
            name1 = name + '.qkv'
            min_features = minmax_data['min_values'][name1]
            max_features = minmax_data['max_values'][name1]
            module.input_quant.get_permutation_matrix(torch.cat([min_features,max_features],dim=1))
            print(f'{name1} finish get_permutation_matrix')
            name2 = name + '.proj'
            min_features = minmax_data['min_values'][name2]
            max_features = minmax_data['max_values'][name2]
            module.act_quant.get_permutation_matrix(torch.cat([min_features,max_features],dim=1))
            print(f'{name2} finish get_permutation_matrix')
        elif isinstance(module,QuantMlp):
            name1 = name + '.fc1'
            min_features = minmax_data['min_values'][name1]
            max_features = minmax_data['max_values'][name1]
            module.input_quant.get_permutation_matrix(torch.cat([min_features,max_features],dim=1))
            print(f'{name1} finish get_permutation_matrix')
            name2 = name + '.fc2'
            min_features = minmax_data['min_values'][name2]
            max_features = minmax_data['max_values'][name2]
            module.act_quant.get_permutation_matrix(torch.cat([min_features,max_features],dim=1))
            print(f'{name2} finish get_permutation_matrix')

def add_timestep_permutation_matrix(model,device,args):
    minmax_data = torch.load(args.minmax_path)
    for name, module in model.named_modules():
        if isinstance(module, QuantAttention):
            name1 = name + '.qkv'
            # 构建每个时间步的特征数据
            timestep_features = {}
            for timestep_key in minmax_data['min_values'][name1].keys():
                min_features = minmax_data['min_values'][name1][timestep_key]  # shape: [n_channels]
                max_features = minmax_data['max_values'][name1][timestep_key]  # shape: [n_channels]
                # 将min和max合并作为特征维度
                combined_features = torch.cat([min_features.unsqueeze(1), max_features.unsqueeze(1)], dim=1)  # shape: [n_channels, 2]
                timestep_features[timestep_key] = combined_features

            module.input_quant.get_permutation_matrix(timestep_features)
            #print(f'{name1} finish get_timestep_permutation_matrices')
            
            # if name1 == 'blocks.0.attn.qkv':
            #     # 保存第0层qkv的时间步激活值
            #     module.input_quant.save_act = True
            #     module.input_quant._saved_activations = {}
            
            name2 = name + '.proj'
            # 为proj层构建时间步特征数据
            timestep_features = {}
            for timestep_key in minmax_data['min_values'][name2].keys():
                min_features = minmax_data['min_values'][name2][timestep_key]
                max_features = minmax_data['max_values'][name2][timestep_key]
                combined_features = torch.cat([min_features.unsqueeze(1), max_features.unsqueeze(1)], dim=1)
                timestep_features[timestep_key] = combined_features

            module.act_quant.get_permutation_matrix(timestep_features)
            print(f'{name2} finish get_timestep_permutation_matrices')
            
        elif isinstance(module,QuantMlp):
            name1 = name + '.fc1'
            # 构建每个时间步的特征数据
            timestep_features = {}
            for timestep_key in minmax_data['min_values'][name1].keys():
                min_features = minmax_data['min_values'][name1][timestep_key]
                max_features = minmax_data['max_values'][name1][timestep_key]
                combined_features = torch.cat([min_features.unsqueeze(1), max_features.unsqueeze(1)], dim=1)
                timestep_features[timestep_key] = combined_features
            
            module.input_quant.get_permutation_matrix(timestep_features)
            print(f'{name1} finish get_timestep_permutation_matrices')
            
            name2 = name + '.fc2'
            timestep_features = {}
            for timestep_key in minmax_data['min_values'][name2].keys():
                min_features = minmax_data['min_values'][name2][timestep_key]
                max_features = minmax_data['max_values'][name2][timestep_key]
                combined_features = torch.cat([min_features.unsqueeze(1), max_features.unsqueeze(1)], dim=1)
                timestep_features[timestep_key] = combined_features
            
            module.act_quant.get_permutation_matrix(timestep_features)
            print(f'{name2} finish get_timestep_permutation_matrices')

def add_act_quant_wrapper(model, device, args, scales):
    blocks = model.blocks
    
    for i in range(len(blocks)):
        args_i = copy.deepcopy(args)
        args_i.weight_group_size = args.weight_group_size[i]
        args_i.act_group_size = args.act_group_size[i]
        m = None
        if isinstance(blocks[i], DiTBlock):
            m = QuantDiTBlock(
                dit_block=blocks[i],
                args=args_i,
            )
        elif isinstance(blocks[i], QuantDiTBlock):
            m = blocks[i]

        if m is None:
            continue

        m = m.to(device)

        nameTemplate = 'blocks.{}.{}.{}'
        
        m.attn.input_quant.configure(
            partial(quantize_activation_wrapper, args=args_i),
            scales[nameTemplate.format(i, 'attn', 'qkv')]
        )
        m.attn.act_quant.configure(
            partial(quantize_activation_wrapper, args=args_i),
            scales[nameTemplate.format(i, 'attn', 'proj')]
        )
        if args.quantize_bmm_input:
            m.attn.q_quant.configure(
                partial(quantize_attn_q_wrapper, args=args_i),
                None
            )
            m.attn.k_quant.configure(
                partial(quantize_attn_k_wrapper, args=args_i),
                None
            )
            m.attn.v_quant.configure(
                partial(quantize_attn_v_wrapper, args=args_i),
                None
            )

        m.mlp.input_quant.configure(
            partial(quantize_activation_wrapper, args=args_i),
            scales[nameTemplate.format(i, 'mlp', 'fc1')]
        )
        m.mlp.act_quant.configure(
            partial(quantize_activation_wrapper, args=args_i),
            scales[nameTemplate.format(i, 'mlp', 'fc2')]
        )
        
        
        blocks[i] = m.cpu()
        torch.cuda.empty_cache()
    return model

def quantize_model(model, device, args):
    blocks = model.blocks
    for i in tqdm(range(len(blocks))):
        args_i = copy.deepcopy(args)
        args_i.weight_group_size = args.weight_group_size[i]
        args_i.act_group_size = args.act_group_size[i]
        m = None
        if isinstance(blocks[i], DiTBlock):
            m = QuantDiTBlock(
                dit_block=blocks[i],
                args=args_i,
            )
        elif isinstance(blocks[i], QuantDiTBlock):
            m = blocks[i]

        if m is None:
            continue

        m = m.to(device)
        m.mlp.fc1.quant()
        m.mlp.fc2.quant()
        m.attn.qkv.quant()
        m.attn.proj.quant()

        blocks[i] = m.cpu()
        torch.cuda.empty_cache()
    return model

def quantize_layer(model, name, device):
    blocks = model.blocks
    i = int(name.split(".")[1])
    assert(isinstance(blocks[i], QuantDiTBlock))
    m = blocks[i]
    m = m.to(device)

    if name.endswith("mlp.fc1"):
        m.mlp.fc1.quant()
    elif name.endswith("mlp.fc2"):
        m.mlp.fc2.quant()
    elif name.endswith("attn.qkv"):
        m.attn.qkv.quant()
    elif name.endswith("attn.proj"):
        m.attn.proj.quant()
    else:
        raise NotImplementedError

    blocks[i] = m.cpu()
    torch.cuda.empty_cache()
    return model

def quantize_block(block, device):
    assert(isinstance(block, QuantDiTBlock))
    block.to(device)

    block.mlp.fc1.quant()
    block.mlp.fc2.quant()
    block.attn.qkv.quant()
    block.attn.proj.quant()

    torch.cuda.empty_cache()

def quantize_model_gptq(model, device, args, dataloader):
    print('Starting GPTQ quantization ...')
    blocks = model.blocks
    
    quantizers = {}
    for i in tqdm(range(len(blocks))):
        args_i = copy.deepcopy(args)
        args_i.weight_group_size = args.weight_group_size[i]
        args_i.act_group_size = args.act_group_size[i]
        if isinstance(blocks[i], DiTBlock):
            m = QuantDiTBlock(
                dit_block=blocks[i],
                args=args_i,
            )
        elif isinstance(blocks[i], QuantDiTBlock):
            m = blocks[i]
        else:
            continue
        
        for name, module in model.named_modules():
            if isinstance(module, TimestepPermutationQuantizer):
                module.cali_gptq = True
                #print(f'{name} set cali_gptq True')

        block = m.to(device)

        block_layers = find_qlinear_layers(block)

        sequential = [list(block_layers.keys())]

        for name, module in model.named_modules():
                if isinstance(module, TimestepPermutationQuantizer):
                    module.cali_gptq = True
       
        for names in sequential:
            subset = {n: block_layers[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer_GPTQ()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.w_sym, mse=False, 
                    channel_group=args.weight_channel_group,
                    clip_ratio=args.w_clip_ratio,
                    quant_type=args.quant_type
                )
                
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            
            
            model.to(device)
            for calib_x, calib_t, calib_y in tqdm(dataloader):
                model(calib_x.to(device), calib_t.to(device), calib_y.to(device))

            for h in handles:
                h.remove()
            
            for name in subset:
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.weight_group_size[0]
                )
                subset[name].quantized = True
                quantizers['model.blocks.%d.%s' % (i, name)] = gptq[name].quantizer.cpu()
                gptq[name].free()

            del gptq

        blocks[i] = block.cpu()
        for name, module in model.named_modules():
            if isinstance(module, TimestepPermutationQuantizer):
                module.cali_gptq = False
        del block, m
        torch.cuda.empty_cache()
        gc.collect()

    return model
