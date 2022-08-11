import torch
import random
import heapq
from torch_pruning.prune.strategy import round_pruning_amount

def l1prune(model, layer_id, compress_rate):
    """对某一层按 compress_rate 以 l1norm 为指标返回需要剪枝的 channel
    layer_id 为要计算 l1 指标的layer的 id ,为 int 数据从 0 到 22 (SSD512到24)
    VGG 有 15 个卷积层，对应 0 到 14，具体为
    VGG: 1_1, 1_2, 2_1, 2_2, 3_1, 3_2, 3_3, 4_1, 4_2, *4_3, 5_1, 5_2, 5_3, fc6, *fc7
    Extra 有 8 个卷积层，对应 15 到 22，具体为
    Extra: 6_1, *6_2, 7_1, *7_2, 8_1, *8_2, 9_1, *9_2, (10_1, *10_2)
    打星号的是传到 detector 的层,其 id 为 9, 14, 16, 18, 20, 22, (24)
    compress_rate是该层需要剪枝的百分比，即会返回 compress_rate*outchannel 个channel
    """
    # conv_id = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28, 31, 33]
    # is_vgg = 'vgg' in layer_id if isinstance(layer_id, str) else layer_id in conv_id
    have_layers = layer_id.split('.')[-1].isdigit()
    if have_layers:
        model_id = layer_id.split(".")[-2] if len(layer_id.split('.')) == 2 else '.'.join(layer_id.split(".")[:-1])
        layer_id = int(layer_id.split('.')[-1])
    else:
        model_id = layer_id
    layers = eval(f'model.module.{model_id} if hasattr(model, "module") else model.{model_id}')
    filters = layers[layer_id].weight if have_layers else layers.weight
    measures = [torch.sum(torch.abs(filters[k])) for k in range(int(filters.size(0)))]
    measures = list(enumerate(measures))
    measures.sort(key=lambda item: item[1])
    num_to_prune = int(int(filters.size(0)) * compress_rate) if compress_rate < 1 else int(compress_rate)
    # num_to_prune = num_to_prune if num_to_prune else 8
    num_to_prune = min(filters.size(0) - 1, num_to_prune)
    if have_layers:
        model_id = '.'.join([model_id, str(layer_id)])
    return ['module.%s.weight.%d' % (model_id, int(measures[k][0])) for k in range(num_to_prune)] if hasattr(model, 'module') \
        else ['%s.weight.%d' % (model_id, int(measures[k][0])) for k in range(num_to_prune)]

def global_bnprune(model, compress_rate, part, exclude=[], groups=[]):
    # conv_id = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28, 31, 33]
    # is_vgg = 'vgg' in layer_id if isinstance(layer_id, str) else layer_id in conv_id
    all_measures = []
    measures = {}
    if len(groups):
        for group in groups:
            measures[group[0]] = []
            for model_id in group:
                layer = eval(f'model.module.{model_id} if hasattr(model, "module") else model.{model_id}')
                measures[group[0]].append(torch.abs(layer.weight))
            measures[group[0]] = torch.stack(measures[group[0]], dim=0)
            measures[group[0]] = [['__'.join([group[0], str(measures[group[0]].shape[1]), str(i)]), torch.mean(measures[group[0]][:,i])] for i in range(measures[group[0]].shape[1])]
            all_measures.extend(measures[group[0]])
    for k, v in model.named_modules():
        if hasattr(v, 'weight') and isinstance(v, torch.nn.BatchNorm2d) or isinstance(v, torch.nn.SyncBatchNorm) and any(
                [k.startswith(p) for p in part]) and all([p not in k for p in exclude]):
            have_layers = [i.isdigit() for i in k.split('.')]
            if any(have_layers):
                model_id = []
                for i, ele in enumerate(k.split('.')):
                    if have_layers[i]:
                        model_id[-1] = model_id[-1] + f'[{ele}]'
                    else:
                        model_id.append(ele)
                model_id = '.'.join(model_id)
            else:
                model_id = k
            if any([model_id in group for group in groups]):
                continue
            layer = eval(f'model.module.{model_id} if hasattr(model, "module") else model.{model_id}')
            measures[model_id] = [['__'.join([model_id, str(layer.weight.shape[0]), str(i)]), torch.abs(layer.weight[i])] for i in range(layer.weight.shape[0])]
            all_measures.extend(measures[model_id])
    all_measures.sort(key=lambda item: item[1])
    num_to_prune = int(len(all_measures) * compress_rate) if compress_rate < 1 else int(compress_rate)
    # num_to_prune = num_to_prune if num_to_prune else 8
    # num_to_prune = min(filters.size(0) - 1, num_to_prune)
    prune_ids = [all_measures[k][0] for k in range(num_to_prune)]
    prune_key = set([prune_ids[i].split('__')[0] for i in range(len(prune_ids))])
    prune_list = {key: [] for key in prune_key}
    for id in prune_ids:
        if len(prune_list[id.split('__')[0]]) < eval(id.split('__')[1]) - 1:
            prune_list[id.split('__')[0]].append(eval(id.split('__')[2]))
        else: print(f"The pruning ratio of {id.split('__')[0]} has reached maximum, we will reserve one channel for this layer.")

    return prune_list

def group_l1prune(model, group, compress_rate, round_to=1):
    """对某一层按 compress_rate 以 l1norm 为指标返回需要剪枝的 channel
    layer_id 为要计算 l1 指标的layer的 id ,为 int 数据从 0 到 22 (SSD512到24)
    VGG 有 15 个卷积层，对应 0 到 14，具体为
    VGG: 1_1, 1_2, 2_1, 2_2, 3_1, 3_2, 3_3, 4_1, 4_2, *4_3, 5_1, 5_2, 5_3, fc6, *fc7
    Extra 有 8 个卷积层，对应 15 到 22，具体为
    Extra: 6_1, *6_2, 7_1, *7_2, 8_1, *8_2, 9_1, *9_2, (10_1, *10_2)
    打星号的是传到 detector 的层,其 id 为 9, 14, 16, 18, 20, 22, (24)
    compress_rate是该层需要剪枝的百分比，即会返回 compress_rate*outchannel 个channel
    """
    # conv_id = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28, 31, 33]
    # is_vgg = 'vgg' in layer_id if isinstance(layer_id, str) else layer_id in conv_id
    filters = 0.
    for model_id in group:
        layers = eval(f'model.module.{model_id} if hasattr(model, "module") else model.{model_id}')
        filters += torch.mean(torch.abs(layers.weight), dim=(1,2,3))
    measures = [filters[k] for k in range(filters.shape[0])]
    measures = list(enumerate(measures))
    measures.sort(key=lambda item: item[1])
    num_to_prune = int(filters.shape[0] * compress_rate) if compress_rate < 1 else int(compress_rate)
    # num_to_prune = num_to_prune if num_to_prune else 8
    num_to_prune = min(filters.shape[0] - 1, num_to_prune)
    num_to_prune = round_pruning_amount(filters.shape[0], num_to_prune, round_to)
    return [measures[k][0] for k in range(num_to_prune)]


def global_l1prune(model, compress_rate, part):
    """对某一层按 compress_rate 以 l1norm 为指标返回需要剪枝的 channel
    layer_id 为要计算 l1 指标的layer的 id ,为 int 数据从 0 到 22 (SSD512到24)
    VGG 有 15 个卷积层，对应 0 到 14，具体为
    VGG: 1_1, 1_2, 2_1, 2_2, 3_1, 3_2, 3_3, 4_1, 4_2, *4_3, 5_1, 5_2, 5_3, fc6, *fc7
    Extra 有 8 个卷积层，对应 15 到 22，具体为
    Extra: 6_1, *6_2, 7_1, *7_2, 8_1, *8_2, 9_1, *9_2, (10_1, *10_2)
    打星号的是传到 detector 的层,其 id 为 9, 14, 16, 18, 20, 22, (24)
    compress_rate是该层需要剪枝的百分比，即会返回 compress_rate*outchannel 个channel
    """
    # conv_id = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28, 31, 33]
    # is_vgg = 'vgg' in layer_id if isinstance(layer_id, str) else layer_id in conv_id
    all_measures = []
    for k, v in model.named_modules():
        if hasattr(v, 'weight') and any([p in k for p in part]):
            have_layers = k.split('.')[-1].isdigit()
            if have_layers:
                model_id = k.split(".")[-2] if len(k.split('.')) == 2 else '.'.join(k.split(".")[:-1])
                layer_id = int(k.split('.')[-1])
            else:
                model_id = k
                layer_id = k
            layers = eval(f'model.module.{model_id} if hasattr(model, "module") else model.{model_id}')
            filters = layers[layer_id].weight if have_layers else layers.weight
            measures = [torch.mean(torch.abs(filters[k])) for k in range(int(filters.size(0)))]
            if have_layers:
                model_id = '.'.join([model_id, str(layer_id)])
            measures = list(zip([f'module.{model_id}.weight.{i}' if hasattr(model, 'module') else f'{model_id}.weight.{i}' for i in range(len(measures))], measures))
            all_measures.extend(measures)
    all_measures.sort(key=lambda item: item[1])
    num_to_prune = int(len(all_measures) * compress_rate) if compress_rate < 1 else int(compress_rate)
    # num_to_prune = num_to_prune if num_to_prune else 8
    # num_to_prune = min(filters.size(0) - 1, num_to_prune)

    return [all_measures[k][0] for k in range(num_to_prune)]


def l2prune(model, layer_id, compress_rate):
    """对某一层按 compress_rate 以 l1norm 为指标返回需要剪枝的 channel
    layer_id 为要计算 l1 指标的layer的 id ,为 int 数据从 0 到 22 (SSD512到24)
    VGG 有 15 个卷积层，对应 0 到 14，具体为
    VGG: 1_1, 1_2, 2_1, 2_2, 3_1, 3_2, 3_3, 4_1, 4_2, *4_3, 5_1, 5_2, 5_3, fc6, *fc7
    Extra 有 8 个卷积层，对应 15 到 22，具体为
    Extra: 6_1, *6_2, 7_1, *7_2, 8_1, *8_2, 9_1, *9_2, (10_1, *10_2)
    打星号的是传到 detector 的层,其 id 为 9, 14, 16, 18, 20, 22, (24)
    compress_rate是该层需要剪枝的百分比，即会返回 compress_rate*outchannel 个channel
    """
    have_layers = layer_id.split('.')[-1].isdigit()
    if have_layers:
        model_id = layer_id.split(".")[-2] if len(layer_id.split('.')) == 2 else '.'.join(layer_id.split(".")[:-1])
        layer_id = int(layer_id.split('.')[-1])
    else:
        model_id = layer_id
    layers = eval(f'model.module.{model_id} if hasattr(model, "module") else model.{model_id}')
    filters = layers[layer_id].weight if have_layers else layers.weight
    measures = [torch.sum(torch.abs(filters[k]) ** 2) for k in range(int(filters.size(0)))]
    measures = list(enumerate(measures))
    measures.sort(key=lambda item: item[1])
    num_to_prune = int(int(filters.size(0)) * compress_rate) if compress_rate <= 1 else int(compress_rate)
    num_to_prune = min(filters.size(0) - 1, num_to_prune)
    if have_layers:
        model_id = '.'.join([model_id, str(layer_id)])
    return ['module.%s.weight.%d' % (model_id, int(measures[k][0])) for k in range(num_to_prune)] if hasattr(model, 'module') \
        else ['%s.weight.%d' % (model_id, int(measures[k][0])) for k in range(num_to_prune)]

def correlate_prune(model, layer_id, compress_rate):
    def correlate(X, Y):
        """
        X: tensor [batch, n]
        Y: tensor [batch, n]
        return: correlate: cov(X,Y)/std(X)std(Y) [batch]
        """
        ex = X.mean(dim=1)  # [batch]
        ey = Y.mean(dim=1)
        exy = torch.mul(X, Y).mean(dim=1)  # [batch]
        cov = exy - torch.mul(ex, ey)  # [batch]
        sigmax = X.std(dim=1)
        sigmay = Y.std(dim=1)
        out = cov / (sigmax * sigmay)
        return torch.abs(out)

    def get_correlate_filter(output):
        a = 1
        b = output.size(0)
        output = output.unsqueeze(0)
        output = output.view(a, b, -1)
        record = torch.zeros([a, b-1, b-1]).to(output.device)
        for i in range(int(b)-1):
            for j in range(i, int(b)-1):
                record[:,i,j] = correlate(output[:,i,:], output[:,j+1,:])
        record = record.squeeze(0)
        return record

    def determine_interval(n, step):
        """
        n: a non-negative real number  e.g. 5
        step: partition [1,3,5,7,9] --> [0, 1] (2, 3] (4, 5] (6, 7] (8, 9]
        return: which interval n belongs to  above example returns 2
        """
        inx = 0
        for i in range(len(step)):
            if n > step[i]:
                inx += 1
            else:
                return inx

    have_layers = layer_id.split('.')[-1].isdigit()
    if have_layers:
        model_id = layer_id.split(".")[-2] if len(layer_id.split('.')) == 2 else '.'.join(layer_id.split(".")[:-1])
        layer_id = int(layer_id.split('.')[-1])
    else:
        model_id = layer_id
    layers = eval(f'model.module.{model_id} if hasattr(model, "module") else model.{model_id}')
    filters = layers[layer_id].weight if have_layers else layers.weight

    corr_mat = get_correlate_filter(filters)
    w = int(corr_mat.size(0))
    corres = [float(corr_mat[i, j]) for i in range(w) for j in range(i, w)]

    num_to_prune = int((w + 1) * compress_rate)
    maxes = map(corres.index, heapq.nlargest(num_to_prune, corres))  # biggest (num_to_prunes) numbers' index
    step = [sum(range(w - i, w + 1)) - 1 for i in range(w)]  # [w-1, w+(w-1)-1, ..., w+(w-1)+...+1-1)]

    pairs_to_prune = []
    for n in maxes:
        idx = determine_interval(n, step)
        if idx > 0:
            jdx = n - step[idx - 1] + idx
        else:
            jdx = n + 1
        pairs_to_prune.append((idx, jdx))

    if have_layers:
        model_id = '.'.join([model_id, str(layer_id)])
    to_prune_list = set()
    to_reserve_list = set()
    for layer_pair in pairs_to_prune:
        id = layer_pair[0] if layer_pair[0] not in to_reserve_list else layer_pair[1]
        if id in to_reserve_list: continue
        to_reserve_list.add(layer_pair[not layer_pair.index(id)])
        to_prune_list.add(f'module.{model_id}.weight.{id}' if hasattr(model, 'module') else f'{model_id}.weight.{id}')
    return list(to_prune_list)