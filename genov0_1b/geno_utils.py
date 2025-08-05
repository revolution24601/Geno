import re
import os
import torch
import json
import shutil


def rename_state_dict(state_dict, model):
    # 获取 model 的参数名
    model_param_names = list(model.state_dict().keys())

    # 分离出包含和不包含 "数字.weight" 格式的参数名
    def separate_names(names, keys=["embed.weight",
                                    "output_head.weight",
                                    'task_filter.task_embeddings.weight',
                                    'task_filter.input_proj.weight',
                                    'task_filter.input_proj.bias',
                                    'task_filter.output_proj.weight',
                                    'task_filter.output_proj.bias',
                                    'layers.6.module.input_proj.bias',
                                    'layers.6.module.input_proj.weight',
                                    'layers.6.module.output_proj.bias',
                                    'layers.6.module.output_proj.weight',
                                    'layers.6.module.task_embeddings.weight',
                                    "layers.0.module.weight",
                                    "layers.7.module.weight"]):
        numbered_weight_names = []
        other_names = []
        for name in names:
            if name not in keys:
                numbered_weight_names.append(name)
            else:
                other_names.append(name)
        return numbered_weight_names, other_names

    state_numbered_weight_names, state_other_names = separate_names(state_dict.keys())
    model_numbered_weight_names, model_other_names = separate_names(model_param_names)
    print(f"state_other_names: {state_other_names}")
    print(f"model_other_names: {model_other_names}")
    # 对不包含 "数字.weight" 格式的参数名进行排序
    state_other_names.sort()
    model_other_names.sort()

    # 对包含 "数字.weight" 格式的参数名进行排序
    state_numbered_weight_names.sort()
    model_numbered_weight_names.sort()

    # print(f"state_numbered_weight_names: {state_numbered_weight_names}")
    # print(f"model_numbered_weight_names: {model_numbered_weight_names}")
    if len(model_numbered_weight_names) == 1:
        state_numbered_weight_names = state_numbered_weight_names[0]

    # 创建新的 state_dict
    new_state_dict = {}
    # 处理不包含 "数字.weight" 格式的参数名
    # for state_name, model_name in zip(state_other_names, model_other_names):
    #     # print(f"{state_name}-->{model_name}")
    #     new_state_dict[model_name] = state_dict[state_name]
    if "output_cnn1.weight" in model_other_names:
        new_state_dict["embed.weight"] = state_dict["layers.0.module.weight"]
        new_state_dict["output_cnn1.weight"] = state_dict["layers.7.module.weight"]
        new_state_dict["output_cnn1.bias"] = state_dict["layers.7.module.bias"]
        new_state_dict["output_cnn2.weight"] = state_dict["layers.9.module.weight"]
        new_state_dict["output_cnn2.bias"] = state_dict["layers.9.module.bias"]
        if "layers.11.module.weight" in state_dict:
            new_state_dict["output_cnn3.weight"] = state_dict["layers.11.module.weight"]
            new_state_dict["output_cnn3.bias"] = state_dict["layers.11.module.bias"]
            new_state_dict["output_head.weight"] = state_dict["layers.14.module.weight"]
        else:
            new_state_dict["output_head.weight"] = state_dict["layers.12.module.weight"]
    elif "output_cnn1.conv.weight" in model_other_names:
        new_state_dict["embed.weight"] = state_dict["layers.0.module.weight"]
        new_state_dict["output_cnn1.conv.weight"] = state_dict["layers.7.module.weight"]
        new_state_dict["output_cnn1.conv.bias"] = state_dict["layers.7.module.bias"]
        new_state_dict["output_cnn2.conv.weight"] = state_dict["layers.9.module.weight"]
        new_state_dict["output_cnn2.conv.bias"] = state_dict["layers.9.module.bias"]
        if "layers.11.module.weight" in state_dict:
            new_state_dict["output_cnn3.conv.weight"] = state_dict["layers.11.module.weight"]
            new_state_dict["output_cnn3.conv.bias"] = state_dict["layers.11.module.bias"]
            new_state_dict["output_head.weight"] = state_dict["layers.14.module.weight"]
        else:
            new_state_dict["output_head.weight"] = state_dict["layers.12.module.weight"]
    elif 'layers.6.module.shared_filter.bias' in state_dict:
        new_state_dict["embed.weight"] = state_dict["layers.0.module.weight"]
        new_state_dict["task_filter.shared_filter.weight"] = state_dict["layers.6.module.shared_filter.weight"]
        new_state_dict["task_filter.shared_filter.bias"] = state_dict["layers.6.module.shared_filter.bias"]
        new_state_dict["task_filter.task_embeddings.weight"] = state_dict["layers.6.module.task_embeddings.weight"]
        new_state_dict["output_head.weight"] = state_dict["layers.7.module.weight"]
    elif 'layers.6.module.input_proj.weight' in state_dict:
        print("V2")
        new_state_dict["embed.weight"] = state_dict["layers.0.module.weight"]
        new_state_dict["task_filter.input_proj.weight"] = state_dict["layers.6.module.input_proj.weight"]
        new_state_dict["task_filter.input_proj.bias"] = state_dict["layers.6.module.input_proj.bias"]
        new_state_dict["task_filter.output_proj.weight"] = state_dict["layers.6.module.output_proj.weight"]
        new_state_dict["task_filter.output_proj.bias"] = state_dict["layers.6.module.output_proj.bias"]
        new_state_dict["task_filter.task_embeddings.weight"] = state_dict["layers.6.module.task_embeddings.weight"]
        new_state_dict["output_head.weight"] = state_dict["layers.7.module.weight"]
    elif 'layers.6.module.task_embeddings.weight' in state_dict:
        new_state_dict["embed.weight"] = state_dict["layers.0.module.weight"]
        new_state_dict["task_filter.task_embeddings.weight"] = state_dict["layers.6.module.task_embeddings.weight"]
        new_state_dict["output_head.weight"] = state_dict["layers.7.module.weight"]
    else:
        new_state_dict["embed.weight"] = state_dict["layers.0.module.weight"]
        new_state_dict["output_head.weight"] = state_dict["layers.6.module.weight"]

    # 处理包含 "数字.weight" 格式的参数名
    for state_name, model_name in zip(state_numbered_weight_names, model_numbered_weight_names):
        # print(f"{state_name}-->{model_name}")
        new_state_dict[model_name] = state_dict[state_name]

    return new_state_dict


def get_checkpoint_paths(directory):
    """
    从指定目录中自动获取所有 .pth 文件的绝对路径。
    :param directory: 存放模型 checkpoint 文件的目录
    :return: .pth 文件的绝对路径列表
    """
    checkpoint_paths = []

    # 遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pth"):
                # 获取文件的绝对路径
                checkpoint_paths.append(os.path.join(root, file))

    # 按文件名排序（可选，按需要进行排序）
    checkpoint_paths.sort()

    return checkpoint_paths


def merge_pipeline_parallel_checkpoints(checkpoint_paths, output_pth_path):
    """
    合并流水线并行训练中各个阶段保存的模型权重文件为一个单一的 .pth 文件。
    :param checkpoint_paths: 包含每个进程（模型阶段）保存的 .pth 文件路径的列表
    :param output_pth_path: 输出合并后的模型文件路径
    """
    # 初始化一个空的模型字典
    model_state_dict = {}

    # 按顺序加载每个进程保存的模型权重
    for idx, checkpoint_path in enumerate(checkpoint_paths):
        print(f"加载第 {idx} 阶段的 checkpoint: {checkpoint_path}")

        # 加载每个 checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(len(model_state_dict.keys()))
        # 将该阶段的权重加入到合并的模型中
        model_state_dict.update(checkpoint)

    # 保存合并后的完整模型为 .pth 文件
    print(f"保存合并后的模型为: {output_pth_path}")
    torch.save(model_state_dict, output_pth_path)
    print("模型合并完成！")


def delete_old_subdirectories(directory, k):
    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"目录 {directory} 不存在。")
        return

    # 获取目录中的所有子目录
    subdirectories = []
    for entry in os.scandir(directory):
        if entry.is_dir():
            # 获取子目录的创建时间和路径
            creation_time = entry.stat().st_ctime
            subdirectories.append((creation_time, entry.path))

    # 按创建时间排序子目录，最新的子目录排在前面
    subdirectories.sort(reverse=True)

    # 保留最新的 k 个子目录，删除其余子目录
    for i in range(k, len(subdirectories)):
        _, subdirectory_path = subdirectories[i]
        try:
            # 使用 shutil.rmtree 递归删除子目录及其内容
            shutil.rmtree(subdirectory_path)
            print(f"已删除子目录: {subdirectory_path}")
        except Exception as e:
            print(f"删除子目录 {subdirectory_path} 时出错: {e}")


def delete_old_files(directory, k):
    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"目录 {directory} 不存在。")
        return

    # 获取目录中的所有文件
    files = []
    for entry in os.scandir(directory):
        if entry.is_file():
            # 获取文件的创建时间和文件路径
            creation_time = entry.stat().st_ctime
            files.append((creation_time, entry.path))

    # 按创建时间排序文件，最新的文件排在前面
    files.sort(reverse=True)

    # 保留最新的 k 个文件，删除其余文件
    for i in range(k, len(files)):
        _, file_path = files[i]
        try:
            os.remove(file_path)
            print(f"已删除文件: {file_path}")
        except Exception as e:
            print(f"删除文件 {file_path} 时出错: {e}")


def process_id_to_dist(args, hostname, proc_id, dist):
    def clear_files_with_suffix(directory, suffix):
        """
        清空指定目录下所有以给定后缀结尾的文件。
        :param directory: 要操作的目录路径
        :param suffix: 文件后缀，如 '.txt'
        """
        # 检查目录是否存在
        if not os.path.exists(directory):
            print(f"指定的目录 {directory} 不存在。")
            return
        # 遍历目录下的所有文件和文件夹
        for root, dirs, files in os.walk(directory):
            for file in files:
                # 检查文件名是否以指定后缀结尾
                if file.endswith(suffix):
                    file_path = os.path.join(root, file)
                    try:
                        # 以写入模式打开文件并立即关闭，从而清空文件内容
                        with open(file_path, 'w') as f:
                            pass
                        print(f"已清空文件: {file_path}")
                    except Exception as e:
                        print(f"清空文件 {file_path} 时出错: {e}")

    def get_ranks_by_extension(directory, extension):
        """
        获取指定目录下以指定字符串结尾的文件的文件名列表
        :param directory: 要搜索的目录路径
        :param extension: 文件扩展名或要匹配的字符串
        :return: 符合条件的文件名列表
        """
        file_list = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(extension):
                    file_list.append(file)
        sorted_file_list = sorted(file_list)
        ranks = [int(rank.split('=')[1]) for rank in sorted_file_list]
        return ranks

    clear_files_with_suffix(args.log_path, "_procs.json")
    dist.barrier()
    proc_name = hostname + "=" + proc_id + "="
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path, exist_ok=True)
    with open(os.path.join(args.log_path, f"{proc_name}_procs.json"), 'w') as f:
        json.dump([], f)
    dist.barrier()

    exp_ranks = get_ranks_by_extension(args.log_path, "_procs.json")

    return exp_ranks


def obtain_spe_names(directory):
    # 读取指定目录下的所有文件名
    filenames = os.listdir(directory)
    new_list = []
    for filename in filenames:
        # 按 '_' 分割文件名
        parts = filename.split('_')
        if len(parts) >= 2:
            # 取分割后的前两个子字符串并用 '_' 拼接
            new_name = "<" + '_'.join(parts[:2]) + ">"
            new_list.append(new_name)
    return new_list


if __name__ == "__main__":
    # 指定存放 checkpoint 的目录路径
    directory = "/home/share/huadjyin/home/liwenbo/projects/geno/models/results/jamba_ds_test/0_idx/test"  # 根据实际路径修改

    # 获取所有 .pth 文件的绝对路径
    checkpoint_paths = get_checkpoint_paths(directory)

    # 输出合并后的模型文件路径
    output_pth_path = "/home/share/huadjyin/home/liwenbo/projects/geno/models/results/jamba_ds_test/0_idx/pth/merged_pipeline_model.pth"

    # 调用合并函数
    merge_pipeline_parallel_checkpoints(checkpoint_paths, output_pth_path)

    # saved_state_dict = rename_state_dict(saved_state_dict, model)
