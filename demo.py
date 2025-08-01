import torch
from genov0_1b.geno_infer_moba import GenoModel
from sklearn.metrics import matthews_corrcoef
import json
from loguru import logger

if __name__ == "__main__":
    model = "model_v2_stage6.18.plus4.pth"
    tokenizer = "tokenizer_v2_k6"
    config = "params_moba_cds.config"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CKPT_PATH = "/home/share/huadjyin/home/liwenbo/geno_evaluate/1b-test/ckpt1"
    # target_labels = "<t_mrna><t_cds><t_lnc_rna><unk_token>"
    target_labels = "<t_exon><t_gene><t_mrna><t_cds><t_lnc_rna><t_trna><unk_token>"
    # target_labels = '<t_mrna><t_ncrna><t_exon><t_cds><unk_token>'
    gemo_model = GenoModel(CKPT_PATH,
                           device=DEVICE,
                           model_name=model,
                           tokenizer_file=tokenizer,
                           config=config,
                           target_labels=target_labels)

    sequence = 'acgtatcgatcg'
    print("input ids: ", gemo_model.tokenizer.encode(sequence))

    input_ids = torch.tensor(
        gemo_model.tokenizer.encode(sequence),
        dtype=torch.int,
    ).unsqueeze(0).to(DEVICE)
    print("origin seq: ", gemo_model.tokenizer.decode(input_ids[0].tolist()))
    print(input_ids.shape)
    gemo_model.model = gemo_model.model.to(DEVICE)
    outputs = gemo_model.infer(input_ids)

    # logits
    logits = outputs.logits
    print('Shape (batch, length, vocab): ', logits.shape)

    # embeddings
    hidden_states = outputs.hidden_states
    print('hidden_states (batch, length, embedding_dim): ', hidden_states.shape)
    # print('hidden_states (batch, length, embedding_dim): ', hidden_states[0, -1, :200])

    # generation
    # generated_sequence = gemo_model.generate(sequence, max_length=100, num_beams=5, top_k=10, top_p=0.9)
    # print(f'generated_sequence ({len(generated_sequence)}): ', generated_sequence)

    # sequence = ['acgtatcgatcg', 'acgtatcgatcgaa', 'acgtatcgatcgaaa']
    # token_cls_hidden_states = gemo_model.get_hidden_states_for_token_cls(sequence)
    # print('token_cls_hidden_states: ', token_cls_hidden_states.shape)

    def read_jsonl_to_lists(file_path):
        """
        读取JSONL文件，提取每个对象的sequence和labels字段为两个列表，保持索引一一对应

        参数:
            file_path (str): JSONL文件路径

        返回:
            sequences (list): 包含所有sequence字段的列表
            labels (list): 包含所有labels字段的列表
        """
        sequences = []
        labels = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 去除行首尾空白字符（处理可能的换行符/空格）
                line = line.strip()
                if not line:  # 跳过空行
                    continue

                try:
                    # 解析JSON对象
                    data = json.loads(line)
                    # 检查必填字段是否存在
                    if 'sequence' not in data or 'labels' not in data:
                        raise ValueError(f"JSON对象缺少必填字段，行内容：{line}")
                    # print(data['sequence'])
                    # print(data['labels'])
                    assert len(data['sequence']) == len(data['labels']), "miss matching between seq and labels"
                    sequences.append(data['sequence'][:8000])
                    labels.append(data['labels'][:8000])

                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON解析错误，行内容：{line}\n错误信息：{str(e)}")

        return sequences, labels


    def split_labels_string(label_str):
        import re
        result = re.findall(r'<[^<>]+>', label_str)
        return [int(item.replace("<", "").replace(">", "")) for item in result]

    def split_labels_string2(label_str):
        import re
        result = re.findall(r'<[^<>]+>', label_str)
        return [item for item in result]


    # sequences, labels = read_jsonl_to_lists("/home/share/huadjyin/home/liwenbo/projects/geno/data/genomics_seq/gene_finding/test.jsonl")
    # sequences, labels = read_jsonl_to_lists("/home/share/huadjyin/home/liwenbo/projects/geno/data/genomics_seq/train_stage3.6/5k_4/test.jsonl")
    # sequences, labels = read_jsonl_to_lists("/home/share/huadjyin/home/liwenbo/projects/geno/data/genomics_seq/train_stage3.6/5k_8.dataset/test_7l.jsonl")
    sequences, labels = read_jsonl_to_lists("/home/share/huadjyin/home/liwenbo/projects/geno/data/genomics_seq/train_stage3.6/5k_8.dataset/test_filtered.jsonl")


    total_preds = []
    total_labels = []

    from tqdm import tqdm
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    count = 0
    for test_seq, test_lab in tqdm(zip(sequences, labels), desc="处理样本", total=len(sequences)):
        if count > 10000:
            break
        with torch.no_grad():
            pred_ids, pred_tokens = gemo_model.get_pred_labels_for_token_cls(test_seq)
        del pred_ids
        torch.cuda.empty_cache()
        gt = [item for item in test_lab]
        rs = split_labels_string2(pred_tokens[0])
        total_labels += gt[5:]
        total_preds += rs
        count += 1

    label_set = sorted(list(set(total_labels)))

    # 计算多分类 F1 分数（假设为 3 类）
    # logger.info(total_labels)
    # logger.info(total_preds)
    f1_macro = f1_score(total_labels, total_preds, average='macro')  # 宏平均
    f1_micro = f1_score(total_labels, total_preds, average='micro')  # 微平均
    mcc = matthews_corrcoef(total_labels,total_preds)
    print(f"多分类宏平均 F1 分数: {f1_macro:.4f}")
    print(f"多分类微平均 F1 分数: {f1_micro:.4f}")
    print(f"MCC分数: {mcc:.4f}")

    precision, recall, f1, support = precision_recall_fscore_support(
        total_labels,
        total_preds,
        labels = label_set,
        average=None,  # 不平均，返回每个类别的指标
        zero_division=0  # 处理零除问题
    )
    # 输出每个标签的指标
    for label, p, r, f1_val, s in zip(label_set, precision, recall, f1, support):
        print(f"标签 {label}:")
        print(f"  精确率: {p:.4f}")
        print(f"  召回率: {r:.4f}")
        print(f"  F1分数: {f1_val:.4f}")
        print(f"  支持度(样本数): {s}")

