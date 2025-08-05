import os
import json
from .geno_models_moba import MGenoConfig as JambaConfig
from .geno_models_moba import MGenoTasked as Jamba
from .geno_tokenizers import KMerTokenizer
from .geno_utils import rename_state_dict
from moba import register_moba, MoBAConfig
from transformers import LogitsProcessorList, MinLengthLogitsProcessor
from transformers.generation.logits_process import LogitsProcessor
import torch


class RestrictTokensProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = torch.tensor(allowed_token_ids)

    def __call__(self, input_ids, scores):
        # 禁用所有不在 allowed_token_ids 中的 token（设为非常大的负数）
        mask = torch.full_like(scores, float('-inf'))
        mask[:, self.allowed_token_ids] = scores[:, self.allowed_token_ids]
        return mask

class BlockTokenProcessor(LogitsProcessor):
    def __init__(self, banned_token_ids):
        self.banned_token_ids = set(banned_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 将不允许的 token 的 logits 设置为 -inf
        for token_id in self.banned_token_ids:
            scores[:, token_id] = float("-inf")
        return scores

class GenoModel:
    def __init__(self, local_path,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 model_name="model.pth",
                 tokenizer_file="tokenizer_k6",
                 config="params_moba.config",
                 token_task_type="<token_task>",
                 target_labels="<t_mrna><t_cds><t_lnc_rna><unk_token>"):
        self.device = device
        config_f = os.path.join(local_path, config)
        with open(config_f, 'r') as f:
            config = json.load(f)
        geno_config = JambaConfig(
            dim=config["dim"],
            depth=config["depth"],
            num_tokens=config["num_tokens"],
            d_state=config["d_state"],
            d_conv=config["d_conv"],
            heads=config["heads"],
            num_experts=config["num_experts"],
            num_experts_per_token=config["num_experts_per_token"],
            expand=config["expand"],
            weights=config["weights"],
            max_token_len=config["max_token_len"],
            attn_class=config["attn_class"],
        )
        register_moba(MoBAConfig(config["moba_chunk_size"], config["moba_topk"]))
        self.token_task_type = token_task_type
        self.tokenizer = KMerTokenizer.from_pretrained(os.path.join(local_path, tokenizer_file))
        task_token_map = {
            self.tokenizer.encode('<token_task>')[0]: 1,
            self.tokenizer.encode('<conservation_task>')[0]: 2,
            self.tokenizer.encode('<token_task_cds>')[0]: 3,
            self.tokenizer.encode('<token_task_exon>')[0]: 4,
            self.tokenizer.encode('<token_task_gene>')[0]: 5,
            self.tokenizer.encode('<token_task_lnc_rna>')[0]: 6,
            self.tokenizer.encode('<token_task_mrna>')[0]: 7,
            self.tokenizer.encode('<token_task_ncrna>')[0]: 8,
            self.tokenizer.encode('<token_task_rrna>')[0]: 9,
            self.tokenizer.encode('<token_task_trna>')[0]: 10
        }
        self.model = self.load_model(geno_config=geno_config,
                                     pth_f=os.path.join(local_path, model_name),
                                     task_token_map=task_token_map).to(self.device)
        self.target_labels = target_labels


    def load_model(self, geno_config, pth_f, task_token_map):
        model = Jamba(geno_config, task_token_id_to_name=task_token_map)
        state_dict = torch.load(pth_f)
        try:
            state_dict = rename_state_dict(state_dict, model)
            missing, unexpected = model.load_state_dict(state_dict)
        except Exception as e:
            missing, unexpected = model.load_state_dict(state_dict)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        return model

    def infer(self, inputs):
        outputs = self.model(input_ids=inputs)
        return outputs

    def hidden_states_for_token_cls(self, seq_data):
        start_idx_list = []
        if not isinstance(seq_data, list):
            seq_data = seq_data + self.token_task_type + seq_data + self.token_task_type
            input_ids = torch.tensor(
                self.tokenizer.encode(seq_data),
                dtype=torch.int,
            ).unsqueeze(0).to(self.device)
            start_idx = int(len(input_ids[0]) / 2)
            start_idx_list.append(start_idx)
            output_hidden_states = self.model(input_ids=input_ids).hidden_states
        else:
            input_ids_batch = []
            for seq in seq_data:
                seq = seq + self.token_task_type + seq + self.token_task_type
                input_ids = self.tokenizer.encode(seq)
                start_idx = int(len(input_ids) / 2)
                start_idx_list.append(start_idx)
                input_ids_batch.append(input_ids)
            input_ids_batch = self.pad_sequences(input_ids_batch, pad_id=self.tokenizer.pad_token_id)
            # print("input_ids_batch type:", type(input_ids_batch))
            # print("Shape (before tensor):", len(input_ids_batch), len(input_ids_batch[0]))
            # print("Sample row:", input_ids_batch[0])
            input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long).to(self.device)
            self.model.eval()
            self.model.half()
            # print(f"{self.model.device}, {self.device}")
            # self.model.to(self.device)
            output_hidden_states = self.model(input_ids=input_ids_batch).hidden_states
            # print("TEST:", output_hidden_states.shape)
        return output_hidden_states, start_idx_list

    def slice_tensor_by_indices(self, tensor, start_indices):
        """
        基于起始索引列表对三维张量进行截取

        参数:
        - tensor: 形状为[batch_size, seq_len, hidden_size]的张量
        - start_indices: 每个batch样本的起始索引列表，长度为batch_size

        返回:
        - 截取后的张量，形状为[batch_size, new_seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = tensor.shape
        result = []

        for i in range(batch_size):
            start_idx = start_indices[i]
            max_len = max(start_indices)
            # 截取从start_idx到倒数第二个索引的部分
            sliced = tensor[i, start_idx: max_len + start_idx - 1, :]
            result.append(sliced)
        # result = self.pad_sequences(result, pad_id=self.tokenizer.pad_token_id)
        # 将各个batch的结果拼接回一个张量
        return torch.stack(result)

    def get_hidden_states_for_token_cls(self, seq_data):
        ref_hidden_states, start_idx_list = self.hidden_states_for_token_cls(seq_data)
        output_hidden_states = self.slice_tensor_by_indices(ref_hidden_states, start_idx_list)
        return output_hidden_states

    @staticmethod
    def pad_sequences(sequences, max_length=None, pad_id=0):
        """
        对二维列表进行padding，使其所有行达到相同长度

        参数:
        - sequences: 待填充的二维列表
        - max_length: 目标长度，若为None则取最长行的长度
        - pad_id: 填充值，默认为0

        返回:
        - 填充后的二维列表
        """
        # 如果未指定max_length，则计算最长行的长度
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)

        # 对每个序列进行填充
        padded_sequences = []
        for seq in sequences:
            # 如果序列长度小于max_length，则填充到max_length
            if len(seq) < max_length:
                padded_seq = seq + [pad_id] * (max_length - len(seq))
            # 如果序列长度大于max_length，则截断（通常不需要，LLM输入有固定上限）
            else:
                padded_seq = seq[:max_length]
            padded_sequences.append(padded_seq)

        return padded_sequences

    def pred_labels_for_token_cls(self, seq_data, labels=None):
        if labels is None:
            start_idx_list = []
            if not isinstance(seq_data, list):
                seq_data = seq_data + self.token_task_type + seq_data + self.token_task_type
                input_ids = torch.tensor(
                    self.tokenizer.encode(seq_data),
                    dtype=torch.int,
                ).unsqueeze(0).to(self.device)
                start_idx = int(len(input_ids[0]) / 2)
                start_idx_list.append(start_idx)
                output_logits = self.model(input_ids=input_ids).logits
            else:
                input_ids_batch = []
                for seq in seq_data:
                    seq = seq + self.token_task_type + seq + self.token_task_type
                    input_ids = self.tokenizer.encode(seq)
                    start_idx = int(len(input_ids) / 2)
                    start_idx_list.append(start_idx)
                    input_ids_batch.append(input_ids)
                input_ids_batch = self.pad_sequences(input_ids_batch, pad_id=self.tokenizer.pad_token_id)
                # print("input_ids_batch type:", type(input_ids_batch))
                # print("Shape (before tensor):", len(input_ids_batch), len(input_ids_batch[0]))
                # print("Sample row:", input_ids_batch[0])
                input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long).to(self.device)
                self.model.eval()
                self.model.half()
                output_logits = self.model(input_ids=input_ids_batch).logits
                # print("TEST:", output_hidden_states.shape)
        else:
            start_idx_list = []
            if not isinstance(seq_data, list):
                seq_data = seq_data + self.token_task_type + seq_data + self.token_task_type
                input_ids = torch.tensor(
                    self.tokenizer.encode(seq_data),
                    dtype=torch.int,
                ).unsqueeze(0).to(self.device)
                start_idx = int(len(input_ids[0]) / 2)
                start_idx_list.append(start_idx)
                output_logits = self.model(input_ids=input_ids).logits
            else:
                input_ids_batch = []
                for seq in seq_data:
                    seq = seq + self.token_task_type + seq + self.token_task_type
                    input_ids = self.tokenizer.encode(seq)
                    start_idx = int(len(input_ids) / 2)
                    start_idx_list.append(start_idx)
                    input_ids_batch.append(input_ids)
                input_ids_batch = self.pad_sequences(input_ids_batch, pad_id=self.tokenizer.pad_token_id)
                # print("input_ids_batch type:", type(input_ids_batch))
                # print("Shape (before tensor):", len(input_ids_batch), len(input_ids_batch[0]))
                # print("Sample row:", input_ids_batch[0])
                input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long).to(self.device)
                self.model.eval()
                self.model.half()
                output_logits = self.model(input_ids=input_ids_batch).logits
                # print("TEST:", output_hidden_states.shape)
        return output_logits, start_idx_list

    def slice_logits_by_indices(self, tensor, start_indices, target_indices):
        """
        基于起始索引列表对三维张量进行截取

        参数:
        - tensor: 形状为[batch_size, seq_len, hidden_size]的张量
        - start_indices: 每个batch样本的起始索引列表，长度为batch_size

        返回:
        - 截取后的张量，形状为[batch_size, new_seq_len, hidden_size]
        """
        batch_size, seq_len, vocab_size = tensor.shape
        result = []

        for i in range(batch_size):
            start_idx = start_indices[i]
            max_len = max(start_indices)
            # 截取从start_idx到倒数第二个索引的部分
            target_probs  = tensor[i, start_idx: -1, target_indices]
            # 对每个位置，获取最大概率的相对索引
            max_prob_token_rel_idx = torch.argmax(target_probs, dim=-1)
            # 映射回全局词典索引
            max_prob_token_idxs = (torch.tensor(target_indices).to(self.device)[max_prob_token_rel_idx.to(self.device)]).tolist()
            # print(max_prob_token_idxs)
            result.append(max_prob_token_idxs)
        result = self.pad_sequences(result, pad_id=self.tokenizer.pad_token_id)
        #print(result)
        result_tokens = [self.tokenizer.decode(res) for res in result]
        # 将各个batch的结果拼接回一个张量
        return torch.tensor(result), result_tokens

    def get_pred_labels_for_token_cls(self, seq_data, labels=None):
        # target_indices = self.tokenizer.encode("<0><1><2><3><4><5><6><7><8>")
        target_indices = self.tokenizer.encode(self.target_labels)
        # target_indices = range(15710)
        output_logits, start_idx_list = self.pred_labels_for_token_cls(seq_data, labels)
        pred_ids, pred_tokens = self.slice_logits_by_indices(output_logits, start_idx_list, target_indices)
        return pred_ids, pred_tokens



    def generate(self, input_seq,
                 allowed_tokens=None,
                 banned_tokens=None,
                 max_new_tokens=None,
                 max_length=100,
                 num_beams=5,
                 top_k=10,
                 top_p=0.9,
                 no_repeat_ngram_size=2,
                 early_stopping=True):

        if allowed_tokens is not None:
            # 设置只允许某些token被生成（比如只允许数字）
            allowed_token_ids = [self.tokenizer.encode(str(i))[0] for i in allowed_tokens]
            # 创建 processor list
            logits_processor = LogitsProcessorList()
            logits_processor.append(RestrictTokensProcessor(allowed_token_ids))
        elif banned_tokens is not None:
            banned_token_ids = [tokenizer.encode(word)[0] for word in banned_tokens]
            logits_processor = LogitsProcessorList([
                BlockTokenProcessor(banned_token_ids)
            ])
        else:
            logits_processor = None

        input_ids = torch.tensor(
            self.tokenizer.encode(input_seq),
            dtype=torch.int,
        ).unsqueeze(0).to(self.device)

        response_seq = self.model.generate(input_ids,
                                           logits_processor=logits_processor,
                                           #max_length=max_length,
                                           max_new_tokens=max_new_tokens,
                                           num_beams=num_beams,
                                           top_k=top_k,
                                           top_p=top_p,
                                           no_repeat_ngram_size=no_repeat_ngram_size,
                                           early_stopping=early_stopping)

        # 解码生成的序列
        generated_seq = self.tokenizer.decode(response_seq.squeeze().tolist())
        return generated_seq


if __name__ == "__main__":
    model = "model_v2_stage5.3.pth"
    tokenizer = "tokenizer_v2_k6"
    config = "params_moba_cds.config"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CKPT_PATH = "/home/share/huadjyin/home/liwenbo/geno_evaluate/1b-test/ckpt1"
    gemo_model = GenoModel(CKPT_PATH,
                           device=DEVICE,
                           model_name=model,
                           tokenizer_file=tokenizer,
                           config=config)

    sequence = 'acgtatcgatcg'
    print("input ids: ", gemo_model.tokenizer.encode(sequence))

    input_ids = torch.tensor(
        gemo_model.tokenizer.encode(sequence),
        dtype=torch.int,
    ).unsqueeze(0).to(device)
    print("origin seq: ", gemo_model.tokenizer.decode(input_ids[0].tolist()))
    print(input_ids.shape)
    gemo_model.model = gemo_model.model.to(device)
    outputs = gemo_model.infer(input_ids)

    logits = outputs.logits
    print('Shape (batch, length, vocab): ', logits.shape)

    hidden_states = outputs.hidden_states
    print('hidden_states: ', hidden_states.shape)

    # token_cls_hidden_states = gemo_model.get_hidden_states_for_token_cls(sequence)
    # print('token_cls_hidden_states: ', token_cls_hidden_states.shape)

    predict_cls_ids, predict_cls_labels = gemo_model.get_pred_labels_for_token_cls(sequence)
    print('predict_cls_labels: ', predict_cls_labels)

    # generated_sequence = gemo_model.generate(sequence, max_length=100, num_beams=5, top_k=10, top_p=0.9)
    # print(f'generated_sequence ({len(generated_sequence)}): ', generated_sequence)
