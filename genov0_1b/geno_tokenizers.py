import json
import os
from typing import Optional, Union, List
import regex as re
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging



logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}


class KMerTokenizer(PreTrainedTokenizer):
    """
    A tokenizer that splits the input text into k-mers using a sliding window approach.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file, k=3, add_prefix_space=False, **kwargs):
        # Load your vocab file (contains the k-mers)
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)  # Assuming vocab is a dict with k-mers as keys
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.k = k
        self.vocab_file = vocab_file
        self.add_prefix_space = add_prefix_space

        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return len(self.encoder)

    def _get_kmers(self, text):
        """ Generate k-mers from text using a sliding window. """
        kmer_tokens = []
        for i in range(len(text) - self.k + 1):
            kmer = text[i:i + self.k]
            kmer_tokens.append(kmer)
        return kmer_tokens

    def get_vocab(self):
        # 返回你的字典（词汇表）
        return self.encoder

    @staticmethod
    def _split_string_by_substring(input_string, special_substrings=("<cds_extraction>", "<cds_translation>")):
        for special_substring in special_substrings:
            index = input_string.find(special_substring)
            if index != -1:
                end_index = index + len(special_substring)
                first_part = input_string[:end_index]
                second_part = input_string[end_index:]
                return first_part, second_part
        first_part = input_string
        second_part = ""
        return first_part, second_part

    # type： kmers, bytes
    def _tokenize(self, text, encode_type="kmers"):
        if self.add_prefix_space:
            text = " " + text

        predefined_token_pattern = r"(<[^>]+>)"

        # 使用 re.split 保留预定义 token，得到交错的纯文本和预定义 token
        segments = re.split(predefined_token_pattern, text)

        token_ids = []

        for segment in segments:
            if not segment:
                continue
            if re.fullmatch(predefined_token_pattern, segment):
                # 是预定义 token，直接转为 ID（小写）
                token_id = self.encoder.get(segment.lower(), self.encoder.get(self.unk_token))
                token_ids.append(token_id)
            else:
                # 是普通文本，做 kmers 编码
                if encode_type == "kmers":
                    kmer_tokens = self._get_kmers(segment)
                    token_ids.extend([self.encoder.get(kmer, self.encoder.get(self.unk_token)) for kmer in kmer_tokens])
                elif encode_type == "bytes":
                    token_ids.extend([self.encoder.get(ch, self.encoder.get(self.unk_token)) for ch in segment])
                else:
                    kmer_tokens = self._get_kmers(segment)
                    token_ids.extend([self.encoder.get(kmer, self.encoder.get(self.unk_token)) for kmer in kmer_tokens])

        return token_ids

    def _convert_token_to_id(self, token):
        """ Converts a token to an ID using the vocab. """
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """ Converts an index to a token using the vocab. """
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a list of tokens back to a string. """
        return "".join(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """ Add special tokens (e.g., <BOS>, <EOS>) to the input. """
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]  # Add BOS and EOS tokens
        if token_ids_1:
            output += token_ids_1
        return output

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """ Get special tokens mask. """
        return [1] * len(token_ids_0)  # Example: mark all tokens as special for now.

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        if not os.path.isdir(save_directory):
            raise ValueError(f"Vocabulary path ({save_directory}) should be a directory")

        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.encoder, f, indent=2, sort_keys=True, ensure_ascii=False)

        return vocab_file

    def encode(self, text: str, **kwargs):
        if text.isupper():
            return self._tokenize(text, "bytes")
        encode_type = kwargs.get('encode_type', "kmers")
        geno_text, aa_text = self._split_string_by_substring(text)
        token_ids = []
        if geno_text:
            token_ids += self._tokenize(geno_text.lower(), encode_type)
        if aa_text:
            token_ids += self._tokenize(aa_text.upper(), "bytes")
        return token_ids

    def add_tokens(self, new_tokens: Union[str, List[str]], special_tokens: bool = False) -> int:
        if isinstance(new_tokens, str):
            new_tokens = [new_tokens]
        elif not isinstance(new_tokens, list):
            raise ValueError("new_tokens 应该是 str 或 List[str]")

        added = 0
        current_vocab_size = len(self.encoder)
        new_tokens = [token for token in new_tokens]

        for token in new_tokens:
            if token in self.encoder:
                continue  # 已存在，跳过
            new_id = current_vocab_size + added
            self.encoder[token] = new_id
            self.decoder[new_id] = token
            added += 1

        with open(self.vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.encoder, f, indent=2, sort_keys=True, ensure_ascii=False)

        return added

    def decode(self, token_ids: list, **kwargs):
        decoded_sequence = ""
        kmer_tokens = [self.decoder.get(token_id, self.unk_token) for token_id in token_ids]

        # 使用一个变量来跟踪滑动窗口的偏移量
        window_position = 0
        for token in kmer_tokens:
            if "<" in token and ">" in token:  # 处理特殊标记
                decoded_sequence += token  # 直接加入特殊标记
                window_position = 0
                continue  # 跳过继续处理下一个 token

            # 对于普通的 k-mer，恢复原始的序列部分
            if window_position == 0:
                # 这是第一个 token，我们直接添加
                decoded_sequence += token
                # 更新窗口位置
                window_position = 1
            else:
                # 这是一个滑动窗口 token，去掉窗口中已存在的前缀字符，添加新的字符
                decoded_sequence += token[-1]  # 只添加当前 token 的最后一个字符

        return decoded_sequence

    def save_pretrained(self, save_directory: str, filename_prefix: Optional[str] = None):
        os.makedirs(save_directory, exist_ok=True)

        # 保存词汇表
        vocab_file = self.save_vocabulary(save_directory, filename_prefix)

        # 保存 special_tokens_map.json
        special_tokens_map = {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token
        }
        with open(os.path.join(save_directory, "special_tokens_map.json"), "w", encoding="utf-8") as f:
            json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)

        # 保存 added_tokens.json
        added_tokens = [self.pad_token, self.unk_token]  # Add any other special tokens
        with open(os.path.join(save_directory, "added_tokens.json"), "w", encoding="utf-8") as f:
            json.dump({"added_tokens": added_tokens}, f, ensure_ascii=False, indent=4)

        # 保存 tokenizer_config.json
        tokenizer_config = {
            "added_tokens_decoder": {
                str(self.encoder.get(self.pad_token_id)): {
                    "content": self.pad_token,
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                str(self.encoder.get(self.unk_token_id)): {
                    "content": self.unk_token,
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                }
            },
            "clean_up_tokenization_spaces": True,
            "model_max_length": 10100,  # Or some reasonable value
            "pad_token": self.pad_token,
            "tokenizer_class": self.__class__.__name__,
            "unk_token": self.unk_token
        }
        with open(os.path.join(save_directory, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=4)

        return vocab_file

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        vocab_file = os.path.join(pretrained_model_name_or_path, "vocab.json")
        merges_file = os.path.join(pretrained_model_name_or_path, "merges.txt")

        if not os.path.exists(vocab_file):
            raise ValueError(f"Vocabulary file '{vocab_file}' not found at path {pretrained_model_name_or_path}.")

        return cls(k=int(pretrained_model_name_or_path[-1]),
                   vocab_file=vocab_file,
                   merges_file=merges_file,
                   unk_token="<unk>",
                   pad_token="<pad>")
