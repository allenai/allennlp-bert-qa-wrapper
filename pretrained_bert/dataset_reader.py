import json
import logging
import collections
from typing import List

import torch
from overrides import overrides
from pytorch_pretrained_bert import BertTokenizer

from allennlp.common.file_utils import cached_path
from allennlp.data.fields import MetadataField
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("squad_for_pretrained_bert")
class SquadReaderForPretrainedBert(DatasetReader):
    def __init__(self,
                 pretrained_bert_model_file: str,
                 lazy: bool = False,
                 max_query_length: int = 64,
                 max_sequence_length: int = 384,
                 document_stride: int = 128) -> None:
        super().__init__(lazy)
        self._tokenizer = BertTokenizer.from_pretrained(pretrained_bert_model_file)
        self._max_query_length = max_query_length
        self._max_sequence_length = max_sequence_length
        self._document_stride = document_stride

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
            for entry in dataset:
                for paragraph in entry["paragraphs"]:
                    paragraph_text = paragraph["context"]
                    for question_answer in paragraph["qas"]:
                        question_text = question_answer["question"]
                        instance = self.text_to_instance(question_text=question_text,
                                                         paragraph_text=paragraph_text)
                        if instance is not None:
                            yield instance

    @staticmethod
    def _check_is_max_context(doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""

        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         paragraph_text: str) -> Instance:
        # pylint: disable=arguments-differ
        def is_whitespace(char):
            if char == " " or char == "\t" or char == "\r" or char == "\n" or ord(char) == 0x202F:
                return True
            return False

        doc_tokens: List[str] = []
        prev_is_whitespace = True
        for char in paragraph_text:
            if is_whitespace(char):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(char)
                else:
                    doc_tokens[-1] += char
                prev_is_whitespace = False
        query_tokens = self._tokenizer.tokenize(question_text)

        if len(query_tokens) > self._max_query_length:
            query_tokens = query_tokens[0:self._max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = self._tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = self._max_sequence_length - len(query_tokens) - 3

        # Different from original pytorch-pretrained-bert,
        # we don't use a sliding window approach here.
        # We just truncate the original doc to defined max_sequence_length.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        if start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))

        # We only select the first index of doc_spans here.
        doc_span_index = 0
        doc_span = doc_spans[0]

        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = self._check_is_max_context(doc_spans,
                                                        doc_span_index,
                                                        split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self._max_sequence_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self._max_sequence_length
        assert len(input_mask) == self._max_sequence_length
        assert len(segment_ids) == self._max_sequence_length
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        input_mask_tensor = torch.tensor(input_mask, dtype=torch.long)
        segment_ids_tensor = torch.tensor(segment_ids, dtype=torch.long)
        instance = Instance({"input_ids": MetadataField(input_ids_tensor),
                             "token_type_ids": MetadataField(segment_ids_tensor),
                             "attention_mask": MetadataField(input_mask_tensor),
                             "tokens": MetadataField(tokens),
                             "document_tokens": MetadataField(doc_tokens),
                             "token_to_original_map": MetadataField(token_to_orig_map),
                             "token_is_max_context": MetadataField(token_is_max_context)})
        # We truncate the original doc to defined max_sequence_length.
        # Here we only process the first part of doc_spans and return the result.
        return instance
