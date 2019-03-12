from typing import Dict, List
import collections
import logging
import math

import torch
from overrides import overrides
from pytorch_pretrained_bert import BertForQuestionAnswering as HuggingFaceBertQA
from pytorch_pretrained_bert import BertConfig
from pytorch_pretrained_bert.tokenization import BasicTokenizer

from allennlp.common import JsonDict
from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


BERT_LARGE_CONFIG = {"attention_probs_dropout_prob": 0.1,
                     "hidden_act": "gelu",
                     "hidden_dropout_prob": 0.1,
                     "hidden_size": 1024,
                     "initializer_range": 0.02,
                     "intermediate_size": 4096,
                     "max_position_embeddings": 512,
                     "num_attention_heads": 16,
                     "num_hidden_layers": 24,
                     "type_vocab_size": 2,
                     "vocab_size": 30522
                    }

BERT_BASE_CONFIG = {"attention_probs_dropout_prob": 0.1,
                    "hidden_act": "gelu",
                    "hidden_dropout_prob": 0.1,
                    "hidden_size": 768,
                    "initializer_range": 0.02,
                    "intermediate_size": 3072,
                    "max_position_embeddings": 512,
                    "num_attention_heads": 12,
                    "num_hidden_layers": 12,
                    "type_vocab_size": 2,
                    "vocab_size": 30522
                   }


@Model.register('bert_for_qa')
class BertForQuestionAnswering(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 bert_model_type: str,
                 pretrained_archive_path: str,
                 null_score_difference_threshold: float,
                 n_best_size: int = 20,
                 max_answer_length: int = 30) -> None:
        super().__init__(vocab)
        if bert_model_type == "bert_base":
            config_to_use = BERT_BASE_CONFIG
        elif bert_model_type == "bert_large":
            config_to_use = BERT_LARGE_CONFIG
        else:
            raise RuntimeError(f"`bert_model_type` should either be \"bert_large\" or \"bert_base\"")
        config = BertConfig(vocab_size_or_config_json_file=config_to_use["vocab_size"],
                            hidden_size=config_to_use["hidden_size"],
                            num_hidden_layers=config_to_use["num_hidden_layers"],
                            num_attention_heads=config_to_use["num_attention_heads"],
                            intermediate_size=config_to_use["intermediate_size"],
                            hidden_act=config_to_use["hidden_act"],
                            hidden_dropout_prob=config_to_use["hidden_dropout_prob"],
                            attention_probs_dropout_prob=config_to_use["attention_probs_dropout_prob"],
                            max_position_embeddings=config_to_use["max_position_embeddings"],
                            type_vocab_size=config_to_use["type_vocab_size"],
                            initializer_range=config_to_use["initializer_range"])
        self.bert_qa_model = HuggingFaceBertQA(config)
        self._loaded_qa_weights = False
        self._pretrained_archive_path = pretrained_archive_path
        self._null_score_difference_threshold = null_score_difference_threshold
        self._n_best_size = n_best_size
        self._max_answer_length = max_answer_length

    @overrides
    def forward(self,  # type: ignore
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                tokens: List[str],
                document_tokens: List[str],
                token_to_original_map: Dict[int, int],
                token_is_max_context: Dict[int, bool]) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        if not self._loaded_qa_weights and self.training:
            self.bert_qa_model = HuggingFaceBertQA.from_pretrained(self._pretrained_archive_path)
            self._loaded_qa_weights = True
        start_logits, end_logits = self.bert_qa_model(torch.stack(input_ids),
                                                      torch.stack(token_type_ids),
                                                      torch.stack(attention_mask))
        output_dict = {"start_logits": start_logits,
                       "end_logits": end_logits,
                       "tokens": tokens,
                       "document_tokens": document_tokens,
                       "token_to_original_map": token_to_original_map,
                       "token_is_max_context": token_is_max_context}
        if self.training:
            loss = torch.sum(start_logits) * 0.0
            output_dict["loss"] = loss
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_start_logits = output_dict["start_logits"]
        batch_end_logits = output_dict["end_logits"]
        batch_tokens = output_dict["tokens"]
        batch_document_tokens = output_dict["document_tokens"]
        batch_token_map = output_dict["token_to_original_map"]
        batch_token_is_max_context = output_dict["token_is_max_context"]
        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                "PrelimPrediction",
                ["start_index", "end_index", "start_logit", "end_logit"])
        predictions: List[str] = []
        nbest_info: JsonDict = []
        for start_logits, end_logits, tokens, document_tokens, token_map, token_is_max_context in zip(
                batch_start_logits,
                batch_end_logits,
                batch_tokens,
                batch_document_tokens,
                batch_token_map,
                batch_token_is_max_context):
            prelim_predictions = []
            score_null = 1000000  # large and positive
            null_start_logit = 0  # the start logit at the slice with min null score
            null_end_logit = 0  # the end logit at the slice with min null score
            start_indexes = self._get_best_indexes(start_logits, self._n_best_size)
            end_indexes = self._get_best_indexes(end_logits, self._n_best_size)
            feature_null_score = start_logits[0] + end_logits[0]
            if feature_null_score < score_null:
                score_null = feature_null_score
                null_start_logit = start_logits[0]
                null_end_logit = end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(tokens):
                        continue
                    if end_index >= len(tokens):
                        continue
                    if start_index not in token_map:
                        continue
                    if end_index not in token_map:
                        continue
                    if not token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > self._max_answer_length:
                        continue
                    prelim_predictions.append(
                            _PrelimPrediction(
                                    start_index=start_index,
                                    end_index=end_index,
                                    start_logit=start_logits[start_index],
                                    end_logit=end_logits[end_index]))
            prelim_predictions.append(
                    _PrelimPrediction(
                            start_index=0,
                            end_index=0,
                            start_logit=null_start_logit,
                            end_logit=null_end_logit))
            prelim_predictions = sorted(
                    prelim_predictions,
                    key=lambda x: (x.start_logit + x.end_logit),
                    reverse=True)

            _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                    "NbestPrediction", ["text", "start_logit", "end_logit"])

            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= self._n_best_size:
                    break
                if pred.start_index > 0:  # this is a non-null prediction
                    tok_tokens = tokens[pred.start_index:(pred.end_index + 1)]
                    orig_doc_start = token_map[pred.start_index]
                    orig_doc_end = token_map[pred.end_index]
                    orig_tokens = document_tokens[orig_doc_start:(orig_doc_end + 1)]
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens)

                    final_text = self._get_final_text(tok_text,
                                                      orig_text,
                                                      do_lower_case=True)
                    if final_text in seen_predictions:
                        continue

                    seen_predictions[final_text] = True
                else:
                    final_text = ""
                    seen_predictions[final_text] = True

                nbest.append(
                        _NbestPrediction(
                                text=final_text,
                                start_logit=pred.start_logit,
                                end_logit=pred.end_logit))
            # if we didn't include the empty option in the n-best, include it
            if "" not in seen_predictions:
                nbest.append(
                        _NbestPrediction(
                                text="!!NO ANSWER!!",
                                start_logit=null_start_logit,
                                end_logit=null_end_logit))
                # In very rare edge cases we could only have single null prediction.
                # So we just create a nonce prediction in this case to avoid failure.
                if len(nbest) == 1:
                    nbest.insert(0,
                                 _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))
            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(
                        _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

            assert len(nbest) >= 1

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)
                if not best_non_null_entry:
                    if entry.text:
                        best_non_null_entry = entry

            probs = self._compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                nbest_json.append(output)

            assert len(nbest_json) >= 1

            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                    best_non_null_entry.end_logit)
            if score_diff > self._null_score_difference_threshold:
                predictions.append("!!NO ANSWER!!")
            else:
                predictions.append(best_non_null_entry.text)
            nbest_info.append(nbest_json)
        output_dict["predictions"] = predictions
        output_dict["nbest_info"] = nbest_info
        return output_dict

    @staticmethod
    def _get_best_indexes(logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = [index_score_pair[0] for index_score_pair in index_and_score[:n_best_size]]
        return best_indexes

    @staticmethod
    def _get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
        """Project the tokenized prediction back to the original text."""

        # When we created the data, we kept track of the alignment between original
        # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
        # now `orig_text` contains the span of our original text corresponding to the
        # span that we predicted.
        #
        # However, `orig_text` may contain extra characters that we don't want in
        # our prediction.
        #
        # For example, let's say:
        #   pred_text = steve smith
        #   orig_text = Steve Smith's
        #
        # We don't want to return `orig_text` because it contains the extra "'s".
        #
        # We don't want to return `pred_text` because it's already been normalized
        # (the SQuAD eval script also does punctuation stripping/lower casing but
        # our tokenizer does additional normalization like stripping accent
        # characters).
        #
        # What we really want to return is "Steve Smith".
        #
        # Therefore, we have to apply a semi-complicated alignment heruistic between
        # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
        # can fail in certain cases in which case we just return `orig_text`.

        def _strip_spaces(text):
            ns_chars = []
            ns_to_s_map = collections.OrderedDict()
            for (i, char) in enumerate(text):
                if char == " ":
                    continue
                ns_to_s_map[len(ns_chars)] = i
                ns_chars.append(char)
            ns_text = "".join(ns_chars)
            return (ns_text, ns_to_s_map)

        # We first tokenize `orig_text`, strip whitespace from the result
        # and `pred_text`, and check if they are the same length. If they are
        # NOT the same length, the heuristic has failed. If they are the same
        # length, we assume the characters are one-to-one aligned.
        tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

        tok_text = " ".join(tokenizer.tokenize(orig_text))

        start_position = tok_text.find(pred_text)
        if start_position == -1:
            if verbose_logging:
                logger.info(f"Unable to find text: '{pred_text}' in '{orig_text}'")
            return orig_text
        end_position = start_position + len(pred_text) - 1

        (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
        (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            if verbose_logging:
                logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
            return orig_text

        # We then project the characters in `pred_text` back to `orig_text` using
        # the character-to-character alignment.
        tok_s_to_ns_map = {}
        for (i, tok_index) in tok_ns_to_s_map.items():
            tok_s_to_ns_map[tok_index] = i

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            if verbose_logging:
                logger.info("Couldn't map start position")
            return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            if verbose_logging:
                logger.info("Couldn't map end position")
            return orig_text

        output_text = orig_text[orig_start_position:(orig_end_position + 1)]
        return output_text

    @staticmethod
    def _compute_softmax(scores):
        """Compute softmax probability over raw logits."""
        if not scores:
            return []

        max_score = None
        for score in scores:
            if max_score is None or score > max_score:
                max_score = score

        exp_scores = []
        total_sum = 0.0
        for score in scores:
            exp_score = math.exp(score - max_score)
            exp_scores.append(exp_score)
            total_sum += exp_score

        probs = []
        for score in exp_scores:
            probs.append(score / total_sum)
        return probs
