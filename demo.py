from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from transformers import AutoModel
from transformers import AutoTokenizer
import transformers


transformers.logging.set_verbosity_error()

"""-----------------Pre-processing input-----------------"""


def remove_special_characters(text):
    # Remove special characters
    text = re.sub(r"[^\w\s]", "", text)
    return text


def remove_dates(text):
    # Define a regular expression pattern to match dates in number format
    date_pattern = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"

    # Find all matches of the date pattern in the text
    matches = re.findall(date_pattern, text)

    # Remove the matched dates from the text
    for match in matches:
        text = text.replace(match, "")

    return text


def remove_timestamps_and_whitespace(text):
    # Remove timestamps
    text = re.sub(r"\d{1,2}:\d{2}(?::\d{2})?", "", text)
    # Remove date
    text = re.sub(
        r"\b(\d{1,2}\s+(?:tháng\s1|tháng\s2|tháng\s3|tháng\s4|tháng\s5|tháng\s6|tháng\s7|tháng\s8|tháng\s9|tháng\s10|tháng\s11|tháng\s12)\s+\d{2,4}|(?:tháng\s1|tháng\s2|tháng\s3|tháng\s4|tháng\s5|tháng\s6|tháng\s7|tháng\s8|tháng\s9|tháng\s10|tháng\s11|tháng\s12)\s+\d{1,2}\s+\d{2,4})|(?:tháng\s1|tháng\s2|tháng\s3|tháng\s4|tháng\s5|tháng\s6|tháng\s7|tháng\s8|tháng\s9|tháng\s10|tháng\s11|tháng\s12)\b",  # noqa: E501
        "",
        text,
    )
    # Remove extra white space
    text = re.sub(r"\s+", " ", text)
    # Also remove None as it appears in some context
    text = text.replace("none", "")
    return text.strip()


def get_stopword_list(stop_file_path):
    """load stop words"""

    with open(stop_file_path, "r", encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return list(frozenset(stop_set))


def remove_stopwords(text, stopwords):
    # Split the text into words
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stopwords]

    # Rejoin the words into a string
    text = " ".join(words)
    return text


def preprocess_text(text):
    # Lower casing
    processed_text = text.lower()

    # Removeal of special characters
    processed_text = remove_special_characters(processed_text)

    # Removal of timestamps and dates
    processed_text = remove_dates(processed_text)
    processed_text = remove_timestamps_and_whitespace(processed_text)

    return processed_text


"""-----------------Functions and classes for multi-tasking model-----------------"""


@dataclass
class Task:
    task_id: int
    name: str
    task_type: str
    num_labels: int


class SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_p=0.1):
        super().__init__()
        # super().__init__(config=PretrainedConfig())
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self._init_weights()

    def forward(self, sequence_output, pooled_output, labels=None, **kwargs):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if labels.dim() != 1:
                # Remove padding
                labels = labels[:, 0]

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.long().view(-1),
            )

        return logits, loss

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()


class MultiTaskModel(nn.Module):
    def __init__(self, encoder_name_or_path, tasks: List):
        super().__init__()
        # self.tasks = tasks

        self.encoder = AutoModel.from_pretrained(
            encoder_name_or_path,
            return_dict=False,
        )

        self.output_heads = nn.ModuleDict()
        for task in tasks:
            decoder = self._create_output_head(
                self.encoder.config.hidden_size,
                task,
            )
            # ModuleDict requires keys to be strings
            self.output_heads[str(task.task_id)] = decoder

    @staticmethod
    def _create_output_head(encoder_hidden_size: int, task):
        if task.task_type == "seq_classification":
            return SequenceClassificationHead(encoder_hidden_size, task.num_labels)
        else:
            raise NotImplementedError()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        task_ids=None,
        **kwargs,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[:2]

        unique_task_ids_list = torch.unique(task_ids).tolist()

        loss_list = []
        logits = None
        for unique_task_id in unique_task_ids_list:
            task_id_filter = task_ids == unique_task_id

            logits, task_loss = self.output_heads[str(unique_task_id)].forward(
                sequence_output[task_id_filter],
                pooled_output[task_id_filter],
                labels=None if labels is None else labels[task_id_filter],
                attention_mask=attention_mask[task_id_filter],
            )

            if labels is not None:
                loss_list.append(task_loss)

        # logits are only used for eval. and in case of eval the batch is not multi task
        # For training only the loss is used
        outputs = (logits, outputs[2:])

        if loss_list:
            loss = torch.stack(loss_list)
            outputs = (loss.mean(),) + outputs

        return outputs


def load_mtl_model(encoder, state_dict_path, tasks, use_cpu=False):
    # Load trained model by state_dict
    load_mtl_model = MultiTaskModel(encoder, tasks=tasks)

    if use_cpu is True:
        map_location = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            logger.warning("PyTorch is running on CPU")

            raise ValueError("No GPU available")

        map_location = torch.device("cuda")

    load_mtl_model.load_state_dict(
        torch.load(state_dict_path, map_location=map_location),
    )
    load_mtl_model.eval()

    return load_mtl_model


def postprocess_nli(
    model_outputs,
    label2id={"entailment": 0, "neutral": 1, "contradiction": 2},
    multi_label=False,
):
    candidate_labels = [outputs["candidate_label"] for outputs in model_outputs]
    sequences = [outputs["sequence"] for outputs in model_outputs]

    logits = np.concatenate([output["logits"].numpy() for output in model_outputs])
    N = logits.shape[0]
    n = len(candidate_labels)

    num_sequences = N // n
    reshaped_outputs = logits.reshape((num_sequences, n, -1))

    entailment_id = label2id["entailment"]
    contradiction_id = label2id["contradiction"]

    if multi_label or len(candidate_labels) == 1:
        # softmax over the entailment vs. contradiction dim for each label independently
        entail_contr_logits = reshaped_outputs[
            ...,
            [
                contradiction_id,
                entailment_id,
            ],
        ]
        scores = np.exp(entail_contr_logits) / np.exp(entail_contr_logits).sum(
            -1,
            keepdims=True,
        )
        scores = scores[..., 1]
    else:
        # softmax the "entailment" logits over all candidate labels
        entail_logits = reshaped_outputs[..., entailment_id]
        scores = np.exp(entail_logits) / np.exp(entail_logits).sum(-1, keepdims=True)

    top_inds = list(reversed(scores[0].argsort()))

    return {
        "sequence": sequences[0],
        "labels": [candidate_labels[i] for i in top_inds],
        "scores": scores[0, top_inds].tolist(),
    }


def sa_classifier(model: nn.Module, tokenizer, text: str, sa_task_id):
    # Inputs for sentiment analysis
    inputs_sa = tokenizer(
        text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs_sa = model(**inputs_sa, task_ids=sa_task_id)[0]

    probs_sa = torch.nn.functional.softmax(outputs_sa, dim=-1)
    # pred_label_sa = torch.argmax(probs_sa, dim=-1).item()

    # if pred_label_sa == 0:
    #     label_sa = "Negative"
    # else:
    #     label_sa = "Positive"

    # return label_sa, f"{float(probs_sa[0][pred_label_sa]):.3%}"

    preds = probs_sa[0].tolist()
    print(preds)

    return {"negative": preds[0], "positive": preds[1]}


def zsl_classifier(
    model: nn.Module,
    tokenizer,
    premise: str,
    candidate_labels: List,
    zsl_task_id,
    hypothesis_template: str = "Đây là một câu nói có nội dung liên quan tới chủ đề {}",
    multi_label: bool = False,
):
    sequence_pairs = []
    sequence_pairs.extend(
        [[premise, hypothesis_template.format(label)] for label in candidate_labels],
    )

    model_outputs = []

    for sequence_pair, candidate_label in zip(sequence_pairs, candidate_labels):
        inputs_nli = tokenizer(
            sequence_pair[0],
            sequence_pair[1],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**inputs_nli, task_ids=zsl_task_id)[0]
            model_outputs.append(
                {
                    "candidate_label": candidate_label,
                    "sequence": premise,
                    "logits": outputs,
                },
            )

    result = postprocess_nli(model_outputs, multi_label=multi_label)
    print(result)

    result_dict = {}

    for label, score in zip(result["labels"], result["scores"]):
        result_dict[label] = score

    # return result['labels'][0], f'{float(result["scores"][0]):.3%}'
    return result_dict


"""-----------------Gradio setup-----------------"""


task_sa = Task(
    task_id=0,
    name="uit-nlp/vietnamese_students_feedback",
    task_type="seq_classification",
    num_labels=2,
)

task_nli = Task(
    task_id=1,
    name="vinli",
    task_type="seq_classification",
    num_labels=3,
)

tasks = [task_sa, task_nli]
sa_task_id = torch.tensor([0], dtype=torch.int32)
zsl_task_id = torch.tensor([1], dtype=torch.int32)

mtl_model = load_mtl_model(
    encoder="vinai/phobert-base-v2",
    state_dict_path="./model/mtl_models/checkpoint-25284/pytorch_model.bin",  # noqa: E501
    tasks=tasks,
)  # noqa: E501

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")


def generate_sa_and_topic(text, candidate_labels, hypothesis_template, multi_label):
    candidate_labels = candidate_labels.split(",")
    multi_label = True if multi_label == "True" else False
    processed_text = preprocess_text(text)

    sa_prediction_dict = sa_classifier(
        model=mtl_model,
        tokenizer=tokenizer,
        text=processed_text,
        sa_task_id=sa_task_id,
    )

    zsl_prediction_dict = zsl_classifier(
        model=mtl_model,
        tokenizer=tokenizer,
        premise=processed_text,
        candidate_labels=candidate_labels,
        zsl_task_id=zsl_task_id,
        hypothesis_template=hypothesis_template,
        multi_label=multi_label,
    )

    return sa_prediction_dict, zsl_prediction_dict  # noqa: E501


with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="pink", secondary_hue="yellow")
) as demo:  # noqa: E501
    with gr.Row():
        with gr.Column():
            seed = gr.Text(label="Input Phrase", autofocus=True)
            candidate_labels = gr.Text(
                label="Candidate Labels",
                value="tin tức, thể thao, giải trí, game, khoa học, công nghệ, tài chính, y tế, cuộc sống, giáo dục",  # noqa: E501
            )
            hypothesis_template = gr.Text(
                label="Hypothesis Template",
                value="Đây là một câu nói có nội dung liên quan tới chủ đề {}",
            )
            multi_label = gr.Dropdown(
                label="Multi-label",
                choices=["True", "False"],
                value="False",
            )
        with gr.Column():
            sentiment_analysis = gr.Label(
                label="Sentiment Analysis Prediction",
            )
            topic_zsl = gr.Label(label="Topic Prediction")
    btn = gr.Button("Generate")
    btn.click(
        generate_sa_and_topic,
        inputs=[seed, candidate_labels, hypothesis_template, multi_label],
        outputs=[sentiment_analysis, topic_zsl],
    )
    gr.Examples(
        [
            "thỏa thuận thực tế của nhà băng với khách gửi nhiều tiền về mức lãi suất là một thỏa thuận riêng tư chỉ người trong cuộc mới có thể biết rõ tường tận",
            "do trình độ tiếng anh của lớp không cao chỉ một số ít có khả năng nghe đọc và hiểu được bài giảng của thấy nên hiệu quả của việc giảng dạy bằng tiếng anh là chưa cao",
            "Tôi thật sự không thích lớp học của ông ấy nhưng vì có crush nên tôi thích đi học mỗi ngày",
        ],
        inputs=[seed],
    )


if __name__ == "__main__":
    demo.launch(share=True)
