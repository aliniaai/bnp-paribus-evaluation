import os
import requests
from datetime import datetime


class APIException(Exception):
    def __init__(self, input_json: dict, status_code: int, response_text: str):
        self.input_json = input_json
        self.status_code = status_code
        self.response_text = response_text

    def __str__(self) -> str:
        return (
            f"API Error:\nstatus code: {self.status_code}\n"
            f"response text: {self.response_text}\ninput json:{self.input_json}"
        )


class Results:
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0

    def calculate_stats(self) -> None:
        if self.tp > 0 and self.fp > 0:
            self.accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        else:
            self.accuracy = None

        if self.tp > 0:
            self.precision = self.tp / (self.tp + self.fp)
            self.recall = self.tp / (self.tp + self.fn)
        else:
            self.precision = None
            self.recall = None

        if self.precision is not None and self.recall is not None:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1_score = None

    def __str__(self) -> str:
        return (
            f"tp: {self.tp}\ntn: {self.tn}\nfp: {self.fp}\nfn: {self.fn}\n"
            f"accuracy: {self.accuracy}\nprecision: {self.precision}\n"
            f"recall: {self.recall}\nf1_score: {self.f1_score}"
        )


def get_detection_config_json(
    violence: bool = False,
    hate: bool = False,
    wrongdoing: bool = False,
    sexual: bool = False,
    adversarial: bool = False,
) -> dict:
    detection_config_json = {}

    if violence or hate or wrongdoing or sexual:
        detection_config_json["safety"] = {}
        if violence:
            detection_config_json["safety"]["violence"] = True
        if hate:
            detection_config_json["safety"]["hate"] = True
        if wrongdoing:
            detection_config_json["safety"]["wrongdoing"] = True
        if sexual:
            detection_config_json["safety"]["sexual"] = True

    if adversarial:
        detection_config_json["security"] = {"adversarial": True}

    return detection_config_json


def get_alinia_input_json(input_str: str, detection_config: dict) -> dict:
    return {"input": input_str, "detection_config": detection_config}


def evaluate(input_json: dict, api_key: str) -> dict:
    response = requests.post(
        "https://staging.api.alinia.ai/moderations/",
        headers={"Authorization": f"Bearer {api_key}"},
        json=input_json,
    )
    if not response.ok:
        print(f">>>> HTTP error {response.status_code}: {response.text}")
        raise APIException(input_json, response.status_code, response.text)
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        print(f">>>> Failed to decode JSON response (status {response.status_code}): {response.text!r}")
        raise APIException(input_json, response.status_code, response.text)


def evaluate_example(
    example, text_parameter, class_parameter, positive_class, negative_class,
    detection_config, results_obj, error_path, api_key
):
    input_str = example[text_parameter]
    label = example[class_parameter]
    response_json = None

    print(f"Evaluating: {input_str}")
    input_json = get_alinia_input_json(input_str, detection_config)
    try:
        response_json = evaluate(input_json, api_key)
    except APIException as e:
        print(e)
        with open(error_path, "a") as errors_fp:
            errors_fp.write(str(datetime.now()) + "\n" + str(e) + "\n")

    is_flagged = False
    if response_json is not None and "result" in response_json:
        is_flagged = response_json["result"].get("flagged", False)

    if response_json is not None:
        if is_flagged and label == positive_class:
            results_obj.tp += 1
        elif is_flagged and label == negative_class:
            results_obj.fp += 1
        elif not is_flagged and label == negative_class:
            results_obj.tn += 1
        elif not is_flagged and label == positive_class:
            results_obj.fn += 1


def evaluate_example_multiclass(
    example, text_parameter, class_parameter, positive_class_arr, negative_class_arr,
    detection_config, results_obj, error_path, api_key
):
    input_str = example[text_parameter]
    label = example[class_parameter]
    response_json = None

    print(f"Evaluating: {input_str}")
    input_json = get_alinia_input_json(input_str, detection_config)
    try:
        response_json = evaluate(input_json, api_key)
    except APIException as e:
        print(e)
        with open(error_path, "a") as errors_fp:
            errors_fp.write(str(datetime.now()) + "\n" + str(e) + "\n")

    is_flagged = False
    if response_json is not None and "result" in response_json:
        is_flagged = response_json["result"].get("flagged", False)

    if response_json is not None:
        if is_flagged and label in positive_class_arr:
            results_obj.tp += 1
        elif is_flagged and label in negative_class_arr:
            results_obj.fp += 1
        elif not is_flagged and label in negative_class_arr:
            results_obj.tn += 1
        elif not is_flagged and label in positive_class_arr:
            results_obj.fn += 1


def calculate_stats_and_save(results_obj, title, filepath):
    results_obj.calculate_stats()
    print(f"{title}:")
    print(results_obj)
    with open(filepath, "w") as results_fp:
        results_fp.write(f"{title}:\n")
        results_fp.write(str(results_obj))