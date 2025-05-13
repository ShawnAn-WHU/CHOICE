import os
import re
import math
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from shapely.geometry import Polygon


class AccEvaluator:

    def __init__(
        self,
        root_path,
        out_path,
        use_llm=False,
    ):
        self.root_path = root_path
        self.out_path = out_path
        self.use_llm = use_llm

    def extract_full_option(self, query, correct_option_letter):
        option_marker = f"\n{correct_option_letter}."
        start_index = query.find(option_marker)
        if start_index == -1:
            return ""
        start_index += len(option_marker)
        remaining_query = query[start_index:]
        next_options = [
            f"\n{letter}." for letter in "ABCD" if letter != correct_option_letter
        ]
        end_indices = [remaining_query.find(opt) for opt in next_options]
        end_indices = [idx for idx in end_indices if idx != -1]
        if end_indices:
            end_index = min(end_indices)
        else:
            end_index = len(remaining_query)
        correct_option_content = remaining_query[:end_index].strip()

        return f"{correct_option_letter}.{correct_option_content}"

    def general_evaluater(self, response, label):
        # Require: option. content
        if response.strip() in ["A", "B", "C", "D"]:
            output = response.strip()
        elif response.split(" ")[0] in ["A", "B", "C", "D"]:
            output = response.split(" ")[0]
        elif "." in response and response.split(".")[0] in ["A", "B", "C", "D"]:
            output = response.split(".")[0]
        elif re.search(r"(answer|Answer|could be|would be|Correct option)", response):
            pattern = r"(answer|Answer|could be|would be|Correct option).*?([ABCD])"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                output = match.group(2)
            else:
                output = "unknown"
        else:
            label_str = label.split(".")[1].replace("_", " ").lower()
            label_opt = label.split(".")[0]
            if (
                response.lower() in label_str
                or label_str.split(" ")[0] in response.lower()
            ):
                output = label_opt
            else:
                output = "unknown"

        if output not in ["A", "B", "C", "D"]:
            print(f"response: {response}\nlabel: {label}\n\n")

        if output == label.split(".")[0]:
            return True
        else:
            return False

    def llm_evaluator(self, query, response, label):
        try:
            client = OpenAI(api_key="sk-123456", base_url="http://0.0.0.0:23333/v1")
            model_name = client.models.list().data[0].id
        except:
            raise ValueError("No client found, please refer to the evaluation.md")

        LLM_SYSTEM_PROMPT = """Question: {question}\nGround Truth Answer: {ground_truth}\nPredicted Answer: {predicted}\nDoes the predicted answer match the ground truth? Answer "True" for match and "False" for not match. Use semantic meaning rather than exact match. Do not include any other text."""
        query_text = LLM_SYSTEM_PROMPT.format(
            question="\n".join(query.split("\n")[1:]),
            ground_truth=label,
            predicted=response,
        )

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": query_text,
                            },
                        ],
                    }
                ],
                timeout=10,
            )
            response = str(response.choices[0].message.content).lower().strip()
            pattern = r"(true|false)"
            match = re.search(pattern, response)
            if match:
                selected_option = match.group(1)
                if selected_option == "true":
                    output = True
                else:
                    output = False
            else:
                print(f"No match true/false for LLM response: {response}")
                output = False
        except:
            print(f"LLM failed for query: {query}")
            output = False

        return output

    def sort_coords_counterclockwise(self, coords):
        centroid_x = sum(x for x, y in coords) / len(coords)
        centroid_y = sum(y for x, y in coords) / len(coords)

        def angle_from_centroid(point):
            x, y = point
            return math.atan2(y - centroid_y, x - centroid_x)

        sorted_coords = sorted(coords, key=angle_from_centroid)

        return sorted_coords

    def compute_miou(self, coords, coords_gt):
        x1, y1, x2, y2 = map(float, coords)
        rect1 = Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
        coords_gt = self.sort_coords_counterclockwise(coords_gt)
        rect2 = Polygon(coords_gt)

        intersection = rect1.intersection(rect2).area
        union = rect1.union(rect2).area
        miou = intersection / union

        return miou

    def is_point_in_polygon(self, gt, response):
        n = len(gt)
        inside = False
        p1x, p1y = gt[0]
        for i in range(n + 1):
            p2x, p2y = gt[i % n]
            if response[1] > min(p1y, p2y):
                if response[1] <= max(p1y, p2y):
                    if response[0] <= max(p1x, p2x):
                        if p1y != p2y:
                            x_inters = (response[1] - p1y) * (p2x - p1x) / (
                                p2y - p1y
                            ) + p1x
                        if p1x == p2x or response[0] <= x_inters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def bbox_evaluater(self, response, label):
        pattern = r"(?<![a-zA-Z/])([-+]?\d*\.\d+|\d+)(?!\s?meter)(?![a-zA-Z/\d])"
        numbers = re.findall(pattern, response)
        if len(numbers) >= 4:
            x1, y1, x2, y2 = map(float, numbers[:4])
            while x1 > 1.0 or y1 > 1.0 or x2 > 1.0 or y2 > 1.0:
                x1, y1, x2, y2 = x1 / 10.0, y1 / 10.0, x2 / 10.0, y2 / 10.0
            coords = [x1, y1, x2, y2]
            miou = self.compute_miou(coords, label)
            center_point = [(x1 + x2) / 2, (y1 + y2) / 2]
            in_polygon = self.is_point_in_polygon(label, center_point)
        else:
            miou = 0
            in_polygon = False

        miou_5 = miou > 0.5
        miou_3 = miou > 0.3

        return [miou_5, miou_3, in_polygon]


def main():
    root_path = "./CHOICE_output"
    out_path = "./CHOICE_output"
    accevaluator = AccEvaluator(
        root_path=root_path,
        out_path=out_path,
        use_llm=False,
    )

    results = {}
    for model_name in [
        "qwen2_vl_7b_instruct",
        "qwen2_vl_72b_instruct",
        "internvl2_8b",
        "internvl2_26b",
        "internvl2_40b",
        "llava1_6_vicuna_7b_instruct",
        "llava1_6_vicuna_13b_instruct",
        "llama3_2_11b_vision_instruct",
        "glm4v_9b_chat",
        "deepseek_vl_7b_chat",
        "minicpm_v_v2_5_chat",
        "phi3_vision_128k_instruct",
        "mplug_owl3_7b_chat",
        "ovis1_6_gemma2_9b",
        "molmo_7b_d",
    ]:
        model_path = os.path.join(root_path, model_name)
        if not os.path.isdir(model_path) or not os.path.exists(model_path):
            continue

        task_order = [
            "image_modality",
            "image_quality",
            "map_recognition",
            "scene_classification",
            "image_caption",
            "landmark_recognition",
            "object_counting",
            "object_localization",
            "object_presence",
            "attribute_recognition",
            "visual_grounding",
            "hallucination_detection",
            "attribute_comparison",
            "spatial_relationship",
            "change_detection",
            "time_property",
            "physical_property",
            "environmental_assessment",
            "resource_assessment",
            "disaster_discrimination",
            "geospatial_determination",
            "situation_inference",
        ]
        model_results = {task: 0 for task in task_order}

        for root, _, files in os.walk(model_path):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as json_file:
                        data = json.load(json_file)
                    correct = 0
                    total = len(data)
                    for obj in tqdm(data):
                        task_name = file.split(".")[0]
                        if task_name == "visual_grounding":
                            result = accevaluator.bbox_evaluater(
                                obj["response"], obj["label"]
                            )[
                                0
                            ]  # result is a list containing three elements, we only use the first one here
                        else:
                            if accevaluator.use_llm:
                                result = accevaluator.llm_evaluator(
                                    obj["query"], obj["response"], obj["label"]
                                )
                            else:
                                label = accevaluator.extract_full_option(
                                    obj["query"], obj["label"]
                                )
                                result = accevaluator.general_evaluater(
                                    obj["response"], label
                                )
                        if result:
                            correct += 1
                        obj["result"] = result
                    model_results[task_name] = (
                        round(correct / total, 4) if total > 0 else 0
                    )

                    with open(file_path, "w") as json_file:
                        json.dump(data, json_file, indent=4)

        results[model_name] = model_results

    df = pd.DataFrame(results).T
    df.fillna(0, inplace=True)
    df.to_excel(os.path.join(out_path, "results.xlsx"))


if __name__ == "__main__":
    main()
