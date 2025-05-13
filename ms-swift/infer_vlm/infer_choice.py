import os
import json
import time
import torch
import os.path as osp
from tqdm import tqdm
from swift.utils import seed_everything
from swift.llm import (
    ModelType,
    inference,
    get_template,
    get_model_tokenizer,
    get_default_template_type,
)


QWEN_PIC_PROMPT = """<img>{}</img>"""
SYSTEM_PROMPT_3 = """Please select the most appropriate answer for the following single-choice question from the given options. Only respond with the corresponding letter (A or B or C). Do not include any additional text.
"""
SYSTEM_PROMPT_4 = """Please select the most appropriate answer for the following single-choice question from the given options. Only respond with the corresponding letter (A, B, C or D). Do not include any additional text.
"""
CHOICE_HIERARCHY = {
    "perception": {
        "image_level_comprehension": [
            "image_modality",
            "image_quality",
            "map_recognition",
            "scene_classification",
            "image_caption",
        ],
        "single_instance_identification": [
            "landmark_recognition",
            "object_counting",
            "object_localization",
            "object_presence",
            "attribute_recognition",
            "visual_grounding",
            "hallucination_detection",
        ],
        "cross_instance_discerment": [
            "attribute_comparison",
            "spatial_relationship",
            "change_detection",
        ],
    },
    "reasoning": {
        "assessment_reasoning": ["environmental_assessment", "resource_assessment"],
        "attribute_reasoning": ["physical_property", "time_property"],
        "common_sense_reasoning": [
            "disaster_discrimination",
            "geospatial_determination",
            "situation_inference",
        ],
    },
}


class RSBenchEvaluator:

    def __init__(
        self,
        root_path,
        out_path,
        sys_prompt,
        model_name,
        model_type,
        model,
        tokenizer,
        top1_level_name,
        ability_name,
        task_name,
    ):
        self.root_path = root_path
        self.out_path = out_path
        self.sys_prompt = sys_prompt
        self.model_name = model_name
        self.model_type = model_type
        self.model = model
        self.tokenizer = tokenizer
        self.top1_level_name = top1_level_name
        self.ability_name = ability_name
        self.task_name = task_name

    def _prepare_query(self, data):
        if self.model_name in [
            "qwen2_vl_7b_instruct",
            "qwen2_vl_72b_instruct",
            "deepseek_vl_7b_chat",
            "phi3_vision_128k_instruct",
        ]:
            query = (
                QWEN_PIC_PROMPT.format(osp.join(self.root_path, data["image_path"]))
                + self.sys_prompt
                + data["question"]
            )
            return query, None
        elif self.model_name in [
            "llava1_6_vicuna_7b_instruct",
            "llava1_6_vicuna_13b_instruct",
            "minicpm_v_v2_5_chat",
        ]:
            query = "<image>" + self.sys_prompt + data["question"]
            images = [osp.join(self.root_path, data["image_path"])]
            return query, images
        elif self.model_name in [
            "internvl2_8b",
            "internvl2_26b",
            "internvl2_40b",
            "llama3_2_11b_vision_instruct",
            "glm4v_9b_chat",
            "ovis1_6_gemma2_9b",
            "molmo_7b_d",
            "mplug_owl3_7b_chat",
        ]:
            query = self.sys_prompt + data["question"]
            images = [osp.join(self.root_path, data["image_path"])]
            return query, images
        else:
            raise NotImplementedError(f"{self.model_name} not implemented")

    def vlm_inference(self):
        self.model.config.seq_length = 4096
        self.model.generation_config.max_new_tokens = 128
        template_type = get_default_template_type(self.model_type)
        template = get_template(template_type, self.tokenizer)
        seed_everything(42)

        task_path = osp.join(
            self.root_path,
            self.top1_level_name,
            self.ability_name,
            self.task_name,
            f"{self.task_name}.json",
        )
        out_path = osp.join(self.out_path, f"{self.task_name}.json")
        if os.path.exists(out_path):
            print(f"Output already exists for {self.task_name}, skipping...")
            return

        with open(task_path, "r") as f:
            datas = json.load(f)

        res_list = []
        for data in tqdm(datas):
            query, images = self._prepare_query(data)
            response, _ = inference(self.model, template, query, images=images)

            res_list.append(
                {
                    "id": data["id"],
                    "image_path": data["image_path"],
                    "query": query,
                    "response": response,
                    "label": data["answer"],
                }
            )

        with open(out_path, "w") as f:
            json.dump(res_list, f, indent=4)


def main():
    root_path = "/home/anxiao/CHOICE/CHOICE_subset"
    out_root_path = "./CHOICE_output"
    os.makedirs(out_root_path, exist_ok=True)

    models = {
        "qwen2_vl_7b_instruct": ModelType.qwen2_vl_7b_instruct,
        "qwen2_vl_72b_instruct": ModelType.qwen2_vl_72b_instruct,
        "internvl2_8b": ModelType.internvl2_8b,
        "internvl2_26b": ModelType.internvl2_26b,
        "internvl2_40b": ModelType.internvl2_40b,
        "llava1_6_vicuna_7b_instruct": ModelType.llava1_6_vicuna_7b_instruct,
        "llava1_6_vicuna_13b_instruct": ModelType.llava1_6_vicuna_13b_instruct,
        "llama3_2_11b_vision_instruct": ModelType.llama3_2_11b_vision_instruct,
        "glm4v_9b_chat": ModelType.glm4v_9b_chat,
        "deepseek_vl_7b_chat": ModelType.deepseek_vl_7b_chat,
        "minicpm_v_v2_5_chat": ModelType.minicpm_v_v2_5_chat,
        "phi3_vision_128k_instruct": ModelType.phi3_vision_128k_instruct,
        "mplug_owl3_7b_chat": ModelType.mplug_owl3_7b_chat,
        "ovis1_6_gemma2_9b": ModelType.ovis1_6_gemma2_9b,
        "molmo_7b_d": ModelType.molmo_7b_d,
    }

    for model_name, model_type in models.items():
        model, tokenizer = get_model_tokenizer(
            model_type, torch.float16, model_kwargs={"device_map": "auto"}
        )
        for top1_level_name, top2_level_abilities in CHOICE_HIERARCHY.items():
            for ability_name, task_list in top2_level_abilities.items():
                for task_name in task_list:
                    if task_name in ["attribute_comparison", "object_presence"]:
                        sys_prompt = SYSTEM_PROMPT_3
                    elif task_name == "visual_grounding":
                        sys_prompt = ""
                    else:
                        sys_prompt = SYSTEM_PROMPT_4
                    out_path = osp.join(
                        out_root_path, model_name, top1_level_name, ability_name
                    )
                    os.makedirs(out_path, exist_ok=True)

                    rebenchevaluator = RSBenchEvaluator(
                        root_path=root_path,
                        out_path=out_path,
                        sys_prompt=sys_prompt,
                        model_name=model_name,
                        model_type=model_type,
                        model=model,
                        tokenizer=tokenizer,
                        top1_level_name=top1_level_name,
                        ability_name=ability_name,
                        task_name=task_name,
                    )
                    rebenchevaluator.vlm_inference()

        del model
        del tokenizer
        torch.cuda.empty_cache()
        time.sleep(10)


if __name__ == "__main__":
    main()
