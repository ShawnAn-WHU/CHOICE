# 🌱 Environment Installation

All open-source general-domain VLMs evaluated in CHOICE are reproduced within the [`ms-swift`](https://github.com/modelscope/ms-swift) framework (release-2.6), employing their default generation configurations.

---

Clone the CHOICE repository and create a conda environment:
```bash
git clone https://github.com/ShawnAn-WHU/CHOICE.git
conda create -n choice python=3.10 -y
conda activate choice
```

To ensure compatibility, it is recommended to manually install the required versions of PyTorch and Transformers as follows:
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.45.1
pip install qwen_vl_utils
``` 

Install ms-swift dependencies from source:
```bash
cd ms-swfit
git checkout release/2.6
pip install -e '.[all]'
```

# 🚀 Run Inference
Navigate to the `infer_vlm` folder:
```bash
cd infer_vlm
```

Take `qwen2_vl_7b_instruct` as an example and infer on the `CHOICE_subset`. Change your `root_path` in the main function, and run:
```bash
python infer_choice.py
```

The model will be downloaded automatically and the inference responses will be save in `<out_root_path>`.

# 📊 Run Evaluation
We provide two approaches for computing the accuracy: (1) choice extraction by template matching and (2) LLM as a judger.

## 📝 Choice extraction by template matching

Modify the following code:
```python
accevaluator = AccEvaluator(root_path=root_path, out_path=out_path, use_llm=False)

df.to_excel(os.path.join(out_path, "results.xlsx"))
```

run:
```bash
python evaluator.py
```

Each item in the `.josn` files will be added a `"result"` key to indicate if the response of the VLM matches the GT label. The metrics will be save in `results.xlsx`.

## 🤖 LLM as a judger

**Deploy a LLM using [`LMDeploy`](https://github.com/InternLM/lmdeploy):**

To prevent environment conflicts, create another environment:
```bash
conda create -n lmdeploy python=3.10 -y
conda activate lmdeploy
```

Install dependencies:
```bash
pip install lmdeploy openai
```

Deploy a LLM locally (take `Qwen2.5-7B-Instruct` as an example):
```bash
export LMDEPLOY_USE_MODELSCOPE=True

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='0,1' lmdeploy serve api_server Qwen/Qwen2.5-7B-Instruct --server-port 23333 --tp 2
```

> Run `lmdeploy serve api_server --help` for more information.

Modify the following code:
```python
accevaluator = AccEvaluator(root_path=root_path, out_path=out_path, use_llm=True)

df.to_excel(os.path.join(out_path, "results_llm.xlsx"))
```

run:
```bash
python evaluator.py
```

Each item in the `.josn` files will be added a `"result"` key to indicate if the response of the VLM matches the GT label. The metrics will be save in `results_llm.xlsx`.
