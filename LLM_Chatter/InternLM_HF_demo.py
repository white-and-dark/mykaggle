import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 指定本地模型路径
local_model_path = "/workspace/models/internlm2-1_8b"

tokenizer = AutoTokenizer.from_pretrained(
    "internlm/internlm2-1_8b",      # 线上模型名
    local_files_only=True,         # 允许联网下载
    trust_remote_code=True
)
tokenizer.save_pretrained(local_model_path)

model = AutoModelForCausalLM.from_pretrained(
    "internlm/internlm2-1_8b",      # 线上模型名
    torch_dtype=torch.float16,
    trust_remote_code=True
).cuda()
model.save_pretrained(local_model_path)

# 自动逻辑：有就加载本地，没有就从hf下载 → 保存到 model_path
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
# `torch_dtype=torch.float16` 可以令模型以 float16 精度加载，否则 transformers 会将模型加载为 float32，有可能导致显存不足
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,      # 线上模型名
    torch_dtype=torch.float16,
    trust_remote_code=True
).cuda()

model = model.eval()

inputs = tokenizer(["来到美丽的大自然"], return_tensors="pt")
for k,v in inputs.items():
    inputs[k] = v.cuda()
gen_kwargs = {"max_length": 128, "top_p": 0.8, "temperature": 0.8, "do_sample": True, "repetition_penalty": 1.0}
output = model.generate(**inputs, **gen_kwargs)
output = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
print(output)
# # 来到美丽的大自然，我们不仅能够观赏到美丽的风景，还能够品尝到许多美味的食物。在大自然中，有许多美味的食物，比如山野菜、野果、野菜等。这些食物不仅美味可口，而且营养非常丰富。
# # 山野菜是一种非常美味的食物，它富含多种维生素、矿物质和蛋白质等营养成分。山野菜的口感也非常独特，有些山野菜口感脆嫩，有些则比较柔软。这些山野菜通常生长在山间、林下、田野等地方，因此得名山野菜。
