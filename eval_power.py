import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from PIL import Image

model_path = "/workspace/models/llava-v1.5-7b"
image_files = ["power1.jpg", "power2.jpg", "power3.jpg"]
prompt_text = "What is dangerous about this image? Please describe any potential hazards to the power lines."

print("正在加载 8bit 模型到显存...")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, model_base=None, model_name="llava-v1.5-7b", load_8bit=True, device="cuda"
)
print("模型加载成功！开始批量测试...\n")

for img_path in image_files:
    print(f"==================================================")
    print(f"正在测试图片: {img_path}")
    try:
        image = Image.open(img_path).convert('RGB')
    except:
        print(f"找不到 {img_path}，跳过。")
        continue

    conv = conv_templates["llava_v1"].copy()
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt_text
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt_text
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_tensor = process_images([image], image_processor, model.config).to("cuda", dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to("cuda")

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids, images=image_tensor, do_sample=True, temperature=0.2, max_new_tokens=256, use_cache=True
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    answer = outputs.split("ASSISTANT:")[-1].strip()
    print(f"模型回答:\n{answer}\n")
