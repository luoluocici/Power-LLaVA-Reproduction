# Power-LLaVA 复现作业代码库

## 1. 仓库说明
本仓库为《基于单卡 2080Ti 的 Power-LLaVA 复现与定性评估》期末大作业的代码仓库。

## 2. 权重说明
*   **Baseline 权重**：本实验使用官方开源的 LLaVA-1.5-7B 权重。
    *   下载方式：`huggingface-cli download liuhaotian/llava-v1.5-7b`
*   **提出方法的权重 (Proposed Method)**：
    *   由于原论文作者尚未开源 Power-LLaVA 的代码与 10 万条 PowerQA 数据集，且本实验室单卡 RTX 2080 Ti (11GB) 在尝试进行 LoRA 微调时发生显存溢出（OOM），导致服务器宕机。
    *   因此，本实验未生成最终的微调权重，重点聚焦于 Baseline 的 8-bit 量化推理评估（见 `eval_power.py`）。

## 3. 核心代码文件
*   `eval_power.py`：用于在 8-bit 量化下，对真实的电力巡检图片进行定性测试。
