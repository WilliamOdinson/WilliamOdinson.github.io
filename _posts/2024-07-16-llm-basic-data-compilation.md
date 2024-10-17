---
title: 'LLM Basic Data Compilation: VRAM and Speed Requirements for Inference'
date: 2024-07-16
permalink: /posts/2024/07/llm-basic-data-compilation
tags: 
 - LLM
excerpt: "This article explores essential factors for running Large Language Models, including VRAM requirements based on model size, inference speeds, and the impact of quantization on performance and memory. It evaluates acceleration tools like vLLM and DeepSeed, and examines how context length and batch size influence VRAM usage and efficiency, using data from leading sources."
---

Through this article, you can understand the following aspects:

- How much VRAM is needed to run an LLM? (What size LLM can my GPU run?)
- How is the inference speed of different LLMs?
- How does quantization affect VRAM, inference speed, and performance?
- How effective are tools like vLLM, DeepSeed, etc., in acceleration?
- What is the impact of context length and batch size on VRAM and inference?

The content and test data of this article mainly come from official materials from Qwen, Zero One Everything, Nvidia, etc. (Please refer to the reference section for relevant documents).

## 1 Quick Method for Calculating VRAM Usage (Overview)

- **8-bit quantized models:** 1B parameters occupy over 1G of VRAM.
  
  For example:
  
  - **8-bit quantization:** 7B model requires over 7G of VRAM
  - **4-bit quantization:** 7B model requires over 3.5G of VRAM
  - **Float16:** 7B model requires over 14G of VRAM

## 2 How Much VRAM Do Different Parameter-Sized LLMs Require for Inference?

**Experimental Setup:** batch-size = 1  
*Note: Some models are only recommended for GPU usage and do not have VRAM data.*

### 2.1 Low-End Usage (Limited Computational Resources)

**Int4 Quantization, approximately 2K context**

| Model (int4)         | Required VRAM (GB) | Recommended GPU                | Reference Model                        |
| :------------------ | :----------------- | :------------------------------ | :------------------------------------- |
| 0.5B                | <5G                 |                                 | Qwen2-0.5B-Instruct                    |
| 1.5B                | <3G                 |                                 | Qwen-1_8B-Chat, Qwen2-1.5B-Instruct    |
| 6B                  | 4G                  |                                 | Yi-6B-Chat-4bits                       |
| 7B                  | <11G                |                                 | Qwen2-7B-Instruct, Qwen-7B-Chat-Int4    |
| 14B                 | 13G                 |                                 | Qwen-14B-Chat-Int4                     |
| 34B                 | 20G                 |                                 | Yi-34B-Chat-4bits                      |
| 57B                 | <35G                |                                 | Qwen2-57B-A14B-Instruct                |
| 72B                 | <47G                |                                 | Qwen2-72B-Instruct                     |
| 130B                | -                   | 8 × RTX 2080 Ti (11G), 4 × RTX 3090 (24G) | GLM-130B                           |
| 236B                | 130G                | 8 × A100 (80G)                   | DeepSeek-V2-Chat                        |

### 2.2 Standard Usage (Balanced Performance and Resources)

**Int8 Quantization, 4k to 6k context**

| Model (int8)          | Required VRAM (GB) | Recommended GPU | Reference Model               |
| :------------------- | :----------------- | :--------------- | :---------------------------- |
| 0.5B                 | 6G                 |                  | Qwen2-0.5B-Instruct           |
| 1.5B                 | 8G                 |                  | Qwen2-1.5B-Instruct           |
| 6B                   | 8G                 |                  | Yi-6B-Chat-8bits              |
| 7B                   | 14G                |                  | Qwen2-7B-Instruct             |
| 14B                  | 27G                |                  | Qwen-14B-Chat-Int8            |
| 34B                  | 38G                |                  | Yi-34B-Chat-8bits             |
| 57B                  | 117G (bf16)        |                  | Qwen2-57B-A14B-Instruct       |
| 72B                  | 80G                |                  | Qwen2-72B-Instruct            |
| 130B                 | -                  | 8 × RTX3090 (24G) | GLM-130B                      |
| 236B                 | 490G (fb16)        | 8 × A100 (80G)    | DeepSeek-V2-Chat               |
| 340B                 | -                  | 16 × A100 (80G), 16 × H100 (80G), 8 × H200 | Nemotron-4-340B-Instruct |

### 2.3 High-End Usage (Advanced Usage, Performance Priority)

**Performance priority, no quantization, data format FB16, 32K context**

| Model (fb16)         | Required VRAM (GB) | Recommended GPU | Reference Model                |
| :------------------- | :----------------- | :--------------- | :----------------------------- |
| 0.5B                 | 27G                |                  | Qwen2-0.5B-Instruct            |
| 1.5B                 | 30G                |                  | Qwen2-1.5B-Instruct            |
| 6B                   | 20G                |                  | Yi-6B-200K                     |
| 7B                   | 43G                |                  | Qwen2-7B-Instruct              |
| 14B                  | 39G (8k)           |                  | Qwen-14B-Chat                  |
| 34B                  | 200G (200k)        | 4 × A800 (80 GB) | Yi-34B-200K                    |
| 57B                  | 117G               |                  | Qwen2-57B-A14B-Instruct        |
| 72B                  | 209G               |                  | Qwen2-72B-Instruct             |

If the above content does not help you make a decision, you can refer to more detailed data on the [Qwen official website](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html).

## 3 How Do Batch Size and Quantization Affect Required VRAM?

**Key Points:**

- Increasing the batch size also increases VRAM usage.
- Quantization can save VRAM: The table below shows that the 6B model occupies 12G VRAM in float16, 7G in 8-bit quantization, and only 4G in 4-bit quantization.

| Model                 | batch=1 | batch=4 | batch=16 | batch=32 |
| --------------------- | ------- | ------- | -------- | -------- |
| Yi-6B-Chat            | 12 GB   | 13 GB   | 15 GB    | 18 GB    |
| Yi-6B-Chat-8bits      | 7 GB    | 8 GB    | 10 GB    | 14 GB    |
| Yi-6B-Chat-4bits      | 4 GB    | 5 GB    | 7 GB     | 10 GB    |
| Yi-34B-Chat           | 65 GB   | 68 GB   | 76 GB    | > 80 GB  |
| Yi-34B-Chat-8bits     | 35 GB   | 37 GB   | 46 GB    | 58 GB    |
| Yi-34B-Chat-4bits     | 19 GB   | 20 GB   | 30 GB    | 40 GB    |

**Data Source:** [Hugging Face - 01-ai/Yi-6B-Chat](https://huggingface.co/01-ai/Yi-6B-Chat)

## 4 How Much Do Context Length and Inference Speed Affect VRAM and Inference Speed?

- The longer the context, the slower the inference speed.
- VRAM usage also increases accordingly.

| Context Length | Inference Speed (Tokens/s) | GPU Usage (GB) |
| -------------- | -------------------------- | -------------- |
| 1              | 37.97                      | 14.92          |
| 6144           | 34.74                      | 20.26          |
| 14336          | 26.63                      | 27.71          |
| 30720          | 17.49                      | 42.62          |

**Data compiled from Qwen2 official test report.**

## 5 How Does Quantization Affect Inference Speed?

**Key Points:**

- After quantization, inference speed may slow down or remain the same.
- When quantization affects GPU usage, such as reducing from multiple GPUs to a single GPU, inference speed can significantly increase.
- Test results for Qwen2 models are as follows:
  
  - **Qwen2-0.5B model:** Quantized model speed slows down
  - **Qwen2-1.5B model:** Quantized and fb16 models have similar speeds
  - **Qwen2-7B model:** Slightly slower, but when using vLLM, the quantized version is faster
  - **Qwen2-72B model:** Speed increases (especially after Int4 quantization, where inference speed noticeably increases when reducing from 2 GPUs to 1 GPU). However, when using long contexts (120k), the quantized version slows down.

For detailed results, please visit the [Qwen benchmark page](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html).

## 6 How Does Parameter Size Affect Inference Speed?

**Unit:** tokens/s

| Inference Tool | 0.5B  | 1.5B  | 7B    | 72B   |
| -------------- | ----- | ----- | ----- | ----- |
| Transformers    | 50.83 | 40.86 | 34.74 | 5.99  |
| vLLM            | 256.16| 166.23| 76.41 | 27.98 |
| vLLM Speed Increase | 5.04× | 4.07× | 2.20× | 4.67× |

**Model:** Qwen2 series, context 6K, FB16 model

## 7 How Do Tools Like vLLM, DeepSeed, CTranslate2 Affect Inference Speed?

- Compared to Transformers, using acceleration tools like vLLM, DeepSeed, etc., can increase inference speed by 2 to 5 times.
- Among the three acceleration tools—DeepSeed, vLLM, CTranslate2—CTranslate2 performs better, especially when the batch size is 1.

**Image Source:** Reference [9]: The generation speed was measured using the time taken to generate the same question (the method feels somewhat unreasonable compared to the commonly used tokens/s, but the results are consistent with those provided by Qwen2).

## 8 How Does Quantization Affect Model Performance?

- **Int8 Quantized Models:** Performance is not significantly different from float16 format.

| Model                         | MMLU 0-shot | MMLU 5-shot | CMMLU 0-shot | CMMLU 5-shot | C-Eval(val) 0-shot | C-Eval(val) 5-shot | Truthful QA 0-shot | BBH 0-shot | BBH 3-shot | GSM8k 0-shot | GSM8k 4-shot |
| ----------------------------- | ----------- | ----------- | ------------ | ------------ | ------------------- | ------------------- | ------------------- | ---------- | ---------- | ------------ | ------------ |
| Yi-34B-Chat                   | 67.62       | 73.46       | 79.11        | 81.34        | 77.04               | 78.53               | 62.43               | 51.41      | 71.74      | 71.65        | 75.97        |
| Yi-34B-Chat-8bits (GPTQ)      | 66.24       | 73.69       | 79.05        | 81.23        | 76.82               | 78.97               | 61.84               | 52.08      | 70.97      | 70.74        | 75.74        |
| Yi-34B-Chat-4bits (AWQ)       | 65.77       | 72.42       | 78.21        | 80.50        | 75.71               | 77.27               | 61.84               | 48.30      | 69.39      | 70.51        | 74.00        |
| Yi-6B-Chat                    | 58.24       | 60.99       | 69.44        | 74.71        | 68.80               | 74.22               | 50.58               | 39.70      | 47.15      | 38.44        | 44.88        |
| Yi-6B-Chat-8bits (GPTQ)        | 58.29       | 60.96       | 69.21        | 74.69        | 69.17               | 73.85               | 49.85               | 40.35      | 47.26      | 39.42        | 44.88        |
| Yi-6B-Chat-4bits (AWQ)         | 56.78       | 59.89       | 67.70        | 73.29        | 67.53               | 72.29               | 50.29               | 37.74      | 43.62      | 35.71        | 38.36        |

- **Int4 Quantized Models:** Compared to float16 models, precision loss is around 1-2 percentage points. (Similar conclusions are drawn for Yi models and Baichuan2 models.)
  
  [Baichuan2 GitHub](https://github.com/baichuan-inc/Baichuan2)

## 9 Common GPU References for LLMs

| GPU              | VRAM |
| ---------------- | :--: |
| H200             | 141GB|
| H100, H800       | 80GB |
| A100, A800       | 80GB |
| A100             | 40GB |
| V100             | 32GB |
| RTXA6000         | 48G  |
| RTX4090, RTX3090, A10, A30 | 24GB |
| RTX4070          | 12GB |
| RTX3070          | 8GB  |

## References

1. [Qwen Benchmark](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html)
2. [Hugging Face - Qwen-1_8B-Chat](https://huggingface.co/Qwen/Qwen-1_8B-Chat)
3. [Hugging Face - Qwen-7B-Chat-Int8](https://huggingface.co/Qwen/Qwen-7B-Chat-Int8)
4. [Hugging Face - Qwen-14B-Chat-Int8](https://huggingface.co/Qwen/Qwen-14B-Chat-Int8)
5. [Hugging Face - 01-ai/Yi-6B-Chat](https://huggingface.co/01-ai/Yi-6B-Chat)
6. [GLM-130B Quantization](https://github.com/THUDM/GLM-130B/blob/main/docs/quantization.md)
7. [Hugging Face - Nemotron-4-340B-Instruct](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct)
8. [Hugging Face - DeepSeek-V2-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat)
9. [Rinna on Zenn](https://zenn.dev/rinna/articles/5fd4f3cc12f7c5)
10. [Baichuan2 GitHub](https://github.com/baichuan-inc/Baichuan2)