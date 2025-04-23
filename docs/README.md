# VideoAds

This codebase is designed for the evaluation of VideoAds dataset based on ![lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). Please refer to the original work for the envs setup and inference. All you need is to simply change the task name!!!

For example:
```
# LLaVA-NeXT-Video
accelerate launch --num_processes 1 --main_process_port 12345 -m lmms_eval \
    --model llava_vid \
    --model_args pretrained=lmms-lab/LLaVA-NeXT-Video-32B-Qwen,conv_template=qwen_1_5,video_decode_backend=decord,max_frames_num=32,mm_spatial_pool_mode=average,mm_newline_position=grid,mm_resampler_location=after \
    --tasks videoads \
    --batch_size 1 \
    --output_path ./logs_final \
    --log_samples
```

You can also set it up for the updated version by simply merging the task files [videoads](../lmms_eval/tasks/videoads).

## Citation
```
@article{zhang2025videoads,
  title={VideoAds for Fast-Paced Video Understanding: Where Opensource Foundation Models Beat GPT-4o \& Gemini-1.5 Pro},
  author={Zhang, Zheyuan and Dou, Monica and Peng, Linkai and Pan, Hongyi and Bagci, Ulas and Gong, Boqing},
  journal={arXiv preprint arXiv:2504.09282},
  year={2025}
}
```
