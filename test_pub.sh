export HF_HOME=""
export HF_TOKEN=""
export OPENAI_API_KEY=""
export CUDA_VISIBLE_DEVICES=""

# LLaVA-NeXT-Video
accelerate launch --num_processes 1 --main_process_port 12345 -m lmms_eval \
    --model llava_vid \
    --model_args pretrained=lmms-lab/LLaVA-NeXT-Video-32B-Qwen,conv_template=qwen_1_5,video_decode_backend=decord,max_frames_num=32,mm_spatial_pool_mode=average,mm_newline_position=grid,mm_resampler_location=after \
    --tasks videoads \
    --batch_size 1 \
    --output_path ./logs_final \
    --log_samples

accelerate launch --num_processes 1 --main_process_port 12345 -m lmms_eval \
    --model llava_vid \
    --model_args pretrained=lmms-lab/LLaVA-NeXT-Video-32B-Qwen,conv_template=qwen_1_5,video_decode_backend=decord,max_frames_num=32,mm_spatial_pool_mode=average,mm_newline_position=grid,mm_resampler_location=after \
    --tasks videoads_w_subtitle \
    --batch_size 1 \
    --output_path ./logs_final \
    --log_samples

accelerate launch --num_processes 1 --main_process_port 12345 -m lmms_eval \
    --model llava_vid \
    --model_args pretrained=lmms-lab/LLaVA-NeXT-Video-32B-Qwen,conv_template=qwen_1_5,video_decode_backend=decord,max_frames_num=32,mm_spatial_pool_mode=average,mm_newline_position=grid,mm_resampler_location=after \
    --tasks videoads_w_cot \
    --batch_size 1 \
    --output_path ./logs_final \
    --log_samples

