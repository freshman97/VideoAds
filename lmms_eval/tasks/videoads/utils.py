import datetime
import json
import os
os.environ["HF_HOME"] = "/data/zheyuan/hf"
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

# with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
#     raw_data = f.readlines()
#     safe_data = []
#     for i, line in enumerate(raw_data):
#         # remove function definition since yaml load cannot handle it
#         if "!function" not in line:
#             safe_data.append(line)

#     config = yaml.safe_load("".join(safe_data))

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
# cache_dir = os.path.join(hf_home, cache_dir)
# base_cache_dir = config["dataset_kwargs"]["cache_dir"]
base_cache_dir = os.path.expanduser(hf_home)

with open(Path(__file__).parent / "videoads.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

base_dataset_path = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
basevideo_path = os.path.join(base_dataset_path, "youtube_videos")
basesubtitle_path = os.path.join(base_dataset_path, "auto_subtitle")

def convert_time_to_frame(time_in_seconds, fps):
    return int(time_in_seconds * fps)


def extract_subtitles(video_path, subtitle_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    subtitles = load_subtitles(subtitle_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        start_frame = convert_time_to_frame(start_time, fps)
        end_frame = convert_time_to_frame(end_time, fps)
        subtitle_frames.append((start_frame, end_frame, text))

    return subtitle_frames, total_frame


def parse_subtitle_time(time_str):
    h, m, s_ms = time_str.split(":")
    

    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def load_subtitles(subtitle_path):
    subtitles = {}
    with open(subtitle_path, "r", encoding="utf-8") as file:
        content = file.read().split("\n\n")
        for section in content:
            if section.strip():
                lines = section.split("\n")
                try:
                    if len(lines) >= 3:
                        time_range = lines[1].split(" --> ")
                        start_time = parse_subtitle_time(time_range[0])
                        end_time = parse_subtitle_time(time_range[1])
                        text = " ".join(line for line in lines[2:])
                        subtitles[(start_time, end_time)] = text
                except Exception as e:
                    print(e)
                    print(subtitle_path)
                    import pdb; pdb.set_trace()
    return subtitles


def convert_time_to_frame(time_in_seconds, fps):
    return int(time_in_seconds * fps)


def extract_subtitles(video_path, subtitle_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    subtitles = load_subtitles(subtitle_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        start_frame = convert_time_to_frame(start_time, fps)
        end_frame = convert_time_to_frame(end_time, fps)
        subtitle_frames.append((start_frame, end_frame, text))

    return subtitle_frames, total_frame


def videoads_doc_to_visual(doc):
    video_path = doc["video_name"]
    video_path = os.path.join(basevideo_path, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def videoads_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_prompt = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc["question"]
    option = str(doc["options"])
    question = question + "\n" + option
    post_prompt = lmms_eval_specific_kwargs["post_prompt"] if "post_prompt" in lmms_eval_specific_kwargs else "The best answer is:"
    full_prompt = option_prompt + "\n" + question + "\n" + post_prompt
    return full_prompt


def videoads_doc_to_cot_text(doc, lmms_eval_specific_kwargs=None):
    option_prompt = ("Select the best answer to the following multiple-choice question based on the video."
                     "You are provided with several hint questions to help you answer the main question."
                     "Please first answer the hint questions shortly, then answer the main question."
                     "For the main question, the response should start with pattern Based on all these information, ###The correct answer is: the letter (A, B, C, or D) of the correct option.")
    hints_prompt = "Hint: "
    question = doc["question"]
    hints = doc["hints"]
    option = str(doc["options"])
    question = question + "\n" + option
    post_prompt = "The answer for hint questions shortly and then main question is:"
    full_prompt = option_prompt + "\n" + question + "\n" + hints_prompt + hints + "\n"  + post_prompt
    return full_prompt

# Frames + Subs
# This video's subtitles are listed below:
# 【subtitles】

# Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.
# 【question】
# The best answer is:
# Frames / Frames + Audio
# Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
# 【question】
# The best answer is:


def videoads_doc_to_text_subtitle(doc, lmms_eval_specific_kwargs=None):
    video_path = doc["video_name"]
    video_path = os.path.join(basevideo_path, video_path)
    subtitle_path = os.path.join(basesubtitle_path, doc["videoID"] + ".txt")

    if os.path.exists(subtitle_path):  # Denote have subtitle
        subtitle = open(subtitle_path).readlines()
        if len(subtitle) < 3:
            subtitle = ""
    else:
        subtitle = ""
    subtitles_prompt = "This video's subtitles are listed below: \n"
    subtitle_text = ""
    if subtitle == "" or subtitle == " ":
        subtitle = "No subtitles available"
    else:
        if "frame_num" in lmms_eval_specific_kwargs:
            frame_num = lmms_eval_specific_kwargs["frame_num"]
            subtitle_by_frame, total_frame = extract_subtitles(video_path, subtitle_path)
            uniform_sampled_frames = np.linspace(0, total_frame - 1, frame_num, dtype=int).tolist()

            subtitle_by_frame_idx = []
            for frame_idx in uniform_sampled_frames:
                for idx, title in enumerate(subtitle_by_frame):
                    if frame_idx < title[1] and frame_idx >= title[0]:
                        subtitle_by_frame_idx.append(idx)
            subtitle_by_frame_idx = list(set(subtitle_by_frame_idx))

            textlist = []
            for idx in subtitle_by_frame_idx:
                raw_text = subtitle_by_frame[idx][2].replace("\n", " ")
                try:
                    textlist.append(raw_text)
                except:
                    continue

            subtitle_text = "\n".join(textlist)
        subtitle = subtitle_text

    option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc["question"]
    option = str(doc["options"])
    question = question + "\n" + option
    post_prompt = lmms_eval_specific_kwargs["post_prompt"] if "post_prompt" in lmms_eval_specific_kwargs else "The best answer is:"
    full_prompt = subtitles_prompt + subtitle + "\n" + option_prompt + "\n" + question + "\n" + post_prompt
    return full_prompt


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
        "###The correct answer is",
    ]
    for answer_prefix in answer_prefixes:
        if  answer_prefix in s:
            s = s.split(answer_prefix)[1]

        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


matrices = []


def videoads_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videoads score), value: metric value
    """
    pred = results[0]
    pred_ans = extract_characters_regex(pred)
    # gt_ans = doc["answer"].lower().strip().replace(".", "")

    data_dict = {"question_id": doc["question_id"],
                 "pred_answer": pred_ans,
                 "answer": doc["answer"],
                 "type": doc["type"],
                 "hints": doc["hints"]}

    # return {f"videoads_percetion_score": data_dict for metric in matrices}
    return {f"videoads_percetion_score": data_dict}

TASK_CATEGORIES = [
    "Finding",
    "Summary",
    "Reasoning",
]

def videoads_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = {}
    for question_type in TASK_CATEGORIES:
        category2score[question_type] = {"correct": 0, "answered": 0}

    for result in results:
        question_type = result["type"]
        category2score[question_type]["answered"] += 1
        category2score[question_type]["correct"] += result["pred_answer"] == result["answer"]

    for question_type in TASK_CATEGORIES:
        total_correct = category2score[question_type]["correct"]
        total_answered = category2score[question_type]["answered"]
    
        eval_logger.info(f"Evaluation on video Type {question_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    total_correct = 0
    total_answered = 0
    for question_type in TASK_CATEGORIES:
        total_correct += category2score[question_type]["correct"] / category2score[question_type]["answered"]
        total_answered += category2score[question_type]["answered"]

    eval_logger.info(f"Total Evaluation: {100 * total_correct / len(TASK_CATEGORIES) if total_answered > 0 else 0 : .1f}%")

    return 100 * total_correct / len(TASK_CATEGORIES) if total_answered > 0 else 0
