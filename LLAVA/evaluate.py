import av
import os
import sys
import logging
import pandas as pd
import torch
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig, AutoModel, AutoTokenizer

# Skip printing "Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation."
logging.getLogger("transformers").setLevel(logging.ERROR)

#Load SimCSE and SBERT models
simcse_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
simcse_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
sbert_model = SentenceTransformer("all-mpnet-base-v2")


def get_embeddings(text):
    """
    Encodes the given text into text features.
    Args:
        text (str): The input text string.
    Returns:
        torch.Tensor: The normalized text features for the input text.
    """
    # Tokenize the text
    text_input = simcse_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    # Get embeddings
    with torch.no_grad():
        embeddings = simcse_model(**text_input, output_hidden_states=True, return_dict=True).pooler_output
    return embeddings

def compute_simcse_similarity(embeddings, expl_embeddings):
    """
    Computes the cosine similarity between two sets of text features.
    Args:
        embeddings (torch.Tensor): The features of a video text, represented as a tensor.
        expl_embeddings (torch.Tensor): The features of an explanation video text, represented as a tensor.
    Returns:
        float: The cosine similarity between the two text feature tensors, rounded to 5 decimal places.
    """
    similarity = torch.nn.functional.cosine_similarity(embeddings, expl_embeddings)
    return round(similarity.item(),5)

def convert_ranges_to_tuples(txt_file_path):
    """
    Reads frame index ranges from a .txt file and returns a list of tuples.
    Args:
        txt_file_path (str): Path to the .txt file containing the frame index ranges.
    Returns:
        List[Tuple[int, int]]: A list of tuples where each tuple represents a frame range
    """
    fragments = []

    try:
        with open(txt_file_path, 'r') as file:
            for line in file:
                start, end = map(int, line.strip().split(','))
                fragments.append((start, end))

        return fragments

    except FileNotFoundError:
        print(f"Error: File not found at {txt_file_path}")
        return None
    except ValueError:
        print(f"Error: Invalid format in {txt_file_path}. Expected 'start,end' on each line.")
        return None
    except IndexError: # catches files with no lines or empty lines
        print(f"Error: File is empty or has invalid content.")
        return None

def get_first_and_last_from_txt(txt_file_path):
    """
    Reads the first and last numbers from a .txt file.
    Args:
        txt_file_path (str): Path to the .txt file.
    Returns:
        List[Tuple[int, int]]: A list of a tuple containing the first and last integers, or None if an error occurs.
    """
    fragments = []
    try:
        with open(txt_file_path, 'r') as file:
            lines = file.readlines()
            if not lines:
                return None  # Empty file

            first_line = lines[0].strip()
            last_line = lines[-1].strip()

            first_start, _ = map(int, first_line.split(','))
            _, last_end = map(int, last_line.split(','))

            fragments.append((first_start,last_end))
            return fragments

    except FileNotFoundError:
        print(f"Error: File not found at {txt_file_path}")
        return None
    except ValueError:
        print(f"Error: Invalid format in {txt_file_path}. Expected 'start,end' on each line.")
        return None
    except IndexError: # catches files with no lines or empty lines
        print(f"Error: File is empty or has invalid content.")
        return None


def read_original_video_pyav(container, intervals, num_frames):
    """
    Decode the video with PyAV decoder and select a fixed number of frames by adjusting the downsample rate dynamically.

    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        intervals (list of tuples): A list of tuples, where each tuple represents a start and end frame index (inclusive) to extract from the video.
        num_frames (int): The fixed number of frames to extract (default 150).

    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)

    # Get total number of frames available in the given intervals
    total_frames = sum(end_index - start_index + 1 for start_index, end_index in intervals)

    # Calculate downsample rate
    downsample_rate = max(1, total_frames // num_frames)

    # Collect frames based on the downsample rate
    for i, frame in enumerate(container.decode(video=0)):
        for start_index, end_index in intervals:
            if start_index <= i <= end_index:
                if (i - start_index) % downsample_rate == 0:
                    frames.append(frame.to_ndarray(format="rgb24"))
                break

    # Ensure we have exactly `num_frames` frames
    if len(frames) > num_frames:
        frames = frames[:num_frames]  # Truncate if we get more frames than needed
    elif len(frames) < num_frames:
        return None

    return np.stack(frames)

def read_video_pyav(container, intervals, downsample_rate, num_frames):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        intervals (list of tuples): A list of tuples, where each tuple represents a start and end frame index (inclusive) to extract from the video.
        downsample_rate (int): The rate at which frames are sampled.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)

    for i, frame in enumerate(container.decode(video=0)):
        for start_index, end_index in intervals:
            if start_index <= i <= end_index and (i - start_index) % downsample_rate == 0:
                frames.append(frame)
                break

    # Ensure we have don't exceed `num_frames` frames
    if len(frames) > num_frames:
        # Calculate new downsample rate
        new_downsample_rate = max(1, len(frames) // num_frames)
        frames = frames[::new_downsample_rate]
        if len(frames) > num_frames:
            frames = frames[:num_frames]

    if frames:
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])
    else:
        return None


def process_video_fragment(model, processor, device, video_fragment, video_prompt, video_prompt_length):
    """
    Processes a video fragment using the LlavaOnevision model.
    """
    video_tensor = torch.tensor(video_fragment).to(device)
    video_inputs = processor(text=video_prompt, videos=video_tensor, return_tensors="pt")
    video_inputs = {key: value.to(device) for key, value in video_inputs.items()}

    video_out = model.generate(**video_inputs, max_new_tokens=700)
    video_result = processor.batch_decode(video_out, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    video_result_text = video_result[0][video_prompt_length:]
    video_words = video_result_text.split()
    video_chunks = [' '.join(video_words[i:i + 20]) for i in range(0, len(video_words), 20)]

    return video_result_text, video_chunks



def run(video_folder, video_prompt, quant=True):
    """
    Processes videos from a given folder, generates text descriptions using a LlavaOnevision model,
    and calculates the similarity between descriptions of the original video and its explanation.
    Args:
        video_folder (str): Path to the folder containing video subfolders.
        video_prompt (str): Text prompt to be used for video description generation.
        quant (bool, optional): Whether to use quantization for the model. Defaults to True.
    """

    setup_logger()
    log_configurations(video_prompt)

    if quant:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf",
                                                                       quantization_config=quantization_config,
                                                                       device_map="auto")
    else:
        model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf",
                                                                       torch_dtype=torch.float16, device_map="auto")

    processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")

    video_conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": video_prompt},
                {"type": "video"},
            ],
        },
    ]

    # Iterate through each subfolder in the video_folder and choose videos to include to analysis
    video_subfolders = []

    # Uncomment this pair (and comment the other) to evaluate subset with at least 1 top scoring fragments
    summe_excluded = {1, 3, 5, 11, 13, 16, 20, 24, 25}                                        # exlude SumMe videos that do NOT have at least 1 top scoring fragment
    tvsum_excluded = {3, 15, 21}                                                              # exlude TVSum videos that do NOT have at least 1 top scoring fragment

    for dataset in os.listdir(video_folder):
        dataset_path = os.path.join(video_folder, dataset)
        if os.path.isdir(dataset_path):
            for subfolder in os.listdir(dataset_path):
                subfolder_path = os.path.join(dataset_path, subfolder)

                # Extract video number from subfolder name (assuming 'video_X' format)
                if dataset == "SumMe" and subfolder.startswith("video_"):
                    try:
                        video_number = int(subfolder.split("_")[1])
                        if video_number in summe_excluded:
                            continue  # Skip excluded videos
                    except ValueError:
                        pass

                # Extract video number from subfolder name (assuming 'video_X' format)
                if dataset == "TVSum" and subfolder.startswith("video_"):
                    try:
                        video_number = int(subfolder.split("_")[1])
                        if video_number in tvsum_excluded:
                            continue  # Skip excluded videos
                    except ValueError:
                        pass

                if os.path.isdir(subfolder_path):
                    video_subfolders.append(subfolder_path)

    #calculation of characters length to be removed for more readable results
    video_prompt_length = len('user ' + video_prompt + 'assistant ')

    count = 0
    for video_subfolder in video_subfolders:
        video_id = os.path.basename(video_subfolder)  # Extracts just the folder name
        video_file = f"{video_id}.mp4"
        dataset_name = os.path.basename(os.path.dirname(video_subfolder))
        video_path = os.path.join(video_folder, dataset_name, video_file)

        print(video_path)  # Debugging line to check output
        video_result_path = os.path.join(video_subfolder, f"{video_id}_text.txt")

        # Check if the result file already exists (in case the script stopped and restarted at some point)
        if os.path.exists(video_result_path):
            print(f"Output already exists for {video_file}, skipping...")
            continue

        print(f"--------------------------- Processing video: {video_id}, count: {count+1} ----------------------------------------")

        container = av.open(video_path)

        # Assuming the .txt files have the same name as the video. Txt files contain the fragments in temporal order
        video_txt_path = os.path.join(video_subfolder, f"{video_id}_shots.txt")
        video_sum_txt_path = os.path.join(video_subfolder, f"{video_id}_sum_shots.txt")
        attention_txt_file_path = os.path.join(video_subfolder, f"{video_id}_attention_explanations.txt")
        lime_txt_file_path = os.path.join(video_subfolder, f"{video_id}_lime_explanations.txt")

        # Attention and LIME top txt files contain the fragments in order of importance as calculated by CA-SUM-XAI
        attention_top_txt_path = os.path.join(video_subfolder, f"{video_id}_attention_importance.txt") ############
        lime_top_txt_path = os.path.join(video_subfolder, f"{video_id}_lime_importance.txt") ###########

        # Convert fragment ranges to tuples
        video_indices = get_first_and_last_from_txt(video_txt_path)
        video_sum_indices = convert_ranges_to_tuples(video_sum_txt_path)
        attention_expl_indices = convert_ranges_to_tuples(attention_txt_file_path)
        lime_expl_indices = convert_ranges_to_tuples(lime_txt_file_path)
        attention_top_fragments = convert_ranges_to_tuples(attention_top_txt_path)
        attention_top_1_fragment = [attention_top_fragments[0]]
        attention_top_2_fragment = [attention_top_fragments[1]]
        attention_top_3_fragment = [attention_top_fragments[2]]
        attention_top_1_2_fragments = sorted(attention_top_fragments[:2])
        lime_top_fragments = convert_ranges_to_tuples(lime_top_txt_path)
        lime_top_1_fragment = [lime_top_fragments[0]]
        lime_top_2_fragment = [lime_top_fragments[1]]
        lime_top_3_fragment = [lime_top_fragments[2]]
        lime_top_1_2_fragments = sorted(lime_top_fragments[:2])

        # print(f"Video: {video_indices}\nVideo summary fragments: {video_sum_indices}\n"
        #       f"Attention fragments (temporal order): {attention_expl_indices}\nLIME fragments (temporal order): {lime_expl_indices}\n"
        #       f"Attention fragments (importance order): {attention_top_fragments}\nLIME fragments (importance order): {lime_top_fragments}")

        device = next(model.parameters()).device
        video_prompt = processor.apply_chat_template(video_conversation, add_generation_prompt=True)


        video = read_original_video_pyav(container, video_indices, num_frames= 150)
        video_sum = read_video_pyav(container, video_sum_indices, downsample_rate=15, num_frames=150)
        video_expl_attention = read_video_pyav(container, attention_expl_indices, downsample_rate=15, num_frames=150)
        video_expl_lime = read_video_pyav(container,lime_expl_indices, downsample_rate=15, num_frames=150)
        attention_top_1_video = read_video_pyav(container, attention_top_1_fragment, downsample_rate=15, num_frames=150)
        attention_top_2_video = read_video_pyav(container, attention_top_2_fragment, downsample_rate=15, num_frames=150)
        attention_top_3_video = read_video_pyav(container, attention_top_3_fragment, downsample_rate=15, num_frames=150)
        attention_top_1_2_video = read_video_pyav(container, attention_top_1_2_fragments, downsample_rate=15, num_frames=150)
        lime_top_1_video = read_video_pyav(container, lime_top_1_fragment, downsample_rate=15, num_frames=150)
        lime_top_2_video = read_video_pyav(container, lime_top_2_fragment, downsample_rate=15, num_frames=150)
        lime_top_3_video = read_video_pyav(container, lime_top_3_fragment, downsample_rate=15, num_frames=150)
        lime_top_1_2_video = read_video_pyav(container, lime_top_1_2_fragments, downsample_rate=15, num_frames=150)

        # print(f"Video: {video.shape}\nVideo summary: {video_sum.shape}\nAttention explanation: {video_expl_attention.shape}\nLIME explanation : {video_expl_lime.shape}\n"
        #       f"Attention Top 1: {attention_top_1_video.shape}\nAttention Top 2: {attention_top_2_video.shape}\nAttention Top 3: {attention_top_3_video.shape}\nAttention Top 1&2: {attention_top_1_2_video.shape}\n"
        #       f"LIME Top 1: {lime_top_1_video.shape}\nLIME Top 2: {lime_top_2_video.shape}\nLIME Top 3: {lime_top_3_video.shape}\nLIME Top 1&2: {lime_top_1_2_video.shape}\n")

        if any(v is None for v in [video, video_sum, video_expl_attention, video_expl_lime,
                                   attention_top_1_video, attention_top_2_video, attention_top_3_video, attention_top_1_2_video,
                                   lime_top_1_video, lime_top_2_video, lime_top_3_video, lime_top_1_2_video]):
            continue

        sentences = []

        # Process video, summary, explanations and individual fragments
        original_video = process_video_fragment(model, processor, device, video, video_prompt, video_prompt_length)
        video_summary = process_video_fragment(model, processor, device, video_sum, video_prompt, video_prompt_length)
        attention_explanation = process_video_fragment(model, processor, device, video_expl_attention, video_prompt, video_prompt_length)
        lime_explanation = process_video_fragment(model, processor, device, video_expl_lime, video_prompt, video_prompt_length)
        attention_top_1 = process_video_fragment(model, processor, device, attention_top_1_video, video_prompt, video_prompt_length)
        attention_top_2 = process_video_fragment(model, processor, device, attention_top_2_video, video_prompt, video_prompt_length)
        attention_top_3 = process_video_fragment(model, processor, device, attention_top_3_video, video_prompt, video_prompt_length)
        attention_top_1_2 = process_video_fragment(model, processor, device, attention_top_1_2_video, video_prompt, video_prompt_length)
        lime_top_1 = process_video_fragment(model, processor, device, lime_top_1_video, video_prompt, video_prompt_length)
        lime_top_2 = process_video_fragment(model, processor, device, lime_top_2_video, video_prompt, video_prompt_length)
        lime_top_3 = process_video_fragment(model, processor, device, lime_top_3_video, video_prompt, video_prompt_length)
        lime_top_1_2 = process_video_fragment(model, processor, device, lime_top_1_2_video, video_prompt, video_prompt_length)

        # gather results (generated texts)
        sentences.append(original_video[0])
        sentences.append(video_summary[0])
        sentences.append(attention_explanation[0])
        sentences.append(lime_explanation[0])
        sentences.append(attention_top_1[0])
        sentences.append(attention_top_2[0])
        sentences.append(attention_top_3[0])
        sentences.append(attention_top_1_2[0])
        sentences.append(lime_top_1[0])
        sentences.append(lime_top_2[0])
        sentences.append(lime_top_3[0])
        sentences.append(lime_top_1_2[0])

        # Save the result to the output folder
        with open(video_result_path, 'a') as file:
            file.write('original_video:\n')
            file.write('\n'.join(original_video[1]) + '\n\n')
            file.write('video_summary:\n')
            file.write('\n'.join(video_summary[1]) + '\n\n')
            file.write('attention_explanation:\n')
            file.write('\n'.join(attention_explanation[1]) + '\n\n')
            file.write('lime_explanation:\n')
            file.write('\n'.join(lime_explanation[1]) + '\n\n')
            file.write('attention_top_1:\n')
            file.write('\n'.join(attention_top_1[1]) + '\n\n')
            file.write('attention_top_2:\n')
            file.write('\n'.join(attention_top_2[1]) + '\n\n')
            file.write('attention_top_3:\n')
            file.write('\n'.join(attention_top_3[1]) + '\n\n')
            file.write('attention_top_1_2:\n')
            file.write('\n'.join(attention_top_1_2[1]) + '\n\n')
            file.write('lime_top_1:\n')
            file.write('\n'.join(lime_top_1[1]) + '\n\n')
            file.write('lime_top_2:\n')
            file.write('\n'.join(lime_top_2[1]) + '\n\n')
            file.write('lime_top_3:\n')
            file.write('\n'.join(lime_top_3[1]) + '\n\n')
            file.write('lime_top_1_2:\n')
            file.write('\n'.join(lime_top_1_2[1]) + '\n\n')

        # compute the SimCSE similarities of each explanation type with the original video
        simcse_attention = compute_simcse_similarity(get_embeddings(original_video[0]), get_embeddings(attention_explanation[0]))
        simcse_lime = compute_simcse_similarity(get_embeddings(original_video[0]), get_embeddings(lime_explanation[0]))

        # compute the SimCSE similarities of each explanation type with the video summary
        simcse_attention_sum = compute_simcse_similarity(get_embeddings(video_summary[0]), get_embeddings(attention_explanation[0]))
        simcse_lime_sum = compute_simcse_similarity(get_embeddings(video_summary[0]), get_embeddings(lime_explanation[0]))

        # compute the SimCSE similarities of each explanation type of each individual fragment (top1, top2, top3 and top1&2) with the original video
        simcse_attention_top_1 = compute_simcse_similarity(get_embeddings(original_video[0]), get_embeddings(attention_top_1[0]))
        simcse_lime_top_1 = compute_simcse_similarity(get_embeddings(original_video[0]), get_embeddings(lime_top_1[0]))
        simcse_attention_top_2 = compute_simcse_similarity(get_embeddings(original_video[0]), get_embeddings(attention_top_2[0]))
        simcse_lime_top_2 = compute_simcse_similarity(get_embeddings(original_video[0]), get_embeddings(lime_top_2[0]))
        simcse_attention_top_3 = compute_simcse_similarity(get_embeddings(original_video[0]), get_embeddings(attention_top_3[0]))
        simcse_lime_top_3 = compute_simcse_similarity(get_embeddings(original_video[0]), get_embeddings(lime_top_3[0]))
        simcse_attention_top_1_2 = compute_simcse_similarity(get_embeddings(original_video[0]), get_embeddings(attention_top_1_2[0]))
        simcse_lime_top_1_2 = compute_simcse_similarity(get_embeddings(original_video[0]), get_embeddings(lime_top_1_2[0]))

        # compute the SimCSE similarities of each explanation type of each individual fragment (top1, top2, top3 and top1&2) with the video summary
        simcse_attention_top_1_sum = compute_simcse_similarity(get_embeddings(video_summary[0]), get_embeddings(attention_top_1[0]))
        simcse_lime_top_1_sum = compute_simcse_similarity(get_embeddings(video_summary[0]), get_embeddings(lime_top_1[0]))
        simcse_attention_top_2_sum = compute_simcse_similarity(get_embeddings(video_summary[0]), get_embeddings(attention_top_2[0]))
        simcse_lime_top_2_sum = compute_simcse_similarity(get_embeddings(video_summary[0]), get_embeddings(lime_top_2[0]))
        simcse_attention_top_3_sum = compute_simcse_similarity(get_embeddings(video_summary[0]), get_embeddings(attention_top_3[0]))
        simcse_lime_top_3_sum = compute_simcse_similarity(get_embeddings(video_summary[0]), get_embeddings(lime_top_3[0]))
        simcse_attention_top_1_2_sum = compute_simcse_similarity(get_embeddings(video_summary[0]), get_embeddings(attention_top_1_2[0]))
        simcse_lime_top_1_2_sum = compute_simcse_similarity(get_embeddings(video_summary[0]), get_embeddings(lime_top_1_2[0]))

        # Create embeddings with SBERT model and similarity matrix
        embeddings = sbert_model.encode(sentences)
        sbert_similarities = sbert_model.similarity(embeddings,embeddings)

        # Gather all results
        similarities_results = []
        similarities_results.append([simcse_attention, simcse_lime, simcse_attention_sum, simcse_lime_sum,
            simcse_attention_top_1, simcse_lime_top_1, simcse_attention_top_2, simcse_lime_top_2,
            simcse_attention_top_3, simcse_lime_top_3, simcse_attention_top_1_2, simcse_lime_top_1_2,
            simcse_attention_top_1_sum, simcse_lime_top_1_sum, simcse_attention_top_2_sum, simcse_lime_top_2_sum,
            simcse_attention_top_3_sum, simcse_lime_top_3_sum, simcse_attention_top_1_2_sum, simcse_lime_top_1_2_sum,
            round(sbert_similarities[0][2].item(), 5), round(sbert_similarities[0][3].item(), 5),
            round(sbert_similarities[1][2].item(), 5), round(sbert_similarities[1][3].item(), 5),
            round(sbert_similarities[0][4].item(), 5), round(sbert_similarities[0][8].item(), 5),
            round(sbert_similarities[0][5].item(), 5), round(sbert_similarities[0][9].item(), 5),
            round(sbert_similarities[0][6].item(), 5), round(sbert_similarities[0][10].item(), 5),
            round(sbert_similarities[0][7].item(), 5), round(sbert_similarities[0][11].item(), 5),
            round(sbert_similarities[1][4].item(), 5), round(sbert_similarities[1][8].item(), 5),
            round(sbert_similarities[1][5].item(), 5), round(sbert_similarities[1][9].item(), 5),
            round(sbert_similarities[1][6].item(), 5), round(sbert_similarities[1][10].item(), 5),
            round(sbert_similarities[1][7].item(), 5), round(sbert_similarities[1][11].item(), 5)])

        # Create DataFrame columns to hold the similarity results
        similarity_columns = [
            'simcse_attention', 'simcse_lime', 'simcse_attention_sum', 'simcse_lime_sum',
            'simcse_attention_top_1', 'simcse_lime_top_1', 'simcse_attention_top_2', 'simcse_lime_top_2',
            'simcse_attention_top_3', 'simcse_lime_top_3', 'simcse_attention_top_1_2', 'simcse_lime_top_1_2',
            'simcse_attention_top_1_sum', 'simcse_lime_top_1_sum', 'simcse_attention_top_2_sum', 'simcse_lime_top_2_sum',
            'simcse_attention_top_3_sum', 'simcse_lime_top_3_sum', 'simcse_attention_top_1_2_sum', 'simcse_lime_top_1_2_sum',
            'sbert_attention', 'sbert_lime', 'sbert_attention_sum', 'sbert_lime_sum',
            'sbert_attention_top_1', 'sbert_lime_top_1', 'sbert_attention_top_2', 'sbert_lime_top_2',
            'sbert_attention_top_3', 'sbert_lime_top_3', 'sbert_attention_top_1_2', 'sbert_lime_top_1_2',
            'sbert_attention_top_1_sum', 'sbert_lime_top_1_sum', 'sbert_attention_top_2_sum', 'sbert_lime_top_2_sum',
            'sbert_attention_top_3_sum', 'sbert_lime_top_3_sum', 'sbert_attention_top_1_2_sum', 'sbert_lime_top_1_2_sum'
        ]

        # Create the dataframe with the scores
        df = pd.DataFrame(similarities_results, columns=similarity_columns)
        # Save the dataframe to a csv file
        csv_output_path = os.path.join(video_subfolder, f"{video_id}_similarities.csv")
        print(f"Saving CSV to: {csv_output_path}")
        df.to_csv(csv_output_path, index=False)
        count = count + 1

    print("\n"*3)

if __name__ == "__main__":
    video_folder = './data'
    video_prompt = "Describe the most prominent objects and events in the video, in 3 sentences. Don't mention background details."

    while True:
        try:
            run(video_folder=video_folder, video_prompt=video_prompt)
            break

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Restarting the process from the last checkpoint...")



