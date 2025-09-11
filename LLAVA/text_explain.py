import argparse
import av
import os
import logging
import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig, AutoModel, AutoTokenizer

# Skip printing "Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation."
logging.getLogger("transformers").setLevel(logging.ERROR)

#Load SimCSE and SBERT models
simcse_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
simcse_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
sbert_model = SentenceTransformer("all-mpnet-base-v2")


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

def read_video_pyav(container, intervals, downsample_rate, num_frames):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        intervals (list of tuples): A list of tuples, where each tuple represents a start and end frame index (inclusive) to extract from the video.
        downsample_rate (int): The rate at which frames are sampled.
        num_frames (int): max frames
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

def create_alternative_sequential_explanation(model, processor, device, text1, text2, text3, prompt_2, indices):

    new_prompt_length = len(' assistant ' + text1 + text2 + text3 + 'Description1:\n '*3 + ' user ' + prompt_2 + '  assistant ')
    sorted_indices = sorted(range(len(indices)), key=lambda i: indices[i][0])
    texts = [text1, text2, text3]
    sorted_texts = [texts[i] for i in sorted_indices]

    refined_conversation = [
        {
            "role":"assistant",
            "content": [
                {"type": "text", "text": "Description 1:\n" + sorted_texts[0]},
                {"type": "text", "text": "Description 2:\n" + sorted_texts[1]},
                {"type": "text", "text": "Description 3:\n" + sorted_texts[2]},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_2}
            ]
        }
    ]

    refined_prompt = processor.apply_chat_template(refined_conversation, add_generation_prompt=True)

    text_input = processor(text=refined_prompt, padding=False, return_tensors="pt")
    text_input = {key: value.to(device) for key, value in text_input.items()}

    text_out = model.generate(**text_input, max_new_tokens=700)
    text_result = processor.batch_decode(text_out, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    text_result_final = text_result[0][new_prompt_length:]
    text_words = text_result_final.split()
    text_chunks = [' '.join(text_words[i:i + 20]) for i in range(0, len(text_words), 20)]

    return text_result_final, text_chunks

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
    return round(similarity.item(), 5)

def run(video_file, video_prompt, model, processor):
    """
    Processes videos from a given folder, generates text descriptions using a LlavaOnevision model,
    and calculates the similarity between descriptions of the original video and its explanation.
    Args:
        video_file (str): Path to the folder containing video subfolders.
        video_prompt (str): Text prompt to be used for video description generation.
    """

    video_conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video"},
            ],
        },
    ]

    #calculation of characters length to be removed for more readable results
    video_prompt_length = len('user ' + video_prompt + 'assistant ')

    # Split into directory and filename
    video_dir = video_file[:-4]
    video_id = os.path.basename(video_dir)
    visual_expl_dir = os.path.join(video_dir, "visual_explanation")

    # Save result as a .txt file alongside the video (or wherever you prefer)
    text_expl_dir = os.path.join(video_dir, "text_explanation")
    os.makedirs(text_expl_dir, exist_ok=True)

    video_result_path = os.path.join(text_expl_dir, f"{video_id}.txt")

    # Check if the result file already exists
    if os.path.exists(video_result_path):
        print(f"Output already exists for {video_id}, skipping...")
        return

    container = av.open(video_file)

    # txt files with the fragments of the summary and the explanation methods (both in temportal order and importance order)
    video_sum_txt_path = os.path.join(visual_expl_dir, f"sum_shots.txt")
    attention_txt_file_path = os.path.join(visual_expl_dir, f"attention_explanations.txt")
    lime_txt_file_path = os.path.join(visual_expl_dir, f"lime_explanations.txt")
    attention_top_txt_path = os.path.join(visual_expl_dir, f"attention_importance.txt")
    lime_top_txt_path = os.path.join(visual_expl_dir, f"lime_importance.txt")

    # Convert fragment ranges to tuples
    video_sum_indices = convert_ranges_to_tuples(video_sum_txt_path)
    attention_expl_indices = convert_ranges_to_tuples(attention_txt_file_path)
    lime_expl_indices = convert_ranges_to_tuples(lime_txt_file_path)
    attention_top_fragments = convert_ranges_to_tuples(attention_top_txt_path)
    attention_top_1_fragment = [attention_top_fragments[0]]
    attention_top_2_fragment = [attention_top_fragments[1]] if len(attention_top_fragments) >= 2 else []
    attention_top_3_fragment = [attention_top_fragments[2]] if len(attention_top_fragments) >= 3 else []
    lime_top_fragments = convert_ranges_to_tuples(lime_top_txt_path)
    lime_top_1_fragment = [lime_top_fragments[0]]
    lime_top_2_fragment = [lime_top_fragments[1]] if len(lime_top_fragments) >= 2 else []
    lime_top_3_fragment = [lime_top_fragments[2]] if len(lime_top_fragments) >= 3 else []

    device = next(model.parameters()).device
    video_prompt = processor.apply_chat_template(video_conversation, add_generation_prompt=True)

    video_sum = read_video_pyav(container, video_sum_indices, downsample_rate=15, num_frames=150)
    video_expl_attention = read_video_pyav(container, attention_expl_indices, downsample_rate=15, num_frames=150) if len(attention_expl_indices) >= 3 else None
    video_expl_lime = read_video_pyav(container, lime_expl_indices, downsample_rate=15, num_frames=150) if len(lime_expl_indices) >= 3 else None
    attention_top_1_video = read_video_pyav(container, attention_top_1_fragment, downsample_rate=15, num_frames=150)
    attention_top_2_video = read_video_pyav(container, attention_top_2_fragment, downsample_rate=15, num_frames=150) if attention_top_2_fragment is not None else None
    attention_top_3_video = read_video_pyav(container, attention_top_3_fragment, downsample_rate=15, num_frames=150) if attention_top_3_fragment is not None else None
    lime_top_1_video = read_video_pyav(container, lime_top_1_fragment, downsample_rate=15, num_frames=150)
    lime_top_2_video = read_video_pyav(container, lime_top_2_fragment, downsample_rate=15, num_frames=150) if lime_top_2_fragment is not None else None
    lime_top_3_video = read_video_pyav(container, lime_top_3_fragment, downsample_rate=15, num_frames=150) if lime_top_3_fragment is not None else None

    if any(v is None for v in [video_sum, attention_top_1_video, lime_top_1_video]):
        return

    # Process video, summary, explanations and individual fragments
    video_summary = process_video_fragment(model, processor, device, video_sum, video_prompt, video_prompt_length)
    attention_explanation = process_video_fragment(model, processor, device, video_expl_attention, video_prompt, video_prompt_length) if video_expl_attention is not None else None
    lime_explanation = process_video_fragment(model, processor, device, video_expl_lime, video_prompt, video_prompt_length) if video_expl_lime is not None else None
    attention_top_1 = process_video_fragment(model, processor, device, attention_top_1_video, video_prompt, video_prompt_length)
    attention_top_2 = process_video_fragment(model, processor, device, attention_top_2_video, video_prompt, video_prompt_length) if attention_top_2_video is not None else None
    attention_top_3 = process_video_fragment(model, processor, device, attention_top_3_video, video_prompt, video_prompt_length) if attention_top_3_video is not None else None
    lime_top_1 = process_video_fragment(model, processor, device, lime_top_1_video, video_prompt, video_prompt_length)
    lime_top_2 = process_video_fragment(model, processor, device, lime_top_2_video, video_prompt, video_prompt_length) if lime_top_2_video is not None else None
    lime_top_3 = process_video_fragment(model, processor, device, lime_top_3_video, video_prompt, video_prompt_length) if lime_top_3_video is not None else None

    attention_alternative_explanation = (
        create_alternative_sequential_explanation(model, processor, device,
                                                  attention_top_1[0], attention_top_2[0], attention_top_3[0],
                                                  prompt_2, attention_top_fragments)
        if attention_top_1 is not None and attention_top_2 is not None and attention_top_3 is not None
        else None
    )
    lime_alternative_explanation = (
        create_alternative_sequential_explanation(model, processor, device,
                                                  lime_top_1[0], lime_top_2[0], lime_top_3[0],
                                                  prompt_2, lime_top_fragments)
        if lime_top_1 is not None and lime_top_2 is not None and lime_top_3 is not None
        else None
    )

    # Save the result to the output folder
    with open(video_result_path, 'a') as file:
        file.write('video_summary:\n')
        file.write('\n'.join(video_summary[1]) + '\n\n')
        if attention_explanation:
            file.write('attention_explanation:\n')
            file.write('\n'.join(attention_explanation[1]) + '\n\n')
        if attention_alternative_explanation:
            file.write('alternative_attention_explanation:\n')
            file.write('\n'.join(attention_alternative_explanation[1]) + '\n\n')
        if lime_explanation:
            file.write('lime_explanation:\n')
            file.write('\n'.join(lime_explanation[1]) + '\n\n')
        if lime_alternative_explanation:
            file.write('alternative_lime_explanation:\n')
            file.write('\n'.join(lime_alternative_explanation[1]) + '\n\n')
        file.write('attention_top_1:\n')
        file.write('\n'.join(attention_top_1[1]) + '\n\n')
        file.write('lime_top_1:\n')
        file.write('\n'.join(lime_top_1[1]) + '\n\n')

    # If there are 3 fragments for each explanation method and therefore the explanations exist
    if attention_explanation and lime_explanation and attention_alternative_explanation and lime_alternative_explanation:

        sentences = []
        sentences.append(video_summary[0])                      # 0
        sentences.append(attention_explanation[0])              # 1
        sentences.append(attention_alternative_explanation[0])  # 2
        sentences.append(lime_explanation[0])                   # 3
        sentences.append(lime_alternative_explanation[0])       # 4
        sentences.append(attention_top_1[0])                    # 5
        sentences.append(lime_top_1[0])                         # 6

        # compute the SimCSE similarities of each explanation type with the video summary
        simcse_summary_attention = compute_simcse_similarity(get_embeddings(sentences[0]), get_embeddings(sentences[1]))
        simcse_summary_lime = compute_simcse_similarity(get_embeddings(sentences[0]), get_embeddings(sentences[3]))
        simcse_summary_alternative_attention = compute_simcse_similarity(get_embeddings(sentences[0]), get_embeddings(sentences[2]))
        simcse_summary_alternative_lime = compute_simcse_similarity(get_embeddings(sentences[0]), get_embeddings(sentences[4]))
        simcse_summary_attention_top_1 = compute_simcse_similarity(get_embeddings(sentences[0]), get_embeddings(sentences[5]))
        simcse_summary_lime_top_1 = compute_simcse_similarity(get_embeddings(sentences[0]), get_embeddings(sentences[6]))

        # Create embeddings with SBERT model and similarity matrix
        embeddings = sbert_model.encode(sentences)
        sbert_similarities = sbert_model.similarity(embeddings, embeddings)

        # Gather all results
        similarities_results = [[round(sbert_similarities[0][1].item(), 5), simcse_summary_attention,
                                 round(sbert_similarities[0][3].item(), 5), simcse_summary_lime,
                                 round(sbert_similarities[0][2].item(), 5), simcse_summary_alternative_attention,
                                 round(sbert_similarities[0][4].item(), 5), simcse_summary_alternative_lime,
                                 round(sbert_similarities[0][5].item(), 5), simcse_summary_attention_top_1,
                                 round(sbert_similarities[0][6].item(), 5), simcse_summary_lime_top_1
                                 ]]

        # Create DataFrame columns to hold the similarity results
        similarity_columns = ['sbert_attention_sum', 'simcse_attention_sum',
                              'sbert_lime_sum', 'simcse_lime_sum',
                              'sbert_alternative_attention_sum', 'simcse_alternative_attention_sum',
                              'sbert_alternative_lime_sum', 'simcse_alternative_lime_sum',
                              'sbert_attention_top_1_sum', 'simcse_attention_top_1_sum',
                              'sbert_lime_top_1_sum', 'simcse_lime_top_1_sum'
                              ]
        # Create the dataframe with the scores
        df = pd.DataFrame(similarities_results, columns=similarity_columns)
        output_file = os.path.join(text_expl_dir, f"{video_id}_similarities.csv")
        df.to_csv(output_file, index=False)

    else:
        sentences = []
        sentences.append(video_summary[0])      # 0
        sentences.append(attention_top_1[0])    # 1
        sentences.append(lime_top_1[0])         # 2

        simcse_summary_attention_top_1 = compute_simcse_similarity(get_embeddings(sentences[0]), get_embeddings(sentences[1]))
        simcse_summary_lime_top_1 = compute_simcse_similarity(get_embeddings(sentences[0]), get_embeddings(sentences[2]))

        # Create embeddings with SBERT model and similarity matrix
        embeddings = sbert_model.encode(sentences)
        sbert_similarities = sbert_model.similarity(embeddings, embeddings)

        # Gather all results
        similarities_results = [[round(sbert_similarities[0][1].item(), 5), simcse_summary_attention_top_1,
                                 round(sbert_similarities[0][2].item(), 5), simcse_summary_lime_top_1
                                 ]]

        # Create DataFrame columns to hold the similarity results
        similarity_columns = ['sbert_attention_top_1_sum', 'simcse_attention_top_1_sum',
                              'sbert_lime_top_1_sum', 'simcse_lime_top_1_sum'
                              ]

        # Create the dataframe with the scores
        df = pd.DataFrame(similarities_results, columns=similarity_columns)
        output_file = os.path.join(text_expl_dir, f"{video_id}_similarities.csv")
        df.to_csv(output_file, index=False)

if __name__ == "__main__":

    # Parse the running parameters
    parser = argparse.ArgumentParser(description="Text explanation runner")
    # Specify the path of the video/data folder
    parser.add_argument(
        "-d", "--data", nargs="+",
        help="Path(s) of video file(s) or a folder",
        required=True
    )
    argument = parser.parse_args()

    prompt = "Describe the most prominent objects and events in the video, in 3 sentences. Don't mention background details."
    prompt_2 = "Write a brief summary that covers all 3 descriptions equally. Avoid assumptions and background details."

    # load llava model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf",
                                                                       quantization_config=quantization_config,
                                                                       device_map="auto")

    processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")

    # Flatten input: could be one video, multiple videos, or a folder
    video_files = []
    for path in argument.data:
        if os.path.isdir(path):
            for f in sorted(os.listdir(path)):
                if f.endswith(".mp4"):
                    video_files.append(os.path.join(path, f))
        elif path.endswith(".mp4"):
            video_files.append(path)

    for vf in video_files:
        try:
            run(video_file=vf, video_prompt=prompt, model=model, processor=processor)
        except Exception as e:
            print(f"Error with {vf}: {e}")
