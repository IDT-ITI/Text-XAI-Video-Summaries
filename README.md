# An Experimental Study on Generating Plausible Textual Explanations for Video Summarization

## PyTorch implementation [[Cite](#citation)]
- This repository provides code and trained models from our paper **"An Experimental Study on Generating Plausible Textual Explanations for Video Summarization"**, by Thomas Eleftheriadis, Evlampios Apostolidis and Vasileios Mezaris, accepted for publication in the Proceedings of the IEEE Int. Conf. on Content-Based Multimedia Indexing (CBMI 2025), Dublin, Ireland, Oct. 2025.
- This software can be used to generate plausible textual explanations for the outcomes of a video summarization model. More specifically, our framework produces: a) visual explanations including the video fragments that influenced the most the decisions of the summarizer, using the model-specific (attention-based) and model-agnostic (LIME-based) explanation methods from [Tsigos et al. (2024)](https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2024.1433388/full), and b) plausible textual explanations by integrating a state-of-the-art Large Multimodal Model (Llava-OneVision) and prompting it to produce natural language descriptions of the produced visual explanations. The plausibility of a visual explanation is quantified by measuring the semantic overlap between its textual description and the textual description of the corresponding video summary, using two sentence embedding methods (SBERT, SimCSE). With this framework, a state-of-the-art method (CA-SUM) and two datasets (SumMe, TVSum) for video summarization, we ran experiments to examine whether the more faithful explanations are also the more plausible ones, and identify the most appropriate approach for generating plausible textual explanations for video summarization.
- This repository includes: **TO BE UPDATED**
  - Models of the CA-SUM video summarization method, pretrained on the SumMe and TVSum datasets
  - Information about the temporal segmentation of the videos, as well as instructions on how to obtain this information
  - Extracted deep features for the videos and a script to re-extract them if needed
  - Scripts for extracting visual explanations (for both explanation methods methods)
  - Scripts for generating text explanations calculating the similarity scores (for both SimCSE and SBERT methods)
  - Scripts for the computation of the evaluation metrics
  - Scripts for evaluation of faithfulness and plausibility
  - Script for renaming original video names to desired format

## Main dependencies
The code was developed, checked and verified on an `Ubuntu 20.04.6` PC with an `NVIDIA RTX 4090` GPU and an `i5-12600K` CPU. All dependencies can be found inside the [requirements.txt](requirements.txt) file, which can be used to set up the necessary virtual enviroment.

Regarding the temporal segmentation of the videos, the utilized fragments in our experiments are available in the [data](data) folder. These fragments were produced by the TransNetV2 shot segmentation method (for multi-shot videos) and the motion-driven method for sub-shot segmentation (for single-shot videos), described in [Apostolidis et al. (2018)](https://link.springer.com/chapter/10.1007/978-3-319-73603-7_3). In case there is a need to re-run shot segmentation, please use the code from the [official Github repository](https://github.com/soCzech/TransNetV2) and set-up the necesary environment following the instructions in the aforementioned repository. In case there is a need to also re-run sub-shot segmentation, please contact us for providing access to the utilized method.

The path of the TransNetV2 project, along with its corresponding virtual environment can be set in the [video_segmentation.py](segmentation/video_segmentation.py#L7:L10) file. Please note that the path for the project is given relatively to the parent directory of this project, while the path of the virtual environment is given relatively to the root directory of the corresponding project.

If there is a need to use the default paths:
- Set the name of the root directory of the project to *TransNetV2* and place it in the parent directory of this project.
- Set the name of the virtual environment of the project to *.venv* and place it inside the root directory of the corresponding project.
This will result in the following project structure:
```Text
/Parent Directory
    /TransNetV2
        /.venv
            ...
        ...
    /Text-XAI-Video-Summaries
        ...
```

## Data
<div align="justify">

The videos of the SumMe and TVSum datasets are available [here](https://zenodo.org/records/4884870). These videos have to be placed into the `SumMe` and `TVSum` directories of the [data](data) folder. Following, they have to be renamed according to the utilized naming format, using the provided [rename_videos.py](rename_videos.py) script.

The extracted deep features for the SumMe and TVSum videos are already available into the aforementioned directories. In case there is a need to extract these deep features from scratch (and store them into h5 files), please run the [feature_extraction.py](features/feature_extraction.py) script. Otherwise, an h5 file will be produced automatically for each video and stored into the relevant directory of the [data](data) folder.

The produced h5 files have the following structure:
```Text
/key
    /features                 2D-array with shape (n_steps, feature-dimension)
    /n_frames                 number of frames in original video
```
</div>

The utilized pre-trained models of the [CA-SUM](https://github.com/e-apostolidis/CA-SUM) method are available within the [models](/explanation/models) directory. Their performance, as well as some other training details, are reported below.
Model| F1 score | Epoch | Split | Reg. Factor
| --- | --- | --- | --- | --- |
summe.pkl | 59.14 | 383 | 4 | 0.5
tvsum.pkl | 63.46 | 44 | 4 | 0.5

## Producing explanations
<div align="justify">

To produce visual explanations for the videos of the SumMe and TVSum datasets, and compute faithfulness (Disc+) scores for these explanations, please run the following command:
```
bash explain.sh
```

For each video in these datasets, this command:
- creates a new folder (if it does not already exist) in the directory where the video is stored
- extracts deep features from the video frames and identifies the shots of the video, and stores the obtained data in h5 and txt files, respectively (if the files containing these data do not already exist)
- creates a subfolder, named **visual_explanation**, with the following files: 
  - "_explanation_and_top_fragments.txt_": contains information (indices of the start and end frame) for the selected video fragments by each explanation method (max default = 3), as well as for the fragments of the video summary (default = 3)
  - "_fragments_explanation.txt_": contains a ranking (in descending order) of the video fragments (represented by the indices of the start and end frame) according to the assigned scores by each explanation method
  - "_fragments_explanation_evaluation_metrics.csv_": contains the computed faithfulness (Disc+) scores for each explanation method
  - "_indexes.csv_": contains the indices of the video fragments ranked (in descending order) according to the assigned scores by each explanation method

Then, to produce textual descriptions of the created visual explanations, run the following command:
```
python explanation/text_explanation.py
```

For each video in these datasets, this command:
- creates another subfolder, named **textual_explanation**, with the following files:
  - "_video_id_attention_explanations.txt_": contains the selected video fragments by the attention-based explanation method, in temporal order (i.e. based on their occurence in the video)
  - "_video_id_attention_importance.txt_": contains the selected video fragments by the attention-based explanation method, ranked based on the assigned scores
  - "_video_id_lime_explanations.txt_": contains the selected video fragments by the LIME-based explanation method, in temporal order (i.e. based on their occurence in the video)
  - "_video_id_lime_importance.txt_": contains the selected video fragments by the LIME-based explanation method, ranked based on the assigned scores
  - "_video_id_shots.txt_": contains information (indices of the start and end frame) about the fragments of the video (rename of shots.txt or opt_shots.txt)
  - "_video_id_sum_shots.txt_": contains information (indices of the start and end frame) about the fragments of the video summary
- calls a subprocess, named LLAVA, which creates the following files:
  - "_video_id_text.txt_": contains the generated textual descriptions of the visual explanations and the video summary
  - "_video_id_similarities.csv_": contains the computed SimCSE and SBERT scores for these textual explanations


To produce visual explanations for an individual video using both the model-specific (attention-based) and model-agnostic (LIME-based) methods of the framework, run:
```
python explanation/explain.py --model MODEL_PATH --video VIDEO_PATH --fragments NUM_OF_FRAGMENTS
```
where, `MODEL_PATH` refers to the path of the trained summarization model, `VIDEO_PATH` refers to the path of the video, and `NUM_OF_FRAGMENTS` refers to the number of utilized video fragments for generating the explanations (optional; default = 3).

Then, to produce textual explanations for this video, run:
```
python explanation/text_explain.py --model MODEL_PATH --video VIDEO_PATH --fragments NUM_OF_FRAGMENTS
```

</div>

## Evaluation results
<div align="justify">

This extended framework was evaulated on 2 subsets of SumMe and TVSum datasets. Video Set 1 contains the videos that have at least 1 top-scoring fragment by the explanation methods. 
Video Set 2 includes a subset of videos that have at least 3 top-scoring fragments by the explanation methods.

To get the evaluation results, run:
```
python explanation/combine_fragment_evaluation_files.py
```
This will compute the average of faithfulness (in terms of Discoverability+) of the obtained visual explanations from the fragments_explanation_evaluation_metrics.csv files of each video and will create a folder “final_scores” with the *.csv files containing the averages of the metrics and more particularly:
- Attention Disc+
- LIME Disc+

and
```
python explanation/combine_similarites_files.py
```
Similarly, this will compute the average of the plausibility of the obtained explanations for each explanation method (Attention or LIME) for each sentence embedding method (SimCSE or SBERT). For VideoSet2 it will do the same for both approaches. More particularly it will compute the following explanation/summary similarity scores:
- SimCSE - Attention
- SBERT - Attention
- SimCSE - LIME
- SBERT - LIME

The code runs for Video Set 2 by default. If you want to run the evaluation for Video Set 1 you have to clear the data folder and run with the following changes:
- [text_explanation.py](explanation/text_explanation.py#L22) at line 22 change _"evaluate2.py"_ to _"evaluate.py"_
- [combine_fragment_evaluation_files.py](explanation/combine_fragment_evaluation_files.py#L25) at line 25 change videoset_key from _"VideoSet2"_ to _"VideoSet1"_
- [combine_similarities_files.py](explanation/combine_similarities_files.py#L57) at line 57 change videoset from _"VideoSet2"_ to _"VideoSet1"_


## Citation
<div align="justify">
    
If you find our work, code or trained models useful in your work, please cite the following publication:

T. Eleftheriadis, E. Apostolidis, V. Mezaris, **"An Experimental Study on Generating Plausible Textual Explanations for Video Summarization"**, IEEE Int. Conf. on Content-Based Multimedia Indexing (CBMI 2025), Dublin, Ireland, Oct. 2025.
</div>

BibTeX:

```
@inproceedings{eleftheriadis2025cbmi,
      title={An Experimental Study on Generating Plausible Textual Explanations for Video Summarization}, 
      author={Thomas Eleftheriadis and Evlampios Apostolidis and Vasileios Mezaris},
      booktitle={IEEE Int. Conf. on Content-Based Multimedia Indexing (CBMI 2025)},
      year={2025},
      organization={IEEE}
}
```

<div align="justify">

Yoy may also want to have a look at our previous publication, where extracting non-textual explanations was presented:

K. Tsigos, E. Apostolidis, V. Mezaris, **"An Integrated Framework for Multi-Granular Explanation of Video Summarization"**, Frontiers in Signal Processing, vol. 4, 2024. [DOI:10.3389/frsip.2024.1433388](https://doi.org/10.3389/frsip.2024.1433388)
</div>

BibTeX:

```
@ARTICLE{10.3389/frsip.2024.1433388,
    AUTHOR={Tsigos, Konstantinos  and Apostolidis, Evlampios  and Mezaris, Vasileios },
    TITLE={An integrated framework for multi-granular explanation of video summarization},
    JOURNAL={Frontiers in Signal Processing},
    VOLUME={4},
    YEAR={2024},
    URL={https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2024.1433388},
    DOI={10.3389/frsip.2024.1433388},
    ISSN={2673-8198},
}
```

## License
<div align="justify">
    
Copyright (c) 2025, Thomas Eleftheriadis, Evlampios Apostolidis, Vasileios Mezaris / CERTH-ITI. All rights reserved. This code is provided for academic, non-commercial use only. Redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
</div>

## Acknowledgement
<div align="justify"> 

This work was supported by the EU Horizon Europe programme under grant agreement 101070109 TransMIXR.
</div>
