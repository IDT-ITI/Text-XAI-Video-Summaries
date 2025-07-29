# An Experimental Study on Generating Plausible Textual Explanations for Video Summarization

## PyTorch implementation [[Paper](https://arxiv.org/abs/2405.10082)] [[Cite](#citation)]
- This repository provides code and trained models from our paper **"An Experimental Study on Generating Plausible Textual Explanations for Video Summarization"**, by Thomas Eleftheriadis, Evlampios Apostolidis and Vasileios Mezaris, accepted for publication in the Proceedings of the IEEE Int. Conf. on Content-Based Multimedia Indexing (CBMI 2025), Dublin, Ireland, Oct. 2025.
- This software can be used to generate plausible textual explanations for the outcomes of a video summarization model. For the needs of this study, we extend an existing framework for multigranular explanation of video summarization by integrating a state-of-the-art. Large Multimodal Model (Llava-OneVision) and prompting it to produce natural language descriptions of the obtained visual explanations. Following, we focus on one of the most desired characteristics for explainable AI, the plausibility of the obtained explanations that relates with their alignment with the humans’ reasoning and expectations. Using the extended framework, we propose an approach for evaluating the plausibility of visual explanation by quantifying the semantic overlap between their textual descriptions and the textual descriptions of the corresponding video summaries, using two sentence embedding methods (SBERT, SimCSE). Using the extended framework and the proposed plausibility evaluation approach, we conduct an experimental study using a state-of-the-art method (CA-SUM) and two datasets (SumMe, TVSum) for video summarization, aiming to examine whether the more faithful explanations are also the more plausible ones, and identify the most appropriate approach for generating plausible textual explanations for video summarization.


## Main dependencies
The code was developed, checked and verified on an `Ubuntu 20.04.6` PC with an `NVIDIA RTX 4090` GPU and an `i5-12600K` CPU. All dependencies can be found inside the [requirements.txt](requirements.txt) file, which can be used to set up the necessary virtual enviroment.

Regarding the temporal segmentation of the videos, the utilized fragments in our experiments are available in the [data](https://github.com/IDT-ITI/XAI-Video-Summaries/tree/main/data) folder. As stated in our paper, these fragments were produced by the TransNetV2 shot segmentation method (for multi-shot videos) and the motion-driven method for sub-shot segmentation (for single-shot videos), described in [Apostolidis et al. (2018)](https://link.springer.com/chapter/10.1007/978-3-319-73603-7_3). In case there is a need to re-run shot segmentation, please use the code from the [official Github repository](https://github.com/soCzech/TransNetV2) and set-up the necesary environment following the instructions in the aforementioned repository. In case there is a need to also re-run sub-shot segmentation, please contact us for providing access to the utilized method.

The path of the TransNetV2 project, along with its corresponding virtual environment can be set in the [video_segmentation.py](segmentation/video_segmentation.py#L7:L10) file. Please note that the paths for the project are given relatively to the parent directory of this project, while the path of the virtual environments is given relatively to the root directory of the corresponding project.

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

Original videos for each dataset are available in the dataset providers' webpages: 

These videos have to be placed into the `SumMe` and `TVSum' directories of the [data](data) folder.

The extracted deep features for the SumMe and TVSum videos are already available into aforementioned directories. In case there is a need to extract these deep features from scratch (and store them into h5 files), please run the [feature_extraction.py](features/feature_extraction.py) script. Otherwise, an h5 file will be produced automatically for each video and stored into the relevant directory of the [data](data) folder.

The produced h5 files have the following structure:
```Text
/key
    /features                 2D-array with shape (n_steps, feature-dimension)
    /n_frames                 number of frames in original video
```
</div>

The utilized pre-trained models of the [CA-SUM](https://github.com/e-apostolidis/CA-SUM) method, are available within the [models](/explanation/models) directory. Their performance, as well as some other training details, are reported below.
Model| F1 score | Epoch | Split | Reg. Factor
| --- | --- | --- | --- | --- |
summe.pkl | 59.138 | 383 | 4 | 0.5
tvsum.pkl | 63.462 | 44 | 4 | 0.5

## Producing explanations
<div align="justify">

To produce visual and text explanations, and faithfulness and plausibility scores of the SumMe and TVSum datasets, please execute the following commands:
```
bash explain.sh
```

For each video in the datasets, this command will:
- create a new folder (if it does not already exist) in the directory where the video is stored
- extract deep features and define the shots of the video, and store them in h5 and txt files, accordingly (if the files containing these data do not already exist)
- create a subfolder, named **explanation** with the following information: 
  - explanation_and_top_fragments.txt: contains the top fragments (default=3) for each XAI method and the top summary fragments
  - fragments_explanation.txt: contains the ranges of the top- and bottom- scoring explanation fragments for each XAI method 
  - fragments_explanation_evaluation_metrics.csv: contains the scores of the metric Disc+ for each XAI method for each manner (one-by-one or sequentially (batch) )
  - indexes.csv: contains the indices of the explanation fragments for each XAI method

For an individual video run:
```
python explanation/explain.py --model MODEL_PATH --video VIDEO_PATH --fragments NUM_OF_FRAGMENTS (optional, default=3)
```
where, `MODEL_PATH` refers to the path of the trained summarization model, `VIDEO_PATH` refers to the path of the video, and `NUM_OF_FRAGMENTS` refers to the number of utilized video fragments for generating the explanations.

Then run the following command:
```
python explanation/text_explanation.py
```
This will at first prepare each video folder creating the necessary files for the generation of the textual explanations and summary:
- video_$_attention_explanations.txt: contains the top scoring fragments of attention method in temporal order
- video_$_attention_importance.txt: contains the top scoring fragments of attention method
- video_$_lime_explanations.txt: contains the top scoring fragments of lime method in temporal order
- video_$_lime_importance.txt: contains the top scoring fragments of lime method
- video_$_shots.txt: contains the shot segmentation of the video (rename of shots.txt or opt_shots.txt)
- video_$_sum_shots.txt: contains the video summary top fragments.

Then it will call subprocess (LLaVA), copy the data the working directory there, get the text explanation and the similarities scores for each video in each video subfolder, copy the results back to data folder and clean the data folder at working directory:
- video_$_text.txt: Contains the generated textual explanation for each XAI method and for the summary
- video_$_similarities.csv: Contains the SimCSE and SBERT similarity scores between the desired comparisons

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
- Attention Discoverability+ (One by One)
- Attention Discoverability+ (Sequentially)
- Lime Discoverability+ (One by One)
- Lime Discoverability+ (Sequentially)

and
```
python explanation/combine_similarites_files.py
```
Similarly, this will compute the average of the plausibility of the obtained explanations for each explanation method (Attention or LIME) for each sentence embedding method (SimCSE or SBERT). For VideoSet2 it will do the same for both approaches. More particularly it will compute the following explanation/summary similarity scores:
- SimCSE - Attention
- SBERT - Attention
- SimCSE - LIME
- SBERT - LIME

This code runs for Video Set 2 by default. If you want to run the evaluation for Video Set 1 you have clear the data folder and run with the following changes:
- text_explanation.py: at line 22 change _"evaluate2.py"_ to _"evaluate.py"_
- combine_fragment_evaluation_files.py: at line 25 change videoset_key from _"VideoSet2"_ to _"VideoSet1"_
- combine_similarities_files.py: at line 57 change videoset from _"VideoSet2"_ to _"VideoSet1"_


## Citation
<div align="justify">
    
If you find our work, code or trained models useful in your work, please cite the following publication:

T. Eleftheriadis, E. Apostolidis, V. Mezaris, **"An Experimental Study on Generating Plausible Textual Explanations for Video Summarization"**, IEEE Int. Conf. on Content-Based Multimedia Indexing (CBMI 2025), Dublin, Ireland, Oct. 2025.

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

Yoy may also want to have a look at the previous publication, where the methods for extracting the non-textual explanations are presented; these non-textual explanations are the ones that are transformed in the aforementioned CBMI 2025 paper into textual form. This previous publication is:

K. Tsigos, E. Apostolidis, V. Mezaris, **"An Integrated Framework for Multi-Granular Explanation of Video Summarization"**, Frontiers in Signal Processing, vol. 4, 2024. [DOI:10.3389/frsip.2024.1433388](https://doi.org/10.3389/frsip.2024.1433388)

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
</div>

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
