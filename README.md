# SummScore
Source code for the paper <a href="https://arxiv.org/pdf/2212.09593.pdf" style = "text-decoration:none;color:#4682B4">Unsupervised Summarization Re-ranking</a>.

Mathieu Ravaut, Shafiq Joty, Nancy F. Chen.

Accepted for publication at ACL Findings 2023. 

## 1 - Download the code
```
git clone https://github.com/Ravoxsg/SummScore.git
cd SummScore
```

## 2 - Install the dependencies
```
conda create --name summscore python=3.8.11
conda activate summscore
pip install -r requirements.txt
```

## 3 - Generate summary candidates
SummScore scores each summary candidate produced by a model (e.g, PEGASUS) and a decoding method (e.g., beam search) on a given data point. 

You need to generate candidates on the validation and training sets:

For instance on SAMSum 100-shot validation set (default code):
```
cd src/candidate_generation/
CUDA_VISIBLE_DEVICES=0 bash main_candidate_generation.sh
```

## 4 - Score summary candidates
Next, you need to score each summary candidate.

```
cd ../summscore/
CUDA_VISIBLE_DEVICES=0 bash main_build_scores.sh
```

## 5 - Train SummScore
Now we can launch SummScore training, which will estimate features coefficients on a 1000 data points subset of the validation set.

```
CUDA_VISIBLE_DEVICES=0 bash main_reranking.sh
```

The code lets you choose among several fine-tuned models hosted on HuggingFace. You can also use our own checkpoints:  
BART fine-tuned on WikiHow: <a href="https://drive.google.com/file/d/1Rb03FYE612wMy9E5GvwxxXfi-SxZPfg3/view?usp=share_link" style = "text-decoration:none;color:#4682B4">here</a>    
PEGASUS fine-tuned on SAMSum: <a href="https://drive.google.com/file/d/1dcDAcg-qTqaP5xWa1f-EUvfxORHTCVps/view?usp=share_link" style = "text-decoration:none;color:#4682B4">here</a>   
BART fine-tuned on SAMSum: <a href="https://drive.google.com/file/d/1OSMws143gfNphHL3smTD7JuDVaKtm5p8/view?usp=share_link" style = "text-decoration:none;color:#4682B4">here</a>   

## DEMO 
Alternatively, if you just want a demo (in a single file) of SummScore on a single data point (default: CNN/DM), run:
```
cd src/summscore/
CUDA_VISIBLE_DEVICES=0 python demo.py
```

## Citation
If you find our paper or this project helps your research, please kindly consider citing our paper in your publication.   
```
@article{ravaut2022unsupervised,
  title={Unsupervised Summarization Re-ranking},
  author={Ravaut, Mathieu and Joty, Shafiq and Chen, Nancy},
  journal={arXiv preprint arXiv:2212.09593},
  year={2022}
}


