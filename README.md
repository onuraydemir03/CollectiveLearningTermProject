# CollectiveLearningTermProject
Text to Text Paraphraser Model with Self Training
## Description
This is a project that uses T5 Text-to-text transformer to paraphrase English sentences. It take advantages of self training approach and tests if it is useful or not on this area. 
## Assumption
Two models tested for this task. 
* T5 Small
* T5 Small finetuned on Quora Question Pairs Dataset(mrm8488 on huggingface)

Three datasets used for training

* Paws
* WikiRow
* Tapaco

One dataset used for testing
* SimpleWiki

First two models are trained on whole datasets and then trained with only first 5K sample split. Then the 5K trained models is used to create self training datasets 5K, 10K, 15K, 20K, 25K respectively. These self training samples concatenated with the original labeled first 5K sample. With these 5 new training datasets per dataset, both models trained again from stratch. Then there will be 7 trained checkpoint per dataset and model. All the models has tested on SimpleWiki dataset. Model results compared with Bleu scores. You can see the results below for Bleu-4 for both models.


<p float="left">
  <img src="/T5-Small_Bleu-4.png" width="500" />
  <img src="/T5-Quora Pretrained_Bleu-4.png" width="500" /> 
</p>

## Thanks
I appreciated to work on this field and these very new models. Thanks to M. Fatih Amasyali for this project idea. In addition, thanks to developers that has been a part of T5 Transformer model and creation of these datasets.
