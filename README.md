# Generating Diverse and Consistent QA pairs from Contexts with Information-Maximizing Hierarchical Conditional VAEs IN SPANISH
This is a **version** of the project "Generating Diverse and Consistent QA pairs from Contexts with
Information-Maximizing Hierarchical Conditional VAEs":
[[GitHub]](https://github.com/seanie12/Info-HCVAE) [[Paper]](https://www.aclweb.org/anthology/2020.acl-main.20/) [[Slide]](https://drive.google.com/file/d/17oakiVKIaQ1Y_hSCkGfUIp8P6ORSYjjz/view?usp=sharing) [[Video]](https://slideslive.com/38928851/generating-diverse-and-consistent-qa-pairs-from-contexts-with-informationmaximizing-hierarchical-conditional-vaes).


## Download Spanish SQuAD v2
```bash
mkdir squad
wget https://raw.githubusercontent.com/ccasimiro88/TranslateAlignRetrieve/c84784219785c1fc05884b26081d9b7b4156c019/SQuAD-es-v2.0/train-v2.0-es.json -O ./squad/train-v2.0.json
wget https://raw.githubusercontent.com/ccasimiro88/TranslateAlignRetrieve/c84784219785c1fc05884b26081d9b7b4156c019/SQuAD-es-v2.0/dev-v2.0-es.json -O ./squad/dev-v2.0.json
```

## Deleting unanswered questions
```bash
import json

def quitar_is_impossible(fichero, nombre_nuevo):
  with open(fichero) as json_file:
    data = json.load(json_file)

  for entry in data["data"]:
    paragraphs = entry["paragraphs"]
    for paragraph in paragraphs:
      paragraph["qas"] = [x for x in paragraph["qas"] if x.get('is_impossible') == False]

  with open(nombre_nuevo, "w") as outfile:
    json.dump(data, outfile)

quitar_is_impossible("squad/train-v2.0.json", "squad/train.json")
quitar_is_impossible("squad/dev-v2.0.json", "squad/dev.json")
```

## Dependencies
This code is written in Python. Dependencies include
* python >= 3.6
* pytorch >= 1.4
* json-lines
* tqdm
* [pytorch_scatter](https://github.com/rusty1s/pytorch_scatter)
* [transfomers](https://github.com/huggingface/transformers)

## Train Info-HCVAE
Train Info-HCVAE with the following command. The checkpoint will be save at ./save/vae-checkpoint.
```bash
cd vae
python main.py --model_dir ./model_folder --train_dir ./squad/train.json --dev_dir ./squad/dev.json
```
If you need to load a checkpoint, you must specify the --load_checkpoint option:
```bash
cd vae
python main.py --model_dir ./model_folder --load_checlpoint --train_dir ./squad/train.json --dev_dir ./squad/dev.json
```