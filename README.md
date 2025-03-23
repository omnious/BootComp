<div align="center">

<h1>BootComp: Controllable Human Image Generation with Personalized Multi-Garments</h1>

<a href='https://omnious.github.io/BootComp/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2411.16801'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://huggingface.co/omniousai/BootComp'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>


</div>



This is the official implementation of the paper ["Controllable Human Image Generation with Personalized Multi-Garments"](https://arxiv.org/abs/2411.16801).

---

![teaser](assets/teaser.png)&nbsp;

Star ⭐ us if you like it!

---




## Requirements

```
git clone https://github.com/omnious/BootComp.git
cd BootComp

conda env create -n bootcomp python=3.10
conda activate bootcomp
pip install -r requirements.txt
```

## Inference


### Decomposition (Synthetic data generation)

You just need to specify the path to the human image from which you want to extract garments, and you're good to go.

```
accelerate launch data_gen.py \
--human_img_path="./example/human.jpg" \
--output_dir="decomp_output"
```

or, you can simply run with the script file.

```
sh data_gen.sh
```

### Composition

You first need to prepare garment images and specify them in a json file following the ./example/info.json.
Your json file should be as follows,

```
{
    "{index}": {
        "{garment category 1}": "{image path of garment1}",
        "{garment category 2}": "{image path of garment2}",
        ...
        "{garment category N}": "{image path of garmentN}",
        "text":{text prompt describing human image}
    }
}
```


With the prepared info_dict.json file, you can generate human images wearing multiple garments. 

```
accelerate launch gen_composition_xl.py \
--output_dir="comp_output" \
--info_path="./example/info.json" \
--num_inference_steps=30 \
--cloth_scale=2.0 \
--guidance_scale=4.0
```

or, you can simply run with the script file.
```
sh gen_composition_xl.sh
```


## Citation
```
@article{choi2024controllable,
  title={Controllable Human Image Generation with Personalized Multi-Garments},
  author={Choi, Yisol and Kwak, Sangkyung and Yu, Sihyun and Choi, Hyungwon and Shin, Jinwoo},
  journal={arXiv preprint arXiv:2411.16801},
  year={2024}
}
```



## License
The codes and checkpoints in this repository are under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).


