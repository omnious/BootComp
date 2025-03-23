accelerate launch gen_composition_xl.py \
--output_dir="comp_output" \
--info_path="./example/info.json" \
--num_inference_steps=30 \
--cloth_scale=2.0 \
--guidance_scale=4.0