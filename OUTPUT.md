# Output Format

## Structure
The output folder contains the results of each model. The results of each model are divided into `itc/` and `itm/` folders. For each task, a JSON file and a folder for storing samples are generated. Each sample contains N correct examples and N incorrect examples (N can be set in `configs/xx.yaml` and the default number is 5). All samples are in the `sample/` directory.

The output folder tree looks like:
```
output/
├── albef
│   ├── itc
│   │   ├── sample
│   │   │   └── Attribute_color_vaw
│   │   │       ├── cor-1.jpg
│   │   │       └── incor-1.jpg
│   │   └── Attribute_color_vaw.json
│   └── itm
│       ├── sample
│       │   └── Attribute_color_vaw
│       │       ├── cor-1.jpg
│       │       └── incor-1.jpg
│       └── Attribute_color_vaw.json
├── ...
```
## Json File
The JSON file records the detailed results of the evaluation, including the list of N correct examples and N incorrect examples, the total accuracy, the number of data, the name of the model, the name of the task and the evaluation time.

For each example in the list, the image path, prediction score, prediction text and prediction result are recorded.

The JSON file for ITC task looks like:
```
{
"sample_correct_outputs": 
    [{"img_path": str, "pos_score": float, "pos_txt": str, "neg_score": float, "neg_txt": str, "result": "correct"},...], 
"sample_incorrect_outputs": 
    [{"img_path": str, "pos_score": float, "pos_txt": str, "neg_score": float, "neg_txt": str, "result": "incorrect"},...], 
"total_acc": float, 
"number_of_data": int, 
"model_name": str, 
"task": "itc", 
"eval_time": float
}
```

The JSON file for ITM task looks like:
```
{
"sample_correct_outputs": 
    [{"img_path": str, "score": float, "text": str, "result": "correct"},...], 
"sample_incorrect_outputs": 
    [{"img_path": str, "score": float, "text": str, "result": "incorrect"},...], 
"total_acc": float, 
"number_of_data": int, 
"model_name": str, 
"task": "itm", 
"eval_time": float
}
```
## Image
For the example of ITC task, we recorded its positive and negative text and score at the bottom of the sample image, such as ![img1](./output/albef/itc/sample/Attribute_color_vaw/cor-5.jpg)

For the example of ITM task, we recorded its text and score at the bottom of the sample image, such as ![img2](./output/albef/itm/sample/Attribute_color_vaw/cor-3.jpg)


