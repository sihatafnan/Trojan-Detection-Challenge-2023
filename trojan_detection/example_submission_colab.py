import torch
import os
import json
from tqdm import tqdm
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from baselines import PEZColab as PEZ, GBDA, UAT
from eval_utils import evaluate, check_submission
import argparse
# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()

# basic random seed
import os 
import random
import numpy as np 

DEFAULT_RANDOM_SEED = 1004

# https://www.kaggle.com/code/rhythmcam/random-seed-everything
def seed_basic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
# tensorflow random seed 
# import tensorflow as tf 
# def seedTF(seed=DEFAULT_RANDOM_SEED):
#     tf.random.set_seed(seed)
    
# torch random seed
import torch
def seed_torch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
      
# basic + tensorflow + torch 
def seed_everything(seed=DEFAULT_RANDOM_SEED):
    seed_basic(seed)
    # seedTF(seed)
    seed_torch(seed)

seed_everything(DEFAULT_RANDOM_SEED)

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Running trojan detection with baseline methods.")
    
    parser.add_argument(
        "--subtrack",
        type=str,
        default='base',
        choices=['base', 'large'],
        help="The subtrack to run the baseline on"
    )
    parser.add_argument(
        "--phase",
        type=str,
        default='dev',
        choices=['dev', 'test'],
        help="The phase of the competition"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default='PEZ',
        choices=['PEZ', 'GBDA', 'UAT', 'groundtruth'],
        help="The baseline method. If the baseline is 'groundtruth' and mode is 'test', we will evaluate the groundtruth triggers on the test set."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default='val',
        choices=['val', 'test'],
        help="If mode is val, methods are evaluated on a val split of the training trojans. If the mode is test, we run the baseline method on the test targets and produce submission.zip for uploading to CodaLab."
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default='./results.json',
        help="The path used for saving results of the local evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda:0',
        help="The device to load the model into",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print out intermediate results"
    )
    args = parser.parse_args()

    return args

def main():
    
    # ========== load input arguments ========== #
    args = parse_args()
    subtrack = args.subtrack
    phase = args.phase
    baseline = args.baseline
    mode = args.mode
    results_path = args.results_path
    device = args.device
    verbose = args.verbose
    
    # ========== setup baseline methods ========== #
    # Here are example configurations for baseline methods.
    if baseline == 'PEZ':
        method = PEZ()
        method_config = {
            # "num_generate": 20,  # how many triggers to generate for each target
            "num_generate": 8*20,  # how many triggers to generate for each target
            # "batch_size": 40, # 20,  # the number of triggers to generate in parallel (for controlling memory usage)
            # "batch_size": 88, # 20,  # the number of triggers to generate in parallel (for controlling memory usage)
            "batch_size": 8*20, # 20,  # the number of triggers to generate in parallel (for controlling memory usage)
            # "num_optim_tokens": 20,  # the length of the optimized triggers
            # "num_optim_tokens": 10,  # the length of the optimized triggers
            "num_optim_tokens": 50,  # the length of the optimized triggers
            "num_steps": 500, # loss Neil Arm: 2.254
            # "num_steps": 1000,          # loss 2.240
            # "num_steps": 10000, # 
            # "lr": 5e-2,
            # "lr": 1e-1,
            # "lr": 7e-2,
            # "lr": 1e-2,
            # "lr": 1e-3,
            "noise_scale": 1e-3
            # "noise_scale": 1e-2
            # "noise_scale": 0.2
        }
    elif baseline == 'GBDA':
        method = GBDA()
        method_config = {
            "num_generate": 20,
            "batch_size": 20,
            "num_optim_tokens": 20,
            "num_steps": 500,
            "lr": 0.2,
            "noise_scale": 0.2
        }
    elif baseline == 'UAT':
        method = UAT()
        method_config = {
            "num_generate": 20,
            "num_optim_tokens": 20,
            "num_steps": 200
        }
    elif baseline == 'groundtruth' and mode == 'val':
        method = None
    else:
        raise ValueError("unknown baseline")
    # ========== load the tokenizer and the model ========== #
    
    # tokenizer_path = model_path = f"./data/{phase}/{subtrack}/model"
    tokenizer_path = model_path = f"./data/{phase}/{subtrack}/model"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side='left')
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device).eval()

    # Running on the training / local mode
    if mode == 'val':

        # ========== train and evaluate on the validation set ========== #
        if baseline in ['PEZ', 'GBDA', 'UAT']:
            val_fraction = 1.0  # no training required
        else:
            val_fraction = 0.5

        # Load the 200 trojan specifications for training
        trojan_specifications = json.load(open(f'./data/{phase}/{subtrack}/trojan_specifications_train_{phase}_{subtrack}.json', 'r'))

        # Create train and val split
        targets = list(trojan_specifications.keys())
        np.random.shuffle(targets)
        targets_train = targets[int(len(targets) * val_fraction):]
        targets_val = targets[:int(len(targets) * val_fraction)]

        trojan_specifications_train = {}
        trojan_specifications_val = {}

        for target in targets_train:
            trojan_specifications_train[target] = trojan_specifications[target]
        for target in targets_val:
            trojan_specifications_val[target] = trojan_specifications[target]

        if len(targets_train) > 0 and method is not None:
            method.train(trojan_specifications_train, tokenizer, model)

        targets = list(trojan_specifications_val.keys())


        if method is None:
            # When method is None, we output the scores with groundtruth triggers.
            combined_score, recall, reasr = evaluate(trojan_specifications_val, trojan_specifications_val, tokenizer, model)
            print("\n")
            print(f'[Groundtruth] Combined Score: {combined_score:.3f} Recall: {recall:.3f} REASR: {reasr:.3f}')

            return

        # Now we run the baseline trojan detection method on the validation set.
        predictions = method.predict(targets, 
                                     tokenizer, 
                                     model, 
                                     verbose=verbose, 
                                     **method_config)
        
        combined_score, recall, reasr = evaluate(predictions, trojan_specifications_val, tokenizer, model)
        print("\n")
        print(f'[{baseline}] Combined Score: {combined_score:.3f} Recall: {recall:.3f} REASR: {reasr:.3f}')

        # Save the results as a json file
        with open(results_path, 'w') as f:
            json.dump(predictions, f)

    # ========== test and generate submission file ========== #
    # Running on the test mode and we will generate a file for submission.
    elif mode == 'test':
        targets = json.load(open(f'./data/{phase}/targets_test.json', 'r'))

        submission_path = '/content/drive/MyDrive/dataset/a100test/tdc2023/submission'
        predictions_test = method.predict(targets, 
                                          tokenizer, 
                                          model, 
                                          verbose=verbose,
                                          submission_path=submission_path,
                                          **method_config)

        if not os.path.exists(submission_path):
            os.makedirs(submission_path)

        json_path = os.path.join(submission_path, 'predictions.json')
        with open(json_path, 'w') as json_file:
            json.dump(predictions_test, json_file, indent=4)

        print("\n")
        print("Prediction is finished.\n")

        print("Checking if predictions.json contains required keys and values.\n")
        check_submission(submission_path+'/', phase=phase, path_to_tokenizer=tokenizer_path)

        print("Run the following command to make submission.zip for CodaLab.\n")
        print("cd submission && zip ../submission.zip ./* && cd ..")
    
    else:
        raise ValueError(f"mode must be either validation or test, but got {mode}.")

if __name__ == "__main__":
    main()

'''
PEZ: cosine + Adam: Combined Score: 0.158 Recall: 0.108 REASR: 0.209
PEZ: reduce lr + AdamW: Combined Score: 0.154 Recall: 0.104 REASR: 0.203
'''