""" Code by Nathan Fradet https://github.com/Natooz """
""" Reorganised from his original Jupyter Notebook into a straight-forward code for quick execution on a supercomputing cluster """

from copy import deepcopy
from pathlib import Path
from random import shuffle

from torch import Tensor, argmax
from torch.utils.data import DataLoader
from torch.cuda import is_available as cuda_available, is_bf16_supported
from torch.backends.mps import is_available as mps_available
from transformers import AutoModelForCausalLM, MistralConfig, Trainer, TrainingArguments, GenerationConfig
from transformers.trainer_utils import set_seed
from evaluate import load as load_metric
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetTok, DataCollator
from tqdm import tqdm

# Seed
set_seed(777)

# Our tokenizer's configuration
PITCH_RANGE = (21, 109)
BEAT_RES = {(0, 1): 8, (1, 2): 4, (2, 4): 2, (4, 8): 1}
NUM_VELOCITIES = 24
SPECIAL_TOKENS = ["PAD", "MASK", "BOS", "EOS"]
USE_CHORDS = False
USE_RESTS = False
USE_TEMPOS = True
USE_TIME_SIGNATURE = False
USE_PROGRAMS = False
NUM_TEMPOS = 32
TEMPO_RANGE = (50, 200)  # (min_tempo, max_tempo)
TOKENIZER_PARAMS = {
    "pitch_range": PITCH_RANGE,
    "beat_res": BEAT_RES,
    "num_velocities": NUM_VELOCITIES,
    "special_tokens": SPECIAL_TOKENS,
    "use_chords": USE_CHORDS,
    "use_rests": USE_RESTS,
    "use_tempos": USE_TEMPOS,
    "use_time_signatures": USE_TIME_SIGNATURE,
    "use_programs": USE_PROGRAMS,
    "num_tempos": NUM_TEMPOS,
    "tempo_range": TEMPO_RANGE,
}
config = TokenizerConfig(**TOKENIZER_PARAMS)

# Creates the tokenizer
tokenizer = REMI(config)

# Trains the tokenizer with Byte Pair Encoding (BPE) to build the vocabulary, here 10k tokens
midi_paths = list(Path('Maestro').glob('**/*.mid')) + list(Path('Maestro').glob('**/*.midi'))

print(midi_paths[:5])

tokenizer.learn_bpe(
    vocab_size=10000,
    files_paths=midi_paths,
    start_from_empty_voc=False,
)
tokenizer.save_params("tokenizer.json")

# Split MIDI paths in train/valid/test sets
total_num_files = len(midi_paths)
num_files_valid = round(total_num_files * 0.2)
num_files_test = round(total_num_files * 0.1)
shuffle(midi_paths)
midi_paths_valid = midi_paths[:num_files_valid]
midi_paths_test = midi_paths[num_files_valid:num_files_valid + num_files_test]
midi_paths_train = midi_paths[num_files_valid + num_files_test:]

# Loads tokens and create data collator
kwargs_dataset = {"min_seq_len": 256, "max_seq_len": 1024, "tokenizer": tokenizer}
dataset_train = DatasetTok(midi_paths_train, **kwargs_dataset)
dataset_valid = DatasetTok(midi_paths_valid, **kwargs_dataset)
dataset_test = DatasetTok(midi_paths_test, **kwargs_dataset)
collator = DataCollator(
    tokenizer["PAD_None"], tokenizer["BOS_None"], tokenizer["EOS_None"]
)

model_config = MistralConfig(
    vocab_size=len(tokenizer),
    hidden_size=512,
    intermediate_size=2048,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=4,
    sliding_window=256,
    max_position_embeddings=8192,
    pad_token_id=tokenizer['PAD_None'],
    bos_token_id=tokenizer['BOS_None'],
    eos_token_id=tokenizer['EOS_None'],
)

# Creates model using the correct configuration
model = AutoModelForCausalLM.from_config(model_config)

metrics = {metric: load_metric(metric) for metric in ["accuracy"]}

def compute_metrics(eval_pred):
    """
    Compute metrics for pretraining.

    Must use preprocess_logits function that converts logits to predictions (argmax or sampling).

    :param eval_pred: EvalPrediction containing predictions and labels
    :return: metrics
    """
    predictions, labels = eval_pred
    not_pad_mask = labels != -100
    labels, predictions = labels[not_pad_mask], predictions[not_pad_mask]
    return metrics["accuracy"].compute(predictions=predictions.flatten(), references=labels.flatten())

def preprocess_logits(logits: Tensor, _: Tensor) -> Tensor:
    """
    Preprocess the logits before accumulating them during evaluation.

    This allows to significantly reduce the memory usage and make the training tractable.
    """
    pred_ids = argmax(logits, dim=-1)  # long dtype
    return pred_ids

# Create config for the Trainer
USE_CUDA = cuda_available()
if not cuda_available():
    FP16 = FP16_EVAL = BF16 = BF16_EVAL = False
elif is_bf16_supported():
    BF16 = BF16_EVAL = True
    FP16 = FP16_EVAL = False
else:
    BF16 = BF16_EVAL = False
    FP16 = FP16_EVAL = True
USE_MPS = not USE_CUDA and mps_available()
training_config = TrainingArguments(
    "runs", False, True, True, False, "steps",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=48,
    gradient_accumulation_steps=3,
    eval_accumulation_steps=None,
    eval_steps=1000,
    learning_rate=1e-4,
    weight_decay=0.01,
    max_grad_norm=3.0,
    max_steps=100000,
    lr_scheduler_type="cosine_with_restarts",
    warmup_ratio=0.3,
    log_level="debug",
    logging_strategy="steps",
    logging_steps=20,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=5,
    no_cuda=not USE_CUDA,
    seed=444,
    fp16=FP16,
    fp16_full_eval=FP16_EVAL,
    bf16=BF16,
    bf16_full_eval=BF16_EVAL,
    load_best_model_at_end=True,
    label_smoothing_factor=0.,
    optim="adamw_torch",
    report_to=["tensorboard"],
    gradient_checkpointing=True,
)

collator = DataCollator(tokenizer["PAD_None"], tokenizer["BOS_None"], tokenizer["EOS_None"], copy_inputs_as_labels=True)
trainer = Trainer(
    model=model,
    args=training_config,
    data_collator=collator,
    train_dataset=dataset_train,
    eval_dataset=dataset_valid,
    compute_metrics=compute_metrics,
    callbacks=None,
    preprocess_logits_for_metrics=preprocess_logits,
)

# Training
train_result = trainer.train()
trainer.save_model()  # Saves the tokenizer too
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()


(gen_results_path := Path('gen_res')).mkdir(parents=True, exist_ok=True)
generation_config = GenerationConfig(
    max_new_tokens=512,  # extends samples by 512 tokens
    num_beams=1,        # no beam search
    do_sample=True,     # but sample instead
    temperature=0.9,
    top_k=15,
    top_p=0.95,
    epsilon_cutoff=3e-4,
    eta_cutoff=1e-3,
    pad_token_id=config.padding_token_id,
)

# Here the sequences are padded to the left, so that the last token along the time dimension
# is always the last token of each seq, allowing to efficiently generate by batch
collator.pad_on_left = True
collator.eos_token = None
dataloader_test = DataLoader(dataset_test, batch_size=16, collate_fn=collator)
model.eval()
count = 0
for batch in tqdm(dataloader_test, desc='Testing model / Generating results'):  # (N,T)
    res = model.generate(
        inputs=batch["input_ids"].to(model.device),
        attention_mask=batch["attention_mask"].to(model.device),
        generation_config=generation_config)  # (N,T)

    # Saves the generated music, as MIDI files and tokens (json)
    for prompt, continuation in zip(batch["input_ids"], res):
        generated = continuation[len(prompt):]
        midi = tokenizer.tokens_to_midi([deepcopy(generated.tolist())])
        tokens = [generated, prompt, continuation]  # list compr. as seqs of dif. lengths
        tokens = [seq.tolist() for seq in tokens]
        for tok_seq in tokens[1:]:
            _midi = tokenizer.tokens_to_midi([deepcopy(tok_seq)])
            midi.instruments.append(_midi.instruments[0])
        midi.instruments[0].name = f'Continuation of original sample ({len(generated)} tokens)'
        midi.instruments[1].name = f'Original sample ({len(prompt)} tokens)'
        midi.instruments[2].name = f'Original sample and continuation'
        midi.dump_midi(gen_results_path / f'{count}.mid')
        tokenizer.save_tokens(tokens, gen_results_path / f'{count}.json') 

        count += 1