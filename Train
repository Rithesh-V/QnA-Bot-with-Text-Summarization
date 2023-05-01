import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, squad_convert_examples_to_features, Trainer, TrainingArguments
from transformers.data.processors.squad import SquadV2Processor


# Define the paths to the training data and output directory
train_file = "train.json"
output_dir = "output"

# Load the training data
with open(train_file) as f:
    train_data = json.load(f)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# Set up the SQuAD v2 processor and convert the training data into features
processor = SquadV2Processor()
train_examples = processor.get_examples_from_dict(train_data, is_training=True)
train_features = squad_convert_examples_to_features(
    examples=train_examples,
    tokenizer=tokenizer,
    max_seq_length=512,
    doc_stride=128,
    max_query_length=64,
    is_training=True,
)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    push_to_hub=False,
    logging_dir="./logs",
)

# Define the trainer and train the model
model = AutoModelForQuestionAnswering.from_pretrained('bert-large-uncased')
trainer = squad.SquadTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Save the trained model
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "model")
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
