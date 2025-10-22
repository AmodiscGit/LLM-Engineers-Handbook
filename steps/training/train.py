from zenml import step
import os
from typing import Optional

from llm_engineering.model.finetuning.sagemaker import run_finetuning_on_sagemaker
from llm_engineering.model.finetuning.local_finetune import finetune_local


@step
def train(
    finetuning_type: str,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    learning_rate: float,
    dataset_huggingface_workspace: str = "mlabonne",
    dataset_path: str = "output/all_cleaned_summaries.json",
    output_dir: str = "output/trained_model",
    is_dummy: bool = False,
) -> Optional[dict]:
    """Train locally when `dataset_path` is a local file, otherwise use SageMaker path.

    Returns a small dict with model path on success for inspection.
    """
    # Local path -> run lightweight local finetune
    if os.path.exists(dataset_path):
        model, tokenizer = finetune_local(
            model_name=None,
            output_dir=output_dir,
            dataset_path=dataset_path,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            is_dummy=is_dummy,
        )

        return {"output_dir": output_dir}

    # Fallback to SageMaker flow (existing behavior)
    run_finetuning_on_sagemaker(
        finetuning_type=finetuning_type,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        dataset_huggingface_workspace=dataset_huggingface_workspace,
        is_dummy=is_dummy,
    )

    return None
