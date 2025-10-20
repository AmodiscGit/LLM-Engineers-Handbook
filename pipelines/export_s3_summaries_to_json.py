from pathlib import Path
from zenml import pipeline
from zenml.client import Client
from steps import export as export_steps

@pipeline
def export_s3_summaries_to_json(output_dir: Path = Path("output")) -> None:
    artifact_name = "s3_summarization_etl_pipeline::summarize_documents::output"
    artifact = Client().get_artifact_version(name_id_or_prefix=artifact_name)
    data = export_steps.serialize_artifact(artifact=artifact, artifact_name=artifact_name)
    export_steps.to_json(data=data, to_file=output_dir / f"{artifact_name}.json")