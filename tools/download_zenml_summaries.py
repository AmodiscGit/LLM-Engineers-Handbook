from zenml.client import Client
import json
import os

def download_summarization_artifact(run_id, step_name="summarize_documents", output_dir="output"):
    client = Client()
    # List all artifacts
    artifacts = client.list_artifacts()
    print(f"Debug: Listing all artifacts for run {run_id}")
    found = False
    for artifact in artifacts:
        name = getattr(artifact, 'name', None)
        if name and name == 's3_summarization_etl_pipeline::summarize_documents::output':
            artifact_id = getattr(artifact, 'id', None)
            print(f"Found summarize_documents output artifact for run {run_id}: id={artifact_id}, name={name}")
            found = True
            # Try to fetch artifact details using ZenML API
            try:
                artifact_obj = client.get_artifact(artifact_id)
                print(f"Artifact details for id={artifact_id}: {artifact_obj}")
                # Try to print useful attributes
                print(f"Attributes: {dir(artifact_obj)}")
                # If artifact_obj has a 'uri' or 'path', print it
                if hasattr(artifact_obj, 'uri'):
                    print(f"Artifact URI: {artifact_obj.uri}")
                if hasattr(artifact_obj, 'path'):
                    print(f"Artifact Path: {artifact_obj.path}")
                # If artifact_obj has a method to get content, use it
                # Example: summaries = artifact_obj.load()
                # If not, print a message
                print("If you see no URI/path above, you may need to fetch the artifact from the ZenML dashboard or CLI.")
            except Exception as e:
                print(f"Could not fetch artifact details for id={artifact_id}: {e}")
    if not found:
        print(f"No summarize_documents output artifact found for run {run_id}")

def download_all_pipeline_runs(pipeline_name, step_name="summarize_documents", output_dir="output"):
    client = Client()
    runs = client.list_pipeline_runs(pipeline_name=pipeline_name)
    run_ids = [run.id for run in runs]
    print(f"Found {len(run_ids)} runs for pipeline '{pipeline_name}'.")
    for run_id in run_ids:
        try:
            download_summarization_artifact(run_id, step_name, output_dir)
        except Exception as e:
            print(f"Failed to download summaries for run {run_id}: {e}")

if __name__ == "__main__":
    pipeline_name = "s3_summarization_etl_pipeline"
    download_all_pipeline_runs(pipeline_name)

