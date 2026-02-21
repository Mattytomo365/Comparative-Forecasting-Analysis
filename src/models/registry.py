from pathlib import Path
from typing import Mapping, Any
import json, time
'''
Model manifest orchestration - saving, reading, validation
'''
# custom exceptions for domain-specific signals
class RegistryError(Exception): pass # base exception for registry module
class ModelNotFound(RegistryError): pass # RegistryError child
class ManifestError(RegistryError): pass

registry_dir = Path("models")


def save_manifest(kind: str, 
                  stage: str, 
                  target: str, 
                  features: list[str], 
                  params: Mapping[str, Any], 
                  oos_path: Path, 
                  metrics_path: Path, 
                  model: Any) -> None: 
    '''
    Writes model manifest JSON (Human-readable model metadata)
    '''
    manifest = {
        "kind": kind,
        "stage": stage, # baseline or tuned
        "target": target,
        "parameters": params,
        "features": features,
        "artifacts":{
            "oos": {"path": str(oos_path)},
            "metrics": {"path": str(metrics_path)},
        },
        "created_at": time.strftime("%Y%m%d-%H%M%S")
    }
    with open((f"registry/{kind}_{stage}.manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)



def read_manifest(kind: str, stage: str) -> dict[str, Any]:
    '''
    Reads manifest JSON file - used for displaying model information to user
    '''
    path = f"registry/{kind}_{stage}.manifest.json"

    if not path.exists():
        raise ModelNotFound(f"Manifest not found")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e: # if deserialised data is invalid
        raise ManifestError(f"Invalid JSON in {path}: {e}") from e
    return data
