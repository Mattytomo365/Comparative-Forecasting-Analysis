from pathlib import Path
import json, time
'''
Model manifest orchestration - saving, reading, validation
'''
# custom exceptions for domain-specific signals
class RegistryError(Exception): pass # base exception for registry module
class ModelNotFound(RegistryError): pass # inherits from RegistryError
class ManifestError(RegistryError): pass

registry_dir = Path("models")

# Writes model manifest JSON (Human-readable model metadata)
def save_manifest(kind, stage, target, features, parameters, oos_path, metrics_path, model): 
    manifest = {
        "kind": kind,
        "stage": stage, # baseline or tuned
        "target": target,
        "parameters": parameters,
        "features": features,
        "artifacts":{
            "oos": {"path": str(oos_path)},
            "metrics": {"path": str(metrics_path)},
        },
        "created_at": time.strftime("%Y%m%d-%H%M%S")
    }
    with open((f"registry/{kind}_{stage}.manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest

# Reads manifest JSON file - used for displaying model information to user
def read_manifest(kind, stage):
    path = f"registry/{kind}_{stage}.manifest.json" # joining path segments

    if not path.exists():
        raise ModelNotFound(f"Manifest not found")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e: # if deserialised data is invalid
        raise ManifestError(f"Invalid JSON in {path}: {e}") from e
    return data

def list_models():
    return