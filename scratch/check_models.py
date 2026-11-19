import os, pathlib, json

base = pathlib.Path(r"C:\Users\bj_iv\.cache\huggingface\hub")

for model_dir in sorted(base.iterdir()):
    if not model_dir.name.startswith("models--"):
        continue
    for cfg in model_dir.rglob("config.json"):
        try:
            data = json.loads(cfg.read_text())
            hidden = data.get("hidden_size", data.get("d_model", "?"))
            arch = data.get("architectures", data.get("model_type", "?"))
            print(f"{model_dir.name}: hidden_size={hidden}, arch={arch}")
            break
        except Exception:
            pass
