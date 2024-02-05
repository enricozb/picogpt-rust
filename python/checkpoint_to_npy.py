import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def set_in_nested_dict(d, keys, val):
    if not keys:
        return val

    d[keys[0]] = set_in_nested_dict(d.get(keys[0], {}), keys[1:], val)

    return d


def save_array(model_dir: Path, subpath: str, array: np.ndarray):
    path = model_dir / "exploded_model" / subpath
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def main(model_dir: str):
    model_dir = Path(model_dir)

    with open(model_dir / "hparams.json") as hparams_file:
        hparams = json.load(hparams_file)

    params = {"blocks": [{} for _ in range(hparams["n_layer"])]}

    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    for name, _ in tf.train.list_variables(tf_ckpt_path):
        array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
        name = name.removeprefix("model/")
        save_array(model_dir, name, array)

        if name.startswith("h"):
            blockname, rest = name.split("/", 1)
            block_idx = int(blockname.removeprefix("h"))
            set_in_nested_dict(params["blocks"][block_idx], rest.split("/"), name)

        else:
            set_in_nested_dict(params, name.split("/"), name)

    with open(model_dir / "model.json", "w") as model_file:
        json.dump(params, model_file)
        print("dumped ")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
