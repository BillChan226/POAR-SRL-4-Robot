
# #utils script for the progressive
import os
import cloudpickle


def _load_from_file(load_path):
    if isinstance(load_path, str):
        if not os.path.exists(load_path):
            if os.path.exists(load_path + ".pkl"):
                load_path += ".pkl"
            else:
                raise ValueError("Error: the file {} could not be found".format(load_path))

        with open(load_path, "rb") as file:
            data, params = cloudpickle.load(file)
    else:
        # Here load_path is a file-like object, not a path
        data, params = cloudpickle.load(load_path)

    return data, params