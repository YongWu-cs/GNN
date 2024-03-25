import os
import io
import json

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("文件夹创建成功！")
    else:
        print("路径已经存在。")

def load_all_file_name(path):
    all_images = []
    for file in os.listdir(path):
        if file.endswith(".jpg") or file.endswith(".png"):
            all_images.append(os.path.join(path, file))

    print("Total images {} in {}".format(len(all_images),path))

    return all_images

def _make_w_io_base(f, mode: str,is_end=False):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode,encoding="UTF-8")
        if mode=="a":
            if f.tell()!=0:
                if is_end==True:
                    f.write("\n")
                else:
                    f.write(",\n")
            else:
                f.write("[")
    return f

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode,encoding="UTF-8")
    return f

def jdump(obj, f, mode="w", is_end=False,indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode,is_end)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

