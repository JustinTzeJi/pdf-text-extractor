import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, model_validator


class contentModel(BaseModel):
    plain: Optional[list]
    html: Optional[list]
    markdown: Optional[list]


class attachementModel(BaseModel):
    url: HttpUrl
    alt: Optional[str]
    file_name: str
    file_type: Literal["pdf", "image", "document"]


class siaranModel(BaseModel):
    language: Literal["en", "ms", "tbd"]
    title: str
    date_published: str | datetime
    type: Literal["kenyataan_media", "ucapan"]
    source: str = Field(pattern=r"^[A-Z]+$")
    content: contentModel
    attachments: List[attachementModel] = []

    @model_validator(mode="before")
    def validate_date_pub(cls, values: Dict[str, Any]):
        if "date_published" in values:
            if values["date_published"] is None:
                values["date_published"] = ""
            if type(values["date_published"]) is str:
                if not values["date_published"]:
                    return values
                date_pub_input = datetime.fromisoformat(values["date_published"])
            else:
                date_pub_input = values["date_published"]
            values["date_published"] = date_pub_input.astimezone(timezone.utc).strftime(
                "%FT%XZ"
            )
            return values
        raise ValueError("date_published Field is missing, input an empty string if date_published is empty")


class longModel(BaseModel):
    data: List[siaranModel]


def upload_subprocess(path_mongoSH_dir: str, upload: bool):
    if not os.path.exists(os.path.join(path_mongoSH_dir, "node_modules")):
        subprocess.run(
            ["npm install"],
            cwd=path_mongoSH_dir,
            shell=True,
            check=True,
        )
    if upload:
        subprocess.run(
            ["node uploadData.mjs"], cwd=path_mongoSH_dir, shell=True, check=True
        )


def validate_one_upload(
    siaran_single: dict, path_mongoSH_dir: str, upload: bool = True
):
    output = siaranModel.model_validate(siaran_single).model_dump_json(indent=4)

    output_dir_path = os.path.join(path_mongoSH_dir, "datas")
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    with open(os.path.join(output_dir_path, "output.json"), "+w") as outFile:
        outFile.write(f"[{output}]")

    upload_subprocess(path_mongoSH_dir, upload)


def validate_long_upload(
    siaran_long: list | longModel, path_mongoSH_dir: str, upload: bool = True, unique: bool = False
):  
    if unique:
        siaran_long = dedup_long(siaran_long)

    if type(siaran_long) is list:
        inp = {"data": siaran_long}
    elif type(siaran_long) is longModel:
        inp = siaran_long
    output = longModel.model_validate(inp).model_dump_json(indent=4)
    output = output.replace('{\n    "data": ', "")[:-2]
    output_dir_path = os.path.join(path_mongoSH_dir, "datas")
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    with open(os.path.join(output_dir_path, "output.json"), "+w") as outFile:
        outFile.write(output)

    upload_subprocess(path_mongoSH_dir, upload)


def _dedup_loop(data_list: list):
    data_dedup = []

    for obj in data_list:
        if obj not in data_dedup:
            data_dedup.append(obj)
    return data_dedup


def dedup_long(siaran_long: list | longModel):
    if type(siaran_long) is longModel:
        siaran_long.data = _dedup_loop(siaran_long.data)
        return siaran_long

    elif type(siaran_long) is list:
        return _dedup_loop(siaran_long)