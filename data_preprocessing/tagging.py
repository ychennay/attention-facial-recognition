import boto3
import os
from core_config import ROOT_DIR
from enum import Enum
from typing import Dict, Any, List, Type
import re
import json

Path = str


class Genders(Enum):
    MALE = "male"
    FEMALE = "female"

class FacialExpressions(Enum):
    SMILING = "smiling"
    NOT_SMILING = "not_smiling"

class MetadataStore:
    classification_dictionary: Dict[str, Dict[str, Any]] = {}
    easy_classification_dictionary: Dict[str, Dict[str, Any]] = {}


IMAGE_NAME_REGEX = r'^([0-9]{1,3})-([0-9]{1,2}|[a-b])\.jpg$'
EASY_IMAGE_NAME_REGEX = r'^([0-9]{1,3})[a-b]\.jpg$'
S3_BUCKET_NAME = "fei-faces-sao-paulo"
LOCAL_JSON_FILENAME = "gender.json"


def populate_classification_dictionary( dimension_enums: Type[Enum], dimension: str = "gender") -> None:
    absolute_path: Path = os.path.join(ROOT_DIR, "face_images", "classification_tags", dimension)
    for dimension_enum in dimension_enums:
        dimension_directory: Path = os.path.join(absolute_path, dimension_enum.value)
        image_names: List[str] = os.listdir(dimension_directory)
        for image_name in image_names:
            match = re.search(IMAGE_NAME_REGEX, image_name)
            if match:
                subject_id: str = match.group(1)
                if subject_id not in MetadataStore.classification_dictionary.keys():
                    MetadataStore.classification_dictionary[image_name] = dimension_enum.value
            else:
                easy_match = re.search(EASY_IMAGE_NAME_REGEX, image_name)
                assert easy_match, f"{image_name} was not a match for any known filename patterns."
                subject_id: str = easy_match.group(1)
                if subject_id not in MetadataStore.easy_classification_dictionary.keys():
                    MetadataStore.easy_classification_dictionary[image_name] = dimension_enum.value
    print(MetadataStore.classification_dictionary)


if __name__ == "__main__":
    dimension = "smiling"
    easy = False

    LOCAL_JSON_FILENAME = f"{dimension}.json"

    populate_classification_dictionary(FacialExpressions, dimension=dimension)

    with open(LOCAL_JSON_FILENAME, "w") as json_file:
        json.dump(MetadataStore.classification_dictionary, json_file)

    s3 = boto3.resource('s3')
    faces_bucket = s3.Bucket(S3_BUCKET_NAME)

    faces_bucket.upload_file(LOCAL_JSON_FILENAME, f"classification/{LOCAL_JSON_FILENAME}")
    print(f"Uploaded {LOCAL_JSON_FILENAME}")

    if easy:
        easy_json_filename = "easy_" + LOCAL_JSON_FILENAME
        with open(easy_json_filename, "w") as easy_json_file:
            json.dump(MetadataStore.easy_classification_dictionary, easy_json_file)

        faces_bucket.upload_file(easy_json_filename, f"classification/{easy_json_filename}")
        print(f"Uploaded {easy_json_filename}")
