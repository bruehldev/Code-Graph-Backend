import json
import os

import datasets
from tqdm.autonotebook import tqdm

_CITATION = """
@inproceedings{ding2021few,
title={Few-NERD: A Few-Shot Named Entity Recognition Dataset},
author={Ding, Ning and Xu, Guangwei and Chen, Yulin, and Wang, Xiaobin and Han, Xu and Xie, 
Pengjun and Zheng, Hai-Tao and Liu, Zhiyuan},
booktitle={ACL-IJCNLP},
year={2021}
}
"""

_DESCRIPTION = """
Few-NERD is a large-scale, fine-grained manually annotated named entity recognition dataset, 
which contains 8 coarse-grained types, 66 fine-grained types, 188,200 sentences, 491,711 entities 
and 4,601,223 tokens. Three benchmark tasks are built, one is supervised: Few-NERD (SUP) and the 
other two are few-shot: Few-NERD (INTRA) and Few-NERD (INTER).
"""

_LICENSE = "CC BY-SA 4.0"

# the original data files (zip of .txt) can be downloaded from tsinghua cloud, but we chose to host them on huggingface.co
# for better reliability and download speed
_URL = "https://huggingface.co/datasets/DFKI-SLT/few-nerd/resolve/main/data"
_URLs = {"supervised": f"{_URL}/supervised.zip", "intra": f"{_URL}/intra.zip", "inter": f"{_URL}/inter.zip"}

# the label ids, for coarse(NER_TAGS_DICT) and fine(FINE_NER_TAGS_DICT)
NER_TAGS_DICT = {
    "O": 0,
    "art": 1,
    "building": 2,
    "event": 3,
    "location": 4,
    "organization": 5,
    "other": 6,
    "person": 7,
    "product": 8,
}

FINE_NER_TAGS_DICT = {
    "O": 0,
    "art-broadcastprogram": 1,
    "art-film": 2,
    "art-music": 3,
    "art-other": 4,
    "art-painting": 5,
    "art-writtenart": 6,
    "building-airport": 7,
    "building-hospital": 8,
    "building-hotel": 9,
    "building-library": 10,
    "building-other": 11,
    "building-restaurant": 12,
    "building-sportsfacility": 13,
    "building-theater": 14,
    "event-attack/battle/war/militaryconflict": 15,
    "event-disaster": 16,
    "event-election": 17,
    "event-other": 18,
    "event-protest": 19,
    "event-sportsevent": 20,
    "location-GPE": 21,
    "location-bodiesofwater": 22,
    "location-island": 23,
    "location-mountain": 24,
    "location-other": 25,
    "location-park": 26,
    "location-road/railway/highway/transit": 27,
    "organization-company": 28,
    "organization-education": 29,
    "organization-government/governmentagency": 30,
    "organization-media/newspaper": 31,
    "organization-other": 32,
    "organization-politicalparty": 33,
    "organization-religion": 34,
    "organization-showorganization": 35,
    "organization-sportsleague": 36,
    "organization-sportsteam": 37,
    "other-astronomything": 38,
    "other-award": 39,
    "other-biologything": 40,
    "other-chemicalthing": 41,
    "other-currency": 42,
    "other-disease": 43,
    "other-educationaldegree": 44,
    "other-god": 45,
    "other-language": 46,
    "other-law": 47,
    "other-livingthing": 48,
    "other-medical": 49,
    "person-actor": 50,
    "person-artist/author": 51,
    "person-athlete": 52,
    "person-director": 53,
    "person-other": 54,
    "person-politician": 55,
    "person-scholar": 56,
    "person-soldier": 57,
    "product-airplane": 58,
    "product-car": 59,
    "product-food": 60,
    "product-game": 61,
    "product-other": 62,
    "product-ship": 63,
    "product-software": 64,
    "product-train": 65,
    "product-weapon": 66,
}


class FewNERDConfig(datasets.BuilderConfig):
    """BuilderConfig for FewNERD"""

    def __init__(self, **kwargs):
        """BuilderConfig for FewNERD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FewNERDConfig, self).__init__(**kwargs)


class FewNERD(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        FewNERDConfig(name="supervised", description="Fully supervised setting."),
        FewNERDConfig(
            name="inter",
            description="Few-shot setting. Each file contains all 8 coarse " "types but different fine-grained types.",
        ),
        FewNERDConfig(name="intra", description="Few-shot setting. Randomly split by coarse type."),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.features.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.features.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "art",
                                "building",
                                "event",
                                "location",
                                "organization",
                                "other",
                                "person",
                                "product",
                            ]
                        )
                    ),
                    "fine_ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "art-broadcastprogram",
                                "art-film",
                                "art-music",
                                "art-other",
                                "art-painting",
                                "art-writtenart",
                                "building-airport",
                                "building-hospital",
                                "building-hotel",
                                "building-library",
                                "building-other",
                                "building-restaurant",
                                "building-sportsfacility",
                                "building-theater",
                                "event-attack/battle/war/militaryconflict",
                                "event-disaster",
                                "event-election",
                                "event-other",
                                "event-protest",
                                "event-sportsevent",
                                "location-GPE",
                                "location-bodiesofwater",
                                "location-island",
                                "location-mountain",
                                "location-other",
                                "location-park",
                                "location-road/railway/highway/transit",
                                "organization-company",
                                "organization-education",
                                "organization-government/governmentagency",
                                "organization-media/newspaper",
                                "organization-other",
                                "organization-politicalparty",
                                "organization-religion",
                                "organization-showorganization",
                                "organization-sportsleague",
                                "organization-sportsteam",
                                "other-astronomything",
                                "other-award",
                                "other-biologything",
                                "other-chemicalthing",
                                "other-currency",
                                "other-disease",
                                "other-educationaldegree",
                                "other-god",
                                "other-language",
                                "other-law",
                                "other-livingthing",
                                "other-medical",
                                "person-actor",
                                "person-artist/author",
                                "person-athlete",
                                "person-director",
                                "person-other",
                                "person-politician",
                                "person-scholar",
                                "person-soldier",
                                "product-airplane",
                                "product-car",
                                "product-food",
                                "product-game",
                                "product-other",
                                "product-ship",
                                "product-software",
                                "product-train",
                                "product-weapon",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://ningding97.github.io/fewnerd/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        url_to_download = dl_manager.download_and_extract(_URLs[self.config.name])
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(
                        url_to_download,
                        self.config.name,
                        "train.txt",
                    )
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(url_to_download, self.config.name, "dev.txt")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(url_to_download, self.config.name, "test.txt")},
            ),
        ]

    def _generate_examples(self, filepath=None):
        # check file type
        assert filepath[-4:] == ".txt"

        num_lines = sum(1 for _ in open(filepath, encoding="utf-8"))
        id = 0

        with open(filepath, "r", encoding="utf-8") as f:
            tokens, ner_tags, fine_ner_tags = [], [], []
            for line in tqdm(f, total=num_lines):
                line = line.strip().split()

                if line:
                    assert len(line) == 2
                    token, fine_ner_tag = line
                    ner_tag = fine_ner_tag.split("-")[0]

                    tokens.append(token)
                    ner_tags.append(NER_TAGS_DICT[ner_tag])
                    fine_ner_tags.append(FINE_NER_TAGS_DICT[fine_ner_tag])

                elif tokens:
                    # organize a record to be written into json
                    record = {
                        "tokens": tokens,
                        "id": str(id),
                        "ner_tags": ner_tags,
                        "fine_ner_tags": fine_ner_tags,
                    }
                    tokens, ner_tags, fine_ner_tags = [], [], []
                    id += 1
                    yield record["id"], record

            # take the last sentence
            if tokens:
                record = {
                    "tokens": tokens,
                    "id": str(id),
                    "ner_tags": ner_tags,
                    "fine_ner_tags": fine_ner_tags,
                }
                yield record["id"], record
