from db.models import Segment, Code, Sentence, Project, Dataset
from sqlalchemy import and_
from sqlalchemy.dialects.postgresql import insert
import time

from db.models import Code


def text_to_json(input_text, options=None):
    # Split text by double newlines to separate sentences
    sentences = input_text.strip().split(options.sentence_split)
    results = []

    for sentence in sentences:
        lines = sentence.split("\n")

        # Extract the words and labels from each line
        words = [line.split(options.split)[options.word_idx].strip() for line in lines]
        labels = [line.split(options.split)[options.label_idx].strip() for line in lines]

        # Construct the sentence
        text = " ".join(words)
        if options.type == "plain":
            entities = []
            start_pos = 0
            in_entity = False
            entity_label = ""
            for i, (word, label) in enumerate(zip(words, labels)):
                if label != "O":
                    # If we are already in an entity, we close it first
                    if not in_entity:
                        start = start_pos
                        entity_label = label
                        in_entity = True
                    elif entity_label != label:
                        entities.append({"start": start, "end": start_pos - 1, "label": entity_label})
                        start = start_pos
                        entity_label = label
                else:
                    if in_entity:
                        entities.append({"start": start, "end": start_pos - 1, "label": entity_label})
                        in_entity = False
                start_pos += len(word) + 1
            if in_entity:
                entities.append({"start": start, "end": start_pos - 1, "label": entity_label})

            results.append({"text": text, "entities": entities})

        if options.type == "B-I-O":
            entities = []
            start_pos = 0
            in_entity = False
            for i, (word, label) in enumerate(zip(words, labels)):
                if label.startswith("B-"):
                    # If we are already in an entity, we close it first
                    if in_entity:
                        entities.append({"start": start, "end": start_pos - 1, "label": entity_label})

                    # Start of a new entity
                    start = start_pos
                    entity_label = label[2:]
                    in_entity = True
                elif label.startswith("I-") and not in_entity:
                    # Continuation of an entity but we missed the beginning
                    start = start_pos
                    entity_label = label[2:].split(options.label_split)
                    in_entity = True
                elif label.startswith("O") and in_entity:
                    # End of the current entity
                    entities.append({"start": start, "end": start_pos - 1, "label": entity_label})
                    in_entity = False

                start_pos += len(word) + 1  # +1 for the space

            # If the last word was part of an entity
            if in_entity:
                entities.append({"start": start, "end": start_pos - 1, "label": entity_label})

            results.append({"text": text, "entities": entities})
    if options.label_split:
        for item in results:
            for entity in item["entities"]:
                entity["label"] = entity["label"].split(options.label_split)
    return {"data": results}


def add_data_to_db(project_id, database_name, json_data, session):
    start_time = time.time()
    project = session.query(Project).filter(and_(Project.project_id == project_id)).first()
    if not project:
        raise Exception("Project not found in the database!")

    dataset = Dataset(
        project_id=project.project_id,
        dataset_name=database_name,
    )
    session.add(dataset)
    session.commit()

    sentence_dicts = [{"text": item["text"], "dataset_id": dataset.dataset_id, "position_in_dataset": i} for i, item in enumerate(json_data["data"])]

    insert_stmt = insert(Sentence).values(sentence_dicts).returning(Sentence.sentence_id)
    sentence_ids = [row[0] for row in session.execute(insert_stmt).fetchall()]

    segment_dicts = []
    annotations_dict = {(a.text, a.parent_code_id): a for a in session.query(Code).filter_by(project_id=project.project_id).all()}
    new_annotations = set()

    for item, sentence_id in zip(json_data["data"], sentence_ids):
        for entity in item["entities"]:
            labels = entity["label"]
            if not isinstance(labels, list):
                labels = [labels]
            print(entity)
            last_id = None
            for i, label in enumerate(labels):
                annotation = annotations_dict.get((label, last_id))
                print(annotation)
                annotation_id = annotation.code_id if annotation else None
                # If the annotation doesn't exist, add it to the database and the dictionary
                if annotation is None or annotation.parent_code_id != last_id:
                    if (label, last_id) not in new_annotations:
                        new_annotation = Code(text=label, project_id=project.project_id, parent_code_id=last_id)
                        session.add(new_annotation)
                        session.commit()
                        annotation_id = new_annotation.code_id
                        annotations_dict[(label, last_id)] = new_annotation
                        new_annotations.add((label, last_id))

                last_id = annotation_id

            segment_dict = {
                "sentence_id": sentence_id,
                "text": item["text"][entity["start"] : entity["end"]],
                "start_position": entity["start"],
                "code_id": annotation_id,
            }
            print(segment_dict)

            segment_dicts.append(segment_dict)

    if segment_dicts:
        session.bulk_insert_mappings(Segment, segment_dicts)

    session.commit()
    session.close()

    segment_time = time.time()

    print(f"Added {len(json_data['data'])} sentences to the database.")
    print(f"Adding the data to the database took {time.time() - start_time} seconds.")
