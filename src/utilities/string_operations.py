import hashlib


def generate_hash(model_dict: dict):
    # Convert dictionary to a string representation
    model_str = str(model_dict)

    # Generate hash using SHA256
    hash_object = hashlib.sha256(model_str.encode())
    print(model_str, hash_object.hexdigest())
    return hash_object.hexdigest()
