# Code-Graph-Backend

This is a FastAPI-based API that provides topic modeling and sentence embedding functionalities using BERT models. The API supports NER dataset like `few_nerd`. You can find the frontend for this project [here](https://github.com/jsp1999/Code-Graph-Frontend).

## Requirements

- Conda for package management
- Ensure you have an NVIDIA GPU with the required drivers:
  ```
  nvidia-smi
  ```
- Install the build-essential packages:
  ```
  sudo apt update
  sudo apt install build-essential
  ```
- PostgreSQL as database.
  - Follow the official PostgreSQL download instructions [here](https://www.postgresql.org/download/).
  - Use the following command to check if PostgreSQL is installed:
    ```
    which psql
    ```
- Setting Up PostgreSQL Database:
  - Access the PostgreSQL prompt:
    ```
    sudo -u postgres psql
    ```
    Create a new user (replace password):
    ```
    CREATE USER admin WITH PASSWORD 'password';
    ```
    Create Database:
    ```
    CREATE DATABASE codegraph OWNER admin;
    ```
    Grant Privileges:
    ```
    GRANT ALL PRIVILEGES ON DATABASE codegraph TO admin;
    ```
    Exit Prompt:
    ```
    \q
    ```
- Adapt environment ([env.json](https://github.com/bruehldev/Code-Graph-Backend/blob/master/env.json)) if necessary    

## Usage

1. Install the required packages by running the following command:
  - Using conda:
    ```
    conda env create --name CodeGraph --file environment.yml
    ```

2. Start the server using the following command:
```
cd src
uvicorn main:app --reload
```

3. Once the server is running, you can access the API at `http://localhost:8000`.

4. API Endpoints:
To access the API documentation use:

[http://localhost:8000/docs](http://localhost:8000/docs)

## Demo
The demo video can be found in the file TryDemoVideo.mp4.

Datasets used by us can be found in the datasets folder:
- movie datasets:
  - from:
  - movie dataset german labels (demo):
  - movie dataset english labels (demo):
  - huggingface model which worked well for us:
- few-ner datasets:
- bio-ner datasets:
- german datasets:

## Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, please submit a pull request.

## License

This project is licensed under the [Apache License, Version 2.0](LICENSE).


## Folder Structure
```
The project follows the following folder structure:
project/
├─ src/
│ ├─ main.py
│ ├─ module
│ │ └─ routes.py
│ │ └─ service.py
│ │ └─ schema.py
│ ├─ ...
├─ exported/ 
├─ data/
├─ .gitignore
├─ requirements.txt
```

- `exported/`: .Used to store generated data
- `.gitignore`: Specifies files and folders to exclude from version control.
- `main.py`: The main entry point of the application.
