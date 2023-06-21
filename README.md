# Code-Graph-Backend

This is a FastAPI-based API that provides topic modeling and sentence embedding functionalities using BERTopic and BERT models. The API supports two datasets: `fetch_20newsgroups` and `few_nerd`.

## Requirements

- Python 3.10.11
- Conda (or pip) for package management

## Usage

1. Install the required packages by running the following command:
   - Using conda:
     ```
     conda create --name CodeGraph --file requirements.txt
     ```
   - Using pip:
     ```
     pip install -r requirements.txt
     ```

2. Start the server using the following command:
```
uvicorn main:app --reload
```

3. Once the server is running, you can access the API at `http://localhost:8000`.

4. API Endpoints:
- **GET /load_model/{dataset}**: Load a BERTopic model for the specified dataset. Replace `{dataset}` with the desired dataset name ("fetch_20newsgroups" or "few_nerd").
- **GET /topicinfo/{dataset}**: Retrieve topic information for the specified dataset.
- **GET /embeddings/{dataset}**: Compute or retrieve BERT embeddings for the specified dataset.
- **GET /results/{dataset}**: Retrieve topic results (documents and their corresponding positions) for the specified dataset.


## Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, please submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).


## Folder Structure
```
The project follows the following folder structure:
project/
├─ api/
│ ├─ init.py
│ ├─ endpoints.py
│ └─ models.py
├─ data/
│ ├─ fetch_20newsgroups/
│ │ └─ ... (dataset files)
│ └─ few_nerd/
│ └─ train.txt
├─ embeddings/
├─ models/
├─ utils/
├─ .gitignore
├─ requirements.txt
└─ main.py
```

- `api/`: Contains the API-related code.
- `data/`: Stores the datasets.
- `embeddings/`: Used to store generated embeddings.
- `models/`: Used to save trained models.
- `utils/`: Contains utility or helper functions.
- `.gitignore`: Specifies files and folders to exclude from version control.
- `requirements.txt`: Lists the project dependencies.
- `main.py`: The main entry point of the application.

