# Code-Graph-Backend

This is a FastAPI-based API that provides topic modeling and sentence embedding functionalities using BERTopic and BERT models. The API supports two datasets: `fetch_20newsgroups` and `few_nerd`.

## Requirements

- Python 3.10.11
- Conda for package management

## Usage

1. Install the required packages by running the following command:
   - Using conda:
     ```
     conda env create --name CodeGraph --file environment.yml
     ```

2. Start the server using the following command:
```
uvicorn main:app --reload
```

3. Once the server is running, you can access the API at `http://localhost:8000`.

4. API Endpoints:
To access the API documentation use:

[http://localhost:8000/docs](http://localhost:8000/docs)


## Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, please submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).


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
- `data/`: Stores the datasets.
- `.gitignore`: Specifies files and folders to exclude from version control.
- `requirements.txt`: Lists the project dependencies.
- `main.py`: The main entry point of the application.