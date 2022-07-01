# Deploying a Machine Learning Model on Heroku with FastAPI  

This project aims to develop a classification model on publicly available Census Bureau data.
Tasks include: 
* create unit tests to monitor the model performance on various slices of the data
* deploy model using the FastAPI package and create API tests
* Both the dataset validations (eg. slice-validation) and the API tests will be incorporated into a CI/CD framework using GitHub Actions.

### Environment Set up  

* Download and install conda if you don’t have it already.
    * Use the supplied requirements file to create a new environment

    ```
    conda env create -f environment.yml
    ```
    * activate the env
    ```
    conda activate heroku
    ````

### Model  

* To run the model:
``` 
python starter/train_model.py
```

* or run the ML pipeline in a local server
```
python main.py
```

### Heroku deployment  

* Test the model live on Heroku by using the requests module to do one POST on the live API:

```
python heroku_api.py
```

### GitHub Actions  

- Runs pytest and flake8 on push and requires both to pass without error.
- Passing CI will result in automatic deployment on Heroku.